#!/usr/bin/env python3
"""
Lloavece Kode - Direct Preference Optimization (DPO) Script

Treina o modelo com pares de preferências (melhor vs pior resposta)
usando DPO para alinhar com qualidade de código.

Pipeline:
1. Carrega modelo base (pós-SFT)
2. Aplica LoRA (opcional)
3. Treina com pares (chosen, rejected)
4. Salva checkpoint final
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset as HFDataset
import wandb
from tqdm import tqdm


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Argumentos do modelo"""
    model_name_or_path: str = field(
        metadata={"help": "Path to base model (local or HuggingFace)"}
    )
    use_lora: bool = field(default=True, metadata={"help": "Use LoRA for training"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )


@dataclass
class DataArguments:
    """Argumentos de dados"""
    train_data: str = field(metadata={"help": "Path to training data (JSONL)"})
    val_data: Optional[str] = field(default=None, metadata={"help": "Path to validation data (JSONL)"})
    max_prompt_length: int = field(default=2048, metadata={"help": "Maximum prompt length"})
    max_completion_length: int = field(default=2048, metadata={"help": "Maximum completion length"})


@dataclass
class DPOArguments(DPOConfig):
    """Argumentos de DPO"""
    output_dir: str = field(default="models/dpo")
    num_train_epochs: float = field(default=2.0)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    report_to: str = field(default="wandb")
    run_name: str = field(default="lloavece-dpo")
    gradient_checkpointing: bool = field(default=True)
    beta: float = field(default=0.1, metadata={"help": "DPO beta parameter"})
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type"})
    max_length: int = field(default=4096, metadata={"help": "Maximum sequence length"})


class PreferenceDataset(Dataset):
    """Dataset para pares de preferências"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_prompt_length: int = 2048,
        max_completion_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} preference pairs from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Formato esperado: {prompt, chosen, rejected}
        # ou {instruction, chosen, rejected}
        
        if 'prompt' in item:
            prompt = item['prompt']
        elif 'instruction' in item:
            prompt = self._format_instruction(item)
        elif 'messages' in item:
            prompt = self._format_messages(item['messages'])
        else:
            raise ValueError(f"Unknown format: {item.keys()}")
        
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None
        )
        
        chosen_tokens = self.tokenizer(
            chosen + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_completion_length,
            padding=False,
            return_tensors=None
        )
        
        rejected_tokens = self.tokenizer(
            rejected + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_completion_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            "prompt": prompt,
            "chosen": chosen + self.tokenizer.eos_token,
            "rejected": rejected + self.tokenizer.eos_token,
        }
    
    def _format_instruction(self, item: Dict) -> str:
        """Formata instrução para prompt"""
        instruction = item['instruction']
        input_text = item.get('input', '')
        
        if input_text:
            return f"### Instrução:\n{instruction}\n\n### Input:\n{input_text}\n\n### Resposta:\n"
        else:
            return f"### Instrução:\n{instruction}\n\n### Resposta:\n"
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Formata mensagens para prompt"""
        formatted = ""
        for msg in messages:
            if msg['role'] == 'user':
                formatted += f"### Usuário:\n{msg['content']}\n\n"
            elif msg['role'] == 'assistant':
                # Não inclui a última resposta do assistente no prompt
                pass
        return formatted + "### Assistente:\n"


def load_and_prepare_model(model_args: ModelArguments):
    """Carrega e prepara modelo para DPO"""
    
    logger.info(f"Loading model from: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Apply LoRA if requested
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def create_preference_pairs_from_data(data_path: str, output_path: str):
    """
    Cria pares de preferências a partir de dados brutos
    
    Formato de entrada esperado:
    {
        "prompt": "...",
        "responses": ["response1", "response2", ...],
        "scores": [0.9, 0.7, ...]  # ou "rankings": [1, 2, ...]
    }
    
    Formato de saída:
    {
        "prompt": "...",
        "chosen": "melhor resposta",
        "rejected": "pior resposta"
    }
    """
    logger.info(f"Creating preference pairs from: {data_path}")
    
    pairs = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            if 'prompt' not in item or 'responses' not in item:
                continue
            
            prompt = item['prompt']
            responses = item['responses']
            
            # Se tem scores, usa para ordenar
            if 'scores' in item:
                scores = item['scores']
                ranked = sorted(zip(responses, scores), key=lambda x: x[1], reverse=True)
            # Se tem rankings, usa para ordenar
            elif 'rankings' in item:
                rankings = item['rankings']
                ranked = sorted(zip(responses, rankings), key=lambda x: x[1])
            else:
                # Assume que as respostas já estão ordenadas
                ranked = [(r, i) for i, r in enumerate(responses)]
            
            # Cria par (melhor, pior)
            if len(ranked) >= 2:
                chosen = ranked[0][0]  # Melhor
                rejected = ranked[-1][0]  # Pior
                
                # Só cria par se diferença for significativa
                if 'scores' in item:
                    score_diff = ranked[0][1] - ranked[-1][1]
                    if score_diff > 0.1:  # Threshold mínimo
                        pairs.append({
                            'prompt': prompt,
                            'chosen': chosen,
                            'rejected': rejected
                        })
                else:
                    pairs.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected
                    })
    
    # Salva pares
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Created {len(pairs)} preference pairs -> {output_path}")
    return pairs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, dpo_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, dpo_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(dpo_args.seed)
    
    # Setup output directory
    output_dir = Path(dpo_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup wandb
    if dpo_args.report_to == "wandb":
        wandb.init(
            project="lloavece-kode",
            name=dpo_args.run_name,
            config={
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "dpo_args": vars(dpo_args)
            }
        )
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(model_args)
    
    # Create reference model (frozen copy)
    logger.info("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Load datasets
    logger.info(f"Loading training data from: {data_args.train_data}")
    
    # Carrega como HuggingFace Dataset
    train_dataset = HFDataset.from_json(data_args.train_data)
    
    eval_dataset = None
    if data_args.val_data and Path(data_args.val_data).exists():
        logger.info(f"Loading validation data from: {data_args.val_data}")
        eval_dataset = HFDataset.from_json(data_args.val_data)
    
    # Initialize DPO trainer
    logger.info("🚀 Initializing DPO trainer...")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        ) if model_args.use_lora else None,
        beta=dpo_args.beta,
        loss_type=dpo_args.loss_type,
        max_prompt_length=data_args.max_prompt_length,
        max_length=dpo_args.max_length,
    )
    
    # Train
    logger.info("🚀 Starting DPO training...")
    logger.info(f"   Epochs: {dpo_args.num_train_epochs}")
    logger.info(f"   Batch size: {dpo_args.per_device_train_batch_size}")
    logger.info(f"   Gradient accumulation: {dpo_args.gradient_accumulation_steps}")
    logger.info(f"   Learning rate: {dpo_args.learning_rate}")
    logger.info(f"   Beta: {dpo_args.beta}")
    logger.info(f"   LoRA: {model_args.use_lora} (r={model_args.lora_r}, alpha={model_args.lora_alpha})")
    
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"💾 Saving model to: {dpo_args.output_dir}")
    trainer.save_model(dpo_args.output_dir)
    tokenizer.save_pretrained(dpo_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Merge LoRA if applicable
    if model_args.use_lora:
        logger.info("🔀 Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        
        merged_output_dir = output_dir / "merged"
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        
        logger.info(f"✅ Merged model saved to: {merged_output_dir}")
    
    logger.info("✅ DPO training completed successfully!")
    
    if dpo_args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
