#!/usr/bin/env python3
"""
Lloavece Kode - Supervised Fine-Tuning (SFT) Script

Treina o modelo com instruções de código em PT-BR usando LoRA.

Pipeline:
1. Carrega modelo base (pós-CPT ou Qwen3.8-27B)
2. Aplica LoRA (treina apenas adaptadores)
3. Treina com instruções de código em PT-BR
4. Salva checkpoint final
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
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
    use_qlora: bool = field(default=False, metadata={"help": "Use QLoRA (4-bit quantization)"})


@dataclass
class DataArguments:
    """Argumentos de dados"""
    train_data: str = field(metadata={"help": "Path to training data (JSONL)"})
    val_data: Optional[str] = field(default=None, metadata={"help": "Path to validation data (JSONL)"})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length"})
    train_on_inputs: bool = field(default=False, metadata={"help": "Train on input tokens too"})


@dataclass
class TrainingArgs(TrainingArguments):
    """Argumentos de treinamento"""
    output_dir: str = field(default="models/sft")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
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
    run_name: str = field(default="lloavece-sft")
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="paged_adamw_8bit")


class InstructionDataset(Dataset):
    """Dataset para instruções de código"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 4096,
        train_on_inputs: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_inputs = train_on_inputs
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt
        prompt = self._format_prompt(item)
        response = item.get('output', item.get('response', ''))
        
        # Tokenize
        full_text = prompt + response + self.tokenizer.eos_token
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Create labels (mask input tokens if train_on_inputs=False)
        labels = input_ids.clone()
        
        if not self.train_on_inputs:
            # Find where response starts
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)
            
            # Mask prompt tokens
            labels[:prompt_length] = -100
        
        # Mask padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _format_prompt(self, item: Dict) -> str:
        """Formata prompt para instrução de código"""
        
        # Formato 1: instruction + input
        if 'instruction' in item and 'input' in item:
            instruction = item['instruction']
            input_text = item['input']
            return f"### Instrução:\n{instruction}\n\n### Input:\n{input_text}\n\n### Resposta:\n"
        
        # Formato 2: apenas instruction
        elif 'instruction' in item:
            instruction = item['instruction']
            return f"### Instrução:\n{instruction}\n\n### Resposta:\n"
        
        # Formato 3: prompt + response
        elif 'prompt' in item:
            return item['prompt']
        
        # Formato 4: messages (chat format)
        elif 'messages' in item:
            messages = item['messages']
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    formatted += f"### Usuário:\n{content}\n\n"
                elif role == 'assistant':
                    formatted += f"### Assistente:\n{content}\n\n"
            return formatted
        
        # Formato 5: text completo
        elif 'text' in item:
            return item['text']
        
        else:
            raise ValueError(f"Unknown format in item: {item.keys()}")


def load_and_prepare_model(model_args: ModelArguments):
    """Carrega e prepara modelo para SFT"""
    
    logger.info(f"Loading model from: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if model_args.use_qlora:
        # QLoRA: 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        model = prepare_model_for_kbit_training(model)
        
    else:
        # Full precision or LoRA
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


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgs))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup output directory
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup wandb
    if training_args.report_to == "wandb":
        wandb.init(
            project="lloavece-kode",
            name=training_args.run_name,
            config={
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "training_args": vars(training_args)
            }
        )
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(model_args)
    
    # Load datasets
    logger.info(f"Loading training data from: {data_args.train_data}")
    train_dataset = InstructionDataset(
        data_args.train_data,
        tokenizer,
        data_args.max_seq_length,
        data_args.train_on_inputs
    )
    
    eval_dataset = None
    if data_args.val_data and Path(data_args.val_data).exists():
        logger.info(f"Loading validation data from: {data_args.val_data}")
        eval_dataset = InstructionDataset(
            data_args.val_data,
            tokenizer,
            data_args.max_seq_length,
            data_args.train_on_inputs
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("🚀 Starting SFT training...")
    logger.info(f"   Epochs: {training_args.num_train_epochs}")
    logger.info(f"   Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"   Learning rate: {training_args.learning_rate}")
    logger.info(f"   LoRA: {model_args.use_lora} (r={model_args.lora_r}, alpha={model_args.lora_alpha})")
    
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"💾 Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Merge LoRA if applicable
    if model_args.use_lora and not model_args.use_qlora:
        logger.info("🔀 Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        
        merged_output_dir = output_dir / "merged"
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        
        logger.info(f"✅ Merged model saved to: {merged_output_dir}")
    
    logger.info("✅ SFT training completed successfully!")
    
    if training_args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
