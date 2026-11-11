#!/usr/bin/env python3
"""
Continued Pretraining (CPT) para Lloavece Kode

Este script realiza o continued pretraining do Qwen3.8-27B em dados
de código e português para especializar o modelo.

Funcionalidades:
- Treinamento distribuído com DeepSpeed
- Suporte a LoRA/QLoRA
- Logging com Weights & Biases
- Checkpoints automáticos
- Avaliação durante treino
- Resume de treino interrompido
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )
    from transformers import Trainer, TrainingArguments
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("❌ transformers não instalado. Instale com: pip install transformers")

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("⚠️  deepspeed não instalado. Treino distribuído desabilitado.")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  peft não instalado. LoRA desabilitado. Instale com: pip install peft")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb não instalado. Logging W&B desabilitado.")


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/train_cpt.log')
    ]
)
logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset para código tokenizado"""
    
    def __init__(self, data_path: Path, max_length: int = 4096):
        logger.info(f"📂 Carregando dataset: {data_path}")
        
        self.data = []
        self.max_length = max_length
        
        # Carrega dados
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
        
        logger.info(f"✓ Carregadas {len(self.data)} sequências")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Trunca se necessário
        input_ids = item['input_ids'][:self.max_length]
        labels = item['labels'][:self.max_length]
        
        # Padding se necessário (geralmente não precisa com packing)
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length
            labels = labels + [-100] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
):
    """Carrega modelo e tokenizer"""
    logger.info(f"🔧 Carregando modelo: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right',
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configurações de carregamento
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if not load_in_4bit else None,
    }
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # Modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Aplica LoRA se solicitado
    if use_lora and HAS_PEFT:
        logger.info("🔧 Aplicando LoRA...")
        
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def create_deepspeed_config(
    output_dir: Path,
    gradient_accumulation_steps: int = 8,
    zero_stage: int = 2,
):
    """Cria configuração do DeepSpeed"""
    config = {
        "fp16": {
            "enabled": False,
        },
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu" if zero_stage == 3 else "none",
            },
            "offload_param": {
                "device": "cpu" if zero_stage == 3 else "none",
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    
    config_path = output_dir / "deepspeed_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(config_path)


def train(args):
    """Função principal de treinamento"""
    
    # Verifica dependências
    if not HAS_TRANSFORMERS:
        logger.error("❌ transformers não instalado!")
        return
    
    # Cria diretórios
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Configura W&B
    if HAS_WANDB and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )
    
    # Carrega dados
    logger.info("📊 Carregando dados...")
    train_dataset = CodeDataset(
        Path(args.train_data),
        max_length=args.max_seq_length
    )
    
    val_dataset = None
    if args.val_data and Path(args.val_data).exists():
        val_dataset = CodeDataset(
            Path(args.val_data),
            max_length=args.max_seq_length
        )
    
    # Carrega modelo
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Configura DeepSpeed
    deepspeed_config = None
    if HAS_DEEPSPEED and args.use_deepspeed:
        deepspeed_config = create_deepspeed_config(
            Path(args.output_dir),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            zero_stage=args.deepspeed_zero_stage,
        )
        logger.info(f"✓ DeepSpeed configurado: {deepspeed_config}")
    
    # Argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=args.save_total_limit,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to="wandb" if (HAS_WANDB and args.use_wandb) else "none",
        deepspeed=deepspeed_config,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    
    # Trainer
    logger.info("🏋️  Iniciando treinamento...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Resume de checkpoint se especificado
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if Path(args.resume_from_checkpoint).exists():
            resume_from_checkpoint = args.resume_from_checkpoint
            logger.info(f"🔄 Resumindo de: {resume_from_checkpoint}")
    
    # Treina
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO TREINAMENTO")
    logger.info("=" * 60)
    logger.info(f"  • Modelo: {args.model_name}")
    logger.info(f"  • LoRA: {args.use_lora}")
    logger.info(f"  • Batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  • Learning rate: {args.learning_rate}")
    logger.info(f"  • Épocas: {args.num_epochs}")
    logger.info(f"  • Sequências: {len(train_dataset)}")
    logger.info("=" * 60)
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Salva modelo final
    logger.info("💾 Salvando modelo final...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Salva métricas
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Avaliação final
    if val_dataset:
        logger.info("📊 Avaliação final...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info("=" * 60)
    logger.info("✅ TREINAMENTO COMPLETO!")
    logger.info("=" * 60)
    logger.info(f"📁 Modelo salvo em: {args.output_dir}")
    
    # Finaliza W&B
    if HAS_WANDB and args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description='Continued Pretraining para Lloavece Kode',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dados
    parser.add_argument('--train-data', type=Path, required=True,
                       help='Arquivo JSONL de treino')
    parser.add_argument('--val-data', type=Path, default=None,
                       help='Arquivo JSONL de validação')
    
    # Modelo
    parser.add_argument('--model-name', type=str,
                       default='Qwen/Qwen3.8-27B',
                       help='Nome do modelo base')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('models/cpt'),
                       help='Diretório de saída')
    
    # LoRA
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Usa LoRA')
    parser.add_argument('--no-lora', action='store_false', dest='use_lora',
                       help='Desabilita LoRA')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='Rank do LoRA')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='Alpha do LoRA')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='Dropout do LoRA')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Carrega modelo em 4-bit (QLoRA)')
    
    # Treinamento
    parser.add_argument('--num-epochs', type=int, default=2,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size por GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='Passos de acumulação de gradiente')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Passos de warmup')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                       choices=['linear', 'cosine', 'constant'],
                       help='Tipo de scheduler')
    parser.add_argument('--max-seq-length', type=int, default=4096,
                       help='Comprimento máximo de sequência')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Norma máxima do gradiente')
    parser.add_argument('--optim', type=str, default='adamw_torch',
                       help='Otimizador')
    
    # DeepSpeed
    parser.add_argument('--use-deepspeed', action='store_true', default=True,
                       help='Usa DeepSpeed')
    parser.add_argument('--no-deepspeed', action='store_false', dest='use_deepspeed',
                       help='Desabilita DeepSpeed')
    parser.add_argument('--deepspeed-zero-stage', type=int, default=2,
                       choices=[1, 2, 3],
                       help='Estágio do DeepSpeed ZeRO')
    
    # Otimizações
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True,
                       help='Usa gradient checkpointing')
    parser.add_argument('--no-gradient-checkpointing', action='store_false',
                       dest='gradient_checkpointing',
                       help='Desabilita gradient checkpointing')
    
    # Logging
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Passos entre logs')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='Passos entre saves')
    parser.add_argument('--eval-steps', type=int, default=500,
                       help='Passos entre avaliações')
    parser.add_argument('--save-total-limit', type=int, default=3,
                       help='Número máximo de checkpoints')
    
    # W&B
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Usa Weights & Biases')
    parser.add_argument('--no-wandb', action='store_false', dest='use_wandb',
                       help='Desabilita W&B')
    parser.add_argument('--wandb-project', type=str, default='lloavece-kode',
                       help='Nome do projeto W&B')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Nome do run W&B')
    
    # Outros
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Número de workers do dataloader')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reprodutibilidade')
    parser.add_argument('--resume-from-checkpoint', type=Path, default=None,
                       help='Resume de checkpoint')
    
    args = parser.parse_args()
    
    # Cria diretório de logs
    Path('logs').mkdir(exist_ok=True)
    
    # Treina
    train(args)


if __name__ == '__main__':
    main()
