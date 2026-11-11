#!/usr/bin/env python3
"""
Processamento de Dados para Treinamento

Este script transforma os dados curados em formato pronto para treinamento,
incluindo tokenização, criação de sequências e preparação para CPT/SFT.

Funcionalidades:
1. Tokenização com tokenizer do modelo
2. Criação de sequências de treinamento
3. Packing de sequências curtas
4. Geração de arquivos de treino otimizados
5. Estatísticas de tokens e sequências
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers não instalado. Instale com: pip install transformers torch")


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/process_data.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig:
    """Configurações para processamento"""
    
    # Modelo e tokenizer
    model_name: str = "Qwen/Qwen3.8-27B"
    max_seq_length: int = 4096  # Comprimento máximo de sequência
    
    # Packing
    enable_packing: bool = True  # Empacota sequências curtas
    packing_separator: str = "\n\n"
    
    # Formatos de saída
    output_format: str = "jsonl"  # jsonl, parquet, arrow
    
    # Templates
    code_template: str = "```{language}\n{code}\n```\n\n{description}"
    qa_template: str = "Pergunta: {question}\n\nResposta: {answer}"
    doc_template: str = "# {title}\n\n{content}"


class DataProcessor:
    """Processa dados para treinamento"""
    
    def __init__(self, config: ProcessConfig):
        self.config = config
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'total_sequences': 0,
            'avg_tokens_per_sample': 0,
            'sequences_by_length': {},
        }
        
        if HAS_TRANSFORMERS:
            logger.info(f"🔧 Carregando tokenizer: {config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                padding_side='right'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None
            logger.warning("⚠️  Tokenizer não disponível (transformers não instalado)")
    
    def load_curated_data(self, input_dir: Path, split: str = 'train') -> pd.DataFrame:
        """Carrega dados curados"""
        logger.info(f"📂 Carregando dados curados ({split})...")
        
        jsonl_path = input_dir / f'{split}.jsonl'
        
        if not jsonl_path.exists():
            logger.error(f"❌ Arquivo não encontrado: {jsonl_path}")
            return pd.DataFrame()
        
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        self.stats['total_samples'] = len(df)
        logger.info(f"✓ Carregadas {len(df)} amostras")
        
        return df
    
    def format_content(self, row: Dict) -> str:
        """Formata conteúdo baseado no tipo"""
        content_type = row.get('content_type', 'general')
        content = row.get('content', '')
        
        if content_type == 'code':
            # Extrai linguagem se disponível
            language = row.get('language', 'python')
            description = row.get('description', '')
            
            if description:
                return self.config.code_template.format(
                    language=language,
                    code=content,
                    description=description
                )
            else:
                return f"```{language}\n{content}\n```"
        
        elif content_type == 'qa':
            question = row.get('question', content)
            answer = row.get('answer', '')
            
            if answer:
                return self.config.qa_template.format(
                    question=question,
                    answer=answer
                )
            else:
                return content
        
        elif content_type == 'documentation':
            title = row.get('title', 'Documentação')
            return self.config.doc_template.format(
                title=title,
                content=content
            )
        
        else:
            return content
    
    def tokenize_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Tokeniza amostras"""
        if not self.tokenizer:
            logger.warning("⚠️  Tokenizer não disponível, pulando tokenização")
            return []
        
        logger.info("🔤 Tokenizando amostras...")
        
        tokenized = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizando"):
            # Formata conteúdo
            formatted = self.format_content(row.to_dict())
            
            # Tokeniza
            tokens = self.tokenizer.encode(
                formatted,
                add_special_tokens=True,
                max_length=self.config.max_seq_length,
                truncation=True
            )
            
            tokenized.append({
                'tokens': tokens,
                'length': len(tokens),
                'content_type': row.get('content_type', 'general'),
                'formatted_content': formatted,
            })
            
            # Atualiza estatísticas
            self.stats['total_tokens'] += len(tokens)
        
        # Calcula média
        if tokenized:
            self.stats['avg_tokens_per_sample'] = (
                self.stats['total_tokens'] / len(tokenized)
            )
        
        logger.info(f"✓ Tokenizadas {len(tokenized)} amostras")
        logger.info(f"  • Total de tokens: {self.stats['total_tokens']:,}")
        logger.info(f"  • Média de tokens/amostra: {self.stats['avg_tokens_per_sample']:.1f}")
        
        return tokenized
    
    def pack_sequences(self, tokenized: List[Dict]) -> List[Dict]:
        """Empacota sequências curtas em sequências maiores"""
        if not self.config.enable_packing:
            logger.info("⏭️  Packing desabilitado")
            return tokenized
        
        logger.info("📦 Empacotando sequências...")
        
        packed = []
        current_tokens = []
        current_types = []
        
        # Ordena por comprimento (maior para menor) para melhor packing
        tokenized_sorted = sorted(tokenized, key=lambda x: x['length'], reverse=True)
        
        for sample in tqdm(tokenized_sorted, desc="Packing"):
            tokens = sample['tokens']
            
            # Se cabe na sequência atual
            if len(current_tokens) + len(tokens) + 1 <= self.config.max_seq_length:
                # Adiciona separador se não é a primeira
                if current_tokens:
                    separator_tokens = self.tokenizer.encode(
                        self.config.packing_separator,
                        add_special_tokens=False
                    )
                    current_tokens.extend(separator_tokens)
                
                current_tokens.extend(tokens)
                current_types.append(sample['content_type'])
            
            else:
                # Salva sequência atual se tem conteúdo
                if current_tokens:
                    packed.append({
                        'tokens': current_tokens,
                        'length': len(current_tokens),
                        'content_types': current_types,
                        'num_samples': len(current_types),
                    })
                
                # Inicia nova sequência
                current_tokens = tokens.copy()
                current_types = [sample['content_type']]
        
        # Adiciona última sequência
        if current_tokens:
            packed.append({
                'tokens': current_tokens,
                'length': len(current_tokens),
                'content_types': current_types,
                'num_samples': len(current_types),
            })
        
        self.stats['total_sequences'] = len(packed)
        
        logger.info(f"✓ Empacotadas em {len(packed)} sequências")
        logger.info(f"  • Sequências com múltiplas amostras: "
                   f"{sum(1 for s in packed if s['num_samples'] > 1)}")
        logger.info(f"  • Média de amostras por sequência: "
                   f"{sum(s['num_samples'] for s in packed) / len(packed):.2f}")
        
        return packed
    
    def create_training_data(self, sequences: List[Dict]) -> List[Dict]:
        """Cria dados de treinamento finais"""
        logger.info("🎯 Criando dados de treinamento...")
        
        training_data = []
        
        for seq in tqdm(sequences, desc="Criando training data"):
            tokens = seq['tokens']
            
            # Cria input_ids e labels (para causal LM, são iguais)
            # O modelo aprende a prever o próximo token
            training_data.append({
                'input_ids': tokens,
                'labels': tokens,  # Para causal LM
                'length': len(tokens),
                'content_types': seq.get('content_types', ['general']),
            })
        
        # Distribuição por comprimento
        lengths = [d['length'] for d in training_data]
        self.stats['sequences_by_length'] = {
            'min': int(min(lengths)) if lengths else 0,
            'max': int(max(lengths)) if lengths else 0,
            'mean': float(np.mean(lengths)) if lengths else 0,
            'median': float(np.median(lengths)) if lengths else 0,
            'std': float(np.std(lengths)) if lengths else 0,
        }
        
        logger.info(f"✓ Criadas {len(training_data)} sequências de treinamento")
        logger.info(f"  • Comprimento: min={self.stats['sequences_by_length']['min']}, "
                   f"max={self.stats['sequences_by_length']['max']}, "
                   f"mean={self.stats['sequences_by_length']['mean']:.1f}")
        
        return training_data
    
    def save_training_data(self, data: List[Dict], output_path: Path, split: str):
        """Salva dados de treinamento"""
        logger.info(f"💾 Salvando dados de treinamento ({split})...")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSONL
        jsonl_path = output_path / f'{split}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"  ✓ Salvo {jsonl_path} ({len(data)} sequências)")
        
        # Estatísticas
        stats_path = output_path / f'{split}_process_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'split': split,
                'timestamp': datetime.now().isoformat(),
                'num_sequences': len(data),
                'total_tokens': sum(d['length'] for d in data),
                'stats': self.stats,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Salvo {stats_path}")
    
    def process(self, input_dir: Path, output_dir: Path, split: str = 'train'):
        """Pipeline completo de processamento"""
        logger.info("=" * 60)
        logger.info(f"🚀 Processando dados ({split})")
        logger.info("=" * 60)
        
        # 1. Carrega dados curados
        df = self.load_curated_data(input_dir, split)
        
        if len(df) == 0:
            logger.error("❌ Nenhum dado para processar!")
            return
        
        # 2. Tokeniza
        tokenized = self.tokenize_samples(df)
        
        if not tokenized:
            logger.error("❌ Falha na tokenização!")
            return
        
        # 3. Empacota sequências
        sequences = self.pack_sequences(tokenized)
        
        # 4. Cria dados de treinamento
        training_data = self.create_training_data(sequences)
        
        # 5. Salva
        self.save_training_data(training_data, output_dir, split)
        
        logger.info("=" * 60)
        logger.info(f"✅ Processamento completo ({split})!")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Processamento de dados para treinamento',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/processed'),
        help='Diretório de entrada com dados curados'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training'),
        help='Diretório de saída com dados processados'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='Split para processar'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3.8-27B',
        help='Nome do modelo para tokenizer'
    )
    
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=4096,
        help='Comprimento máximo de sequência'
    )
    
    parser.add_argument(
        '--no-packing',
        action='store_true',
        help='Desabilita packing de sequências'
    )
    
    parser.add_argument(
        '--all-splits',
        action='store_true',
        help='Processa todos os splits (train, validation, test)'
    )
    
    args = parser.parse_args()
    
    # Cria diretório de logs
    Path('logs').mkdir(exist_ok=True)
    
    # Configuração
    config = ProcessConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        enable_packing=not args.no_packing,
    )
    
    # Processa
    processor = DataProcessor(config)
    
    if args.all_splits:
        for split in ['train', 'validation', 'test']:
            processor.process(args.input_dir, args.output_dir, split)
    else:
        processor.process(args.input_dir, args.output_dir, args.split)
    
    logger.info("\n🎉 Processamento concluído com sucesso!")
    logger.info(f"📁 Dados prontos para treinamento em: {args.output_dir}")


if __name__ == '__main__':
    main()
