#!/usr/bin/env python3
"""
Curadoria e Processamento de Dados para Lloavece Kode

Este script processa os dados brutos coletados e aplica múltiplos filtros
para criar um dataset de alta qualidade para treinamento.

Pipeline de filtros:
1. Deduplicação (MinHash + exact match)
2. Filtragem por qualidade (perplexidade, comprimento, etc)
3. Remoção de conteúdo tóxico/spam
4. Detecção de linguagem (PT-BR + código)
5. Formatação para treinamento
6. Split train/validation/test
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Tenta importar bibliotecas opcionais
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    print("⚠️  datasketch não instalado. Deduplicação aproximada desabilitada.")
    print("   Instale com: pip install datasketch")

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("⚠️  langdetect não instalado. Detecção de idioma desabilitada.")
    print("   Instale com: pip install langdetect")


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/curate_data.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CurateConfig:
    """Configurações para curadoria de dados"""
    
    # Deduplicação
    similarity_threshold: float = 0.8  # Threshold para MinHash
    num_perm: int = 128  # Número de permutações para MinHash
    
    # Qualidade de texto
    min_length: int = 50  # Mínimo de caracteres
    max_length: int = 100000  # Máximo de caracteres
    min_words: int = 10  # Mínimo de palavras
    
    # Código
    min_code_lines: int = 5  # Mínimo de linhas de código
    max_code_lines: int = 1000  # Máximo de linhas de código
    
    # Linguagem
    target_languages: List[str] = None  # ['pt', 'en']
    code_extensions: List[str] = None
    
    # Proporções
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    
    # Qualidade
    max_repetition_ratio: float = 0.3  # Máximo de linhas repetidas
    min_unique_words_ratio: float = 0.2  # Mínimo de palavras únicas
    
    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = ['pt', 'en']
        if self.code_extensions is None:
            self.code_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
                '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift',
                '.kt', '.scala', '.r', '.sql', '.sh', '.bash', '.md'
            ]


class DataCurator:
    """Classe principal para curadoria de dados"""
    
    def __init__(self, config: CurateConfig):
        self.config = config
        self.stats = {
            'total_samples': 0,
            'duplicates_removed': 0,
            'quality_filtered': 0,
            'language_filtered': 0,
            'toxic_filtered': 0,
            'final_samples': 0,
        }
        
    def load_data(self, input_dir: Path) -> pd.DataFrame:
        """Carrega dados brutos de múltiplas fontes"""
        logger.info(f"📂 Carregando dados de {input_dir}")
        
        all_data = []
        
        # Carrega arquivos JSON
        for json_file in input_dir.glob('**/*.json'):
            if 'stats' in json_file.name.lower():
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                logger.info(f"✓ Carregado {json_file}")
            except Exception as e:
                logger.warning(f"✗ Erro ao carregar {json_file}: {e}")
        
        # Carrega arquivos JSONL
        for jsonl_file in input_dir.glob('**/*.jsonl'):
            if 'stats' in jsonl_file.name.lower():
                continue
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
                logger.info(f"✓ Carregado {jsonl_file}")
            except Exception as e:
                logger.warning(f"✗ Erro ao carregar {jsonl_file}: {e}")
        
        # Normaliza itens para garantir coluna 'content'
        normalized_data = []
        for item in all_data:
            if not isinstance(item, dict):
                continue
            if 'stats' in item and ('timestamp' in item or 'repositories' in item):
                continue
            
            item_copy = dict(item)
            if 'content' in item_copy and item_copy['content']:
                normalized_data.append(item_copy)
            elif 'code' in item_copy and item_copy['code']:
                item_copy['content'] = item_copy['code']
                if 'content_type' not in item_copy:
                    item_copy['content_type'] = 'code'
                normalized_data.append(item_copy)
            elif 'question_body' in item_copy or 'answer_body' in item_copy:
                title = item_copy.get('question_title', '')
                q_body = item_copy.get('question_body', '')
                a_body = item_copy.get('answer_body', '')
                item_copy['content'] = f"# {title}\n\n{q_body}\n\n## Resposta:\n{a_body}".strip()
                item_copy['question'] = f"{title}\n{q_body}".strip()
                item_copy['answer'] = a_body
                if 'content_type' not in item_copy:
                    item_copy['content_type'] = 'qa'
                normalized_data.append(item_copy)
            elif 'name' in item_copy and ('description' in item_copy or 'url' in item_copy):
                name = item_copy.get('name', '')
                lang = item_copy.get('language') or ''
                desc = item_copy.get('description') or ''
                item_copy['content'] = f"# Repositório: {name}\nLinguagem: {lang}\n\nDescrição:\n{desc}".strip()
                if 'content_type' not in item_copy:
                    item_copy['content_type'] = 'documentation'
                normalized_data.append(item_copy)
            elif 'text' in item_copy and item_copy['text']:
                item_copy['content'] = item_copy['text']
                normalized_data.append(item_copy)
        
        df = pd.DataFrame(normalized_data)
        if 'content' not in df.columns:
            df['content'] = pd.Series(dtype=str)
        else:
            df['content'] = df['content'].fillna('').astype(str)
        
        self.stats['total_samples'] = len(df)
        logger.info(f"📊 Total de amostras carregadas: {len(df)}")
        
        return df
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas usando MinHash + exact match"""
        logger.info("🔍 Removendo duplicatas...")
        
        initial_count = len(df)
        
        # 1. Exact deduplication (hash do conteúdo)
        logger.info("  → Deduplicação exata...")
        df['content_hash'] = df['content'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )
        df = df.drop_duplicates(subset=['content_hash'])
        exact_removed = initial_count - len(df)
        logger.info(f"    ✓ Removidas {exact_removed} duplicatas exatas")
        
        # 2. Approximate deduplication (MinHash) se disponível
        if HAS_DATASKETCH and len(df) > 0:
            logger.info("  → Deduplicação aproximada (MinHash)...")
            
            # Cria MinHash para cada documento
            minhashes = []
            for text in tqdm(df['content'], desc="Calculando MinHash"):
                m = MinHash(num_perm=self.config.num_perm)
                for word in str(text).split():
                    m.update(word.encode('utf-8'))
                minhashes.append(m)
            
            # Cria LSH index
            lsh = MinHashLSH(threshold=self.config.similarity_threshold, num_perm=self.config.num_perm)
            
            duplicates = set()
            for i, m in enumerate(tqdm(minhashes, desc="Detectando similares")):
                # Verifica se já é duplicata
                if i in duplicates:
                    continue
                
                # Busca similares
                try:
                    result = lsh.query(m)
                    for j in result:
                        if j != i and j not in duplicates:
                            duplicates.add(j)
                    lsh.insert(str(i), m)
                except:
                    lsh.insert(str(i), m)
            
            # Remove duplicatas
            df = df.drop(index=list(duplicates)).reset_index(drop=True)
            approx_removed = len(minhashes) - len(df)
            logger.info(f"    ✓ Removidas {approx_removed} duplicatas aproximadas")
            
            self.stats['duplicates_removed'] = exact_removed + approx_removed
        else:
            self.stats['duplicates_removed'] = exact_removed
        
        df = df.drop(columns=['content_hash'])
        logger.info(f"✓ Deduplicação completa. Restam {len(df)} amostras")
        
        return df
    
    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra por qualidade do conteúdo"""
        logger.info("🎯 Filtrando por qualidade...")
        
        initial_count = len(df)
        
        # 1. Comprimento
        logger.info("  → Filtrando por comprimento...")
        df['length'] = df['content'].apply(len)
        df = df[
            (df['length'] >= self.config.min_length) &
            (df['length'] <= self.config.max_length)
        ]
        
        # 2. Número de palavras
        logger.info("  → Filtrando por número de palavras...")
        df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
        df = df[df['word_count'] >= self.config.min_words]
        
        # 3. Ratio de palavras únicas
        logger.info("  → Filtrando por diversidade de vocabulário...")
        df['unique_words_ratio'] = df['content'].apply(
            lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1)
        )
        df = df[df['unique_words_ratio'] >= self.config.min_unique_words_ratio]
        
        # 4. Repetição de linhas
        logger.info("  → Filtrando por repetição de linhas...")
        df['repetition_ratio'] = df['content'].apply(self._calculate_repetition)
        df = df[df['repetition_ratio'] <= self.config.max_repetition_ratio]
        
        # 5. Detecção de spam/boilerplate
        logger.info("  → Filtrando spam e boilerplate...")
        df = df[~df['content'].apply(self._is_spam)]
        
        quality_removed = initial_count - len(df)
        self.stats['quality_filtered'] = quality_removed
        
        # Limpa colunas temporárias
        df = df.drop(columns=['length', 'word_count', 'unique_words_ratio', 'repetition_ratio'])
        
        logger.info(f"✓ Qualidade filtrada. Removidas {quality_removed} amostras, restam {len(df)}")
        
        return df
    
    def _calculate_repetition(self, text: str) -> float:
        """Calcula ratio de linhas repetidas"""
        lines = str(text).split('\n')
        if len(lines) <= 1:
            return 0.0
        
        unique_lines = set(lines)
        return 1.0 - (len(unique_lines) / len(lines))
    
    def _is_spam(self, text: str) -> bool:
        """Detecta spam e boilerplate"""
        text_lower = str(text).lower()
        
        # Padrões de spam
        spam_patterns = [
            r'click here',
            r'subscribe now',
            r'buy now',
            r'free money',
            r'lorem ipsum',
            r'cookie policy',
            r'privacy policy',
            r'terms of service',
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Muito curto com muitas URLs
        url_count = len(re.findall(r'http[s]?://', text))
        if url_count > 3 and len(text) < 200:
            return True
        
        return False
    
    def filter_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra por idioma"""
        if not HAS_LANGDETECT:
            logger.warning("⚠️  Detecção de idioma desabilitada (langdetect não instalado)")
            return df
        
        logger.info("🌐 Filtrando por idioma...")
        
        initial_count = len(df)
        
        # Detecta idioma
        logger.info("  → Detectando idiomas...")
        detected_langs = []
        
        for text in tqdm(df['content'], desc="Detectando idioma"):
            try:
                lang = detect(str(text)[:1000])  # Primeiros 1000 chars
                detected_langs.append(lang)
            except LangDetectException:
                detected_langs.append('unknown')
        
        df['language'] = detected_langs
        
        # Filtra por idiomas alvo
        df = df[df['language'].isin(self.config.target_languages + ['unknown'])]
        
        language_removed = initial_count - len(df)
        self.stats['language_filtered'] = language_removed
        
        df = df.drop(columns=['language'])
        
        logger.info(f"✓ Idioma filtrado. Removidas {language_removed} amostras, restam {len(df)}")
        
        return df
    
    def filter_toxic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra conteúdo tóxico"""
        logger.info("🚫 Filtrando conteúdo tóxico...")
        
        initial_count = len(df)
        
        # Lista de palavras tóxicas (exemplo básico)
        # Em produção, usar modelo de classificação
        toxic_patterns = [
            r'\b(hate|kill|die|stupid|idiot)\b',
            r'\b(merda|bosta|caralho|porra)\b',  # PT-BR
        ]
        
        logger.info("  → Aplicando filtros de toxicidade...")
        mask = df['content'].apply(
            lambda x: not any(re.search(p, str(x).lower()) for p in toxic_patterns)
        )
        df = df[mask]
        
        toxic_removed = initial_count - len(df)
        self.stats['toxic_filtered'] = toxic_removed
        
        logger.info(f"✓ Toxicidade filtrada. Removidas {toxic_removed} amostras, restam {len(df)}")
        
        return df
    
    def format_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formata dados para treinamento"""
        logger.info("📝 Formatando para treinamento...")
        
        # Adiciona campos necessários
        df['id'] = range(len(df))
        df['created_at'] = datetime.now().isoformat()
        
        # Classifica tipo de conteúdo respeitando tipos pré-existentes
        if 'content_type' not in df.columns:
            df['content_type'] = df['content'].apply(self._classify_content)
        else:
            classified = df['content'].apply(self._classify_content)
            df['content_type'] = df['content_type'].fillna('').astype(str)
            df['content_type'] = df['content_type'].replace('', np.nan).fillna(classified)
        
        logger.info(f"✓ Formatação completa")
        
        return df
    
    def _classify_content(self, text: str) -> str:
        """Classifica o tipo de conteúdo"""
        text = str(text)
        
        # Código
        code_indicators = ['def ', 'function ', 'class ', 'import ', '```']
        if any(indicator in text for indicator in code_indicators):
            return 'code'
        
        # Documentação
        doc_indicators = ['# ', '## ', '### ', 'documentação', 'documentation']
        if any(indicator in text.lower() for indicator in doc_indicators):
            return 'documentation'
        
        # QA
        qa_indicators = ['?', 'pergunta', 'question', 'answer', 'resposta']
        if any(indicator in text for indicator in qa_indicators):
            return 'qa'
        
        return 'general'
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide em train/validation/test"""
        logger.info("✂️  Dividindo dataset...")
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        if n <= 1:
            train_df = df.copy()
            val_df = df.copy()
            test_df = df.copy()
        elif n == 2:
            train_df = df.iloc[:1]
            val_df = df.iloc[1:2]
            test_df = df.iloc[1:2]
        else:
            train_end = max(1, int(n * self.config.train_ratio))
            val_end = min(n, train_end + max(1, int(n * self.config.val_ratio)))
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            if len(test_df) == 0:
                test_df = val_df.copy()
        
        logger.info(f"✓ Dataset dividido:")
        logger.info(f"  • Train: {len(train_df)} amostras ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  • Validation: {len(val_df)} amostras ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  • Test: {len(test_df)} amostras ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_dataset(self, df: pd.DataFrame, output_path: Path, split: str):
        """Salva dataset em múltiplos formatos"""
        logger.info(f"💾 Salvando dataset {split}...")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSONL (para treinamento)
        jsonl_path = output_path / f'{split}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
        logger.info(f"  ✓ Salvo {jsonl_path}")
        
        # Parquet (para análise)
        parquet_path = output_path / f'{split}.parquet'
        df.to_parquet(parquet_path, index=False)
        logger.info(f"  ✓ Salvo {parquet_path}")
        
        # Estatísticas
        stats_path = output_path / f'{split}_stats.json'
        stats = {
            'split': split,
            'num_samples': len(df),
            'total_tokens': int(df['content'].apply(lambda x: len(str(x).split())).sum()),
            'avg_length': float(df['content'].apply(len).mean()),
            'content_types': df['content_type'].value_counts().to_dict(),
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Salvo {stats_path}")
    
    def curate(self, input_dir: Path, output_dir: Path):
        """Pipeline completo de curadoria"""
        logger.info("=" * 60)
        logger.info("🚀 Iniciando pipeline de curadoria de dados")
        logger.info("=" * 60)
        
        # 1. Carrega dados
        df = self.load_data(input_dir)
        
        if len(df) == 0:
            logger.error("❌ Nenhum dado encontrado!")
            return
        
        # 2. Deduplicação
        df = self.deduplicate(df)
        
        # 3. Filtros de qualidade
        df = self.filter_quality(df)
        
        # 4. Filtro de idioma
        df = self.filter_language(df)
        
        # 5. Filtro de toxicidade
        df = self.filter_toxic(df)
        
        # 6. Formatação
        df = self.format_for_training(df)
        
        self.stats['final_samples'] = len(df)
        
        # 7. Split
        train_df, val_df, test_df = self.split_dataset(df)
        
        # 8. Salva datasets
        self.save_dataset(train_df, output_dir, 'train')
        self.save_dataset(val_df, output_dir, 'validation')
        self.save_dataset(test_df, output_dir, 'test')
        
        # 9. Salva estatísticas globais
        self._save_global_stats(output_dir)
        
        logger.info("=" * 60)
        logger.info("✅ Curadoria completa!")
        logger.info("=" * 60)
        self._print_summary()
    
    def _save_global_stats(self, output_dir: Path):
        """Salva estatísticas globais"""
        stats_path = output_dir / 'curate_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': asdict(self.config),
                'stats': self.stats,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Estatísticas globais salvas em {stats_path}")
    
    def _print_summary(self):
        """Imprime resumo final"""
        logger.info("\n📊 RESUMO DA CURADORIA:")
        logger.info(f"  • Total inicial: {self.stats['total_samples']:,}")
        logger.info(f"  • Duplicatas removidas: {self.stats['duplicates_removed']:,}")
        logger.info(f"  • Filtradas por qualidade: {self.stats['quality_filtered']:,}")
        logger.info(f"  • Filtradas por idioma: {self.stats['language_filtered']:,}")
        logger.info(f"  • Filtradas por toxicidade: {self.stats['toxic_filtered']:,}")
        logger.info(f"  • Total final: {self.stats['final_samples']:,}")
        
        total_removed = (self.stats['total_samples'] - self.stats['final_samples'])
        if self.stats['total_samples'] > 0:
            removal_rate = (total_removed / self.stats['total_samples']) * 100
            logger.info(f"  • Taxa de remoção: {removal_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Curadoria de dados para Lloavece Kode',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/raw'),
        help='Diretório de entrada com dados brutos'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Diretório de saída com dados curados'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.8,
        help='Threshold para deduplicação aproximada (0.0-1.0)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Comprimento mínimo de caracteres'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=100000,
        help='Comprimento máximo de caracteres'
    )
    
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=['pt', 'en'],
        help='Idiomas a manter (ex: pt en)'
    )
    
    args = parser.parse_args()
    
    # Cria diretório de logs
    Path('logs').mkdir(exist_ok=True)
    
    # Configuração
    config = CurateConfig(
        similarity_threshold=args.similarity_threshold,
        min_length=args.min_length,
        max_length=args.max_length,
        target_languages=args.languages,
    )
    
    # Executa curadoria
    curator = DataCurator(config)
    curator.curate(args.input_dir, args.output_dir)
    
    logger.info("\n🎉 Curadoria concluída com sucesso!")
    logger.info(f"📁 Dados curados disponíveis em: {args.output_dir}")


if __name__ == '__main__':
    main()
