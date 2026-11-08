import os
import json
from typing import Iterator
import sentencepiece as spm
from src.tokenizer.special_tokens import ALL_SPECIAL_TOKENS

VOCAB_SIZE = 32768
OUTPUT_DIR = "data/tokenized"
MODEL_PREFIX = os.path.join(OUTPUT_DIR, "lloavece_code_bpe")

def default_text_iterator(data_path: str, max_samples: int = 500000) -> Iterator[str]:
    """Lê arquivos de código e texto linha por linha para alimentar o SentencePiece."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo de treino do tokenizer não encontrado: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                doc = json.loads(line)
                text = doc.get("content") or doc.get("text") or doc.get("code") or ""
                if text.strip():
                    yield text
            except Exception:
                continue

def train_code_tokenizer(
    data_path: str,
    vocab_size: int = VOCAB_SIZE,
    max_samples: int = 500000
):
    """
    Treina o Tokenizer BPE oficial da Lloavece-Coder.
    Configurado com byte-fallback, split de dígitos e preservação estrita de espaços.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_defined_symbols = ",".join(ALL_SPECIAL_TOKENS)

    print(f"\n=======================================================")
    print(f"=== TREINANDO TOKENIZER LLOAVECE-CODER (VOCAB: {vocab_size}) ===")
    print(f"=======================================================\n")
    print(f"Fonte de dados: {data_path}")
    print(f"Símbolos especiais: {len(ALL_SPECIAL_TOKENS)} tokens configurados")

    spm.SentencePieceTrainer.train(
        sentence_iterator=default_text_iterator(data_path, max_samples=max_samples),
        model_prefix=MODEL_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,           # 100% de cobertura graças ao byte_fallback
        byte_fallback=True,               # Nunca gera <unk>, converte bytes desconhecidos em hex
        split_digits=True,                # Fundamental para código e matemática: 1234 -> 1, 2, 3, 4
        allow_whitespace_only_pieces=True,# Preserva tabs e blocos de espaços (indentação Python)
        user_defined_symbols=user_defined_symbols,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
        normalization_rule_name="identity" # Não normaliza ou apaga caracteres especiais de código
    )

    print(f"\n[OK] Tokenizer treinado com sucesso!")
    print(f"Salvo em: {MODEL_PREFIX}.model e {MODEL_PREFIX}.vocab")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Treinador do Tokenizer Lloavece-Coder")
    parser.add_argument("--data", type=str, default="data/raw/code_corpus.jsonl", help="Caminho do arquivo JSONL de código")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE, help="Tamanho do vocabulário")
    parser.add_argument("--samples", type=int, default=500000, help="Quantidade máxima de amostras")
    args = parser.parse_args()

    train_code_tokenizer(args.data, vocab_size=args.vocab_size, max_samples=args.samples)
