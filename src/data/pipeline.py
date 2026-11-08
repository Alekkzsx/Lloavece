import os
import json
import argparse
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
from src.data.processors.fim import apply_fim

TOKENIZER_MODEL = "data/tokenized/lloavece_code_bpe.model"
RAW_DATA = "data/raw/code_corpus.jsonl"
PROCESSED_DIR = "data/processed"
TRAIN_BIN = os.path.join(PROCESSED_DIR, "train.bin")
VAL_BIN = os.path.join(PROCESSED_DIR, "val.bin")

def build_code_binary_dataset(
    raw_data: str = RAW_DATA,
    tokenizer_path: str = TOKENIZER_MODEL,
    val_ratio: float = 0.05,
    fim_rate: float = 0.5
):
    if not os.path.exists(raw_data):
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_data}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer não encontrado: {tokenizer_path}. Execute src/tokenizer/code_tokenizer.py primeiro.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("=== PIPELINE BINÁRIO DE CÓDIGO (MEMMAP + FIM) ===")
    print("=" * 60)
    print(f"Lendo dados de: {raw_data}")
    print(f"Tokenizer: {tokenizer_path}")

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    eos_id = sp.piece_to_id("<|eos|>")
    fim_prefix_id = sp.piece_to_id("<|fim_prefix|>")
    fim_middle_id = sp.piece_to_id("<|fim_middle|>")
    fim_suffix_id = sp.piece_to_id("<|fim_suffix|>")

    # Contagem de linhas
    with open(raw_data, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    val_threshold = int(total_lines * (1.0 - val_ratio))

    # Limpa arquivos antigos
    for f in [TRAIN_BIN, VAL_BIN]:
        if os.path.exists(f):
            os.remove(f)

    f_train = open(TRAIN_BIN, "wb")
    f_val = open(VAL_BIN, "wb")

    train_tokens = 0
    val_tokens = 0

    with open(raw_data, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), total=total_lines, desc="Processando Código"):
            try:
                doc = json.loads(line)
                code = doc.get("content") or doc.get("text") or doc.get("code") or ""
                if not code.strip():
                    continue

                token_ids = sp.encode(code)
                is_val = idx >= val_threshold

                if not is_val and fim_rate > 0.0:
                    # Aplica FIM para autocompletar em IDEs
                    token_ids = apply_fim(
                        token_ids,
                        fim_prefix_id=fim_prefix_id,
                        fim_middle_id=fim_middle_id,
                        fim_suffix_id=fim_suffix_id,
                        fim_rate=fim_rate
                    )

                token_ids.append(eos_id)
                arr = np.array(token_ids, dtype=np.uint16)

                if is_val:
                    f_val.write(arr.tobytes())
                    val_tokens += len(token_ids)
                else:
                    f_train.write(arr.tobytes())
                    train_tokens += len(token_ids)

            except Exception:
                continue

    f_train.close()
    f_val.close()

    print(f"\n[OK] Pipeline concluído com sucesso!")
    print(f"Treino     : {train_tokens:,} tokens salvos em {TRAIN_BIN} ({os.path.getsize(TRAIN_BIN)/1e6:.2f} MB)")
    print(f"Validação  : {val_tokens:,} tokens salvos em {VAL_BIN} ({os.path.getsize(VAL_BIN)/1e6:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=RAW_DATA)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_MODEL)
    parser.add_argument("--fim_rate", type=float, default=0.5)
    args = parser.parse_args()

    build_code_binary_dataset(raw_data=args.data, tokenizer_path=args.tokenizer, fim_rate=args.fim_rate)
