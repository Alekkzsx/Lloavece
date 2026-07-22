import os
import json
import sentencepiece as spm
from tqdm import tqdm

# O numpy é crucial aqui, ele lida com a matriz binária
try:
    import numpy as np
except ImportError:
    print("ERRO: O pacote 'numpy' é necessário. Rode no terminal: pip install numpy")
    exit(1)

RAW_DATA = "data/raw/wikipedia/ptwiki_articles.jsonl"
TOKENIZER_MODEL = "data/tokenized/ia_br_bpe.model"
PROCESSED_DIR = "data/processed"

# O arquivo oficial que a IA vai ler no treinamento
TRAIN_BIN = os.path.join(PROCESSED_DIR, "train.bin")
VAL_BIN = os.path.join(PROCESSED_DIR, "val.bin")

def build_binary_dataset():
    if not os.path.exists(RAW_DATA):
        raise FileNotFoundError(f"Arquivo não encontrado: {RAW_DATA}")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    print(f"Carregando o nosso novo Tokenizer BPE...")
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    
    # A IA precisa saber onde um artigo acaba e o outro começa.
    eos_id = sp.piece_to_id("<|eos|>")

    print(f"Iniciando conversão de texto para Binário Puro (Memmap uint16)...")
    
    total_articles = 1232533 # O número exato que coletamos
    val_articles = 10000     # Vamos separar os últimos 10 mil artigos para Validação
    
    train_tokens_count = 0
    val_tokens_count = 0
    
    buffer = []
    buffer_size = 2_000_000 # Guarda 2 milhões de tokens na RAM antes de salvar no SSD
    
    # Deleta os arquivos antigos se existirem
    for f in [TRAIN_BIN, VAL_BIN]:
        if os.path.exists(f): os.remove(f)
        
    # Vamos criar os arquivos binários abertos no modo "append binary"
    f_train = open(TRAIN_BIN, 'wb')
    f_val = open(VAL_BIN, 'wb')
    
    with open(RAW_DATA, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), total=total_articles, desc="Comprimindo Wikipedia"):
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue
                
            # O momento mágico: Texto -> Lista de Números
            ids = sp.encode(text)
            ids.append(eos_id) 
            
            buffer.extend(ids)
            
            # Se for artigo de treino vs validação
            is_val = (total_articles - i) <= val_articles
            
            if is_val:
                val_tokens_count += len(ids)
            else:
                train_tokens_count += len(ids)
            
            # Quando o buffer encher, joga tudo no SSD como binário maciço
            if len(buffer) >= buffer_size:
                # uint16 (Unsigned Integer de 16 bits). Vai de 0 a 65535.
                # Como nosso vocabulário tem exatos 65536 tokens, cabe PERFEITAMENTE.
                # Cada token vai ocupar APENAS 2 bytes (extrema economia de memória)
                arr = np.array(buffer, dtype=np.uint16)
                
                if is_val:
                    f_val.write(arr.tobytes())
                else:
                    f_train.write(arr.tobytes())
                buffer = []
                
        # Grava os últimos resíduos do buffer
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f_train.write(arr.tobytes())

    f_train.close()
    f_val.close()

    print(f"\n✅ Pipeline Concluído! O combustível está pronto.")
    print(f"Total de Tokens (Treino): {train_tokens_count:,}")
    print(f"Total de Tokens (Validação): {val_tokens_count:,}")
    
    train_mb = os.path.getsize(TRAIN_BIN) / (1024 * 1024)
    print(f"\nTamanho no SSD (train.bin): {train_mb:.2f} MB")
    print("\nO seu modelo agora pode ler milhões de tokens em fração de segundos usando memória mapeada!")

if __name__ == "__main__":
    build_binary_dataset()
