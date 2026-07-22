import os
import time
import json
import torch
import sentencepiece as spm
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel

# Caminhos
CHECKPOINT_PATH = 'checkpoints/ckpt.pt'
OUT_DIR = 'checkpoints_sft'
DATASET_PATH = 'data/raw/alpaca_ptbr.jsonl'
TOKENIZER_PATH = 'data/tokenized/ia_br_bpe.model'
DEVICE = 'cpu'

# O Fine-Tuning requer uma Taxa de Aprendizado (LR) muito menor para não "destruir" 
# a gramática que o modelo aprendeu no passo anterior.
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
MAX_ITERS = 2000
LOG_INTERVAL = 10
EVAL_INTERVAL = 50

def load_sft_dataset(sp, block_size):
    print("Formatando as conversas para o padrão ChatML...")
    
    user_id = sp.piece_to_id("<|user|>")
    asst_id = sp.piece_to_id("<|assistant|>")
    eos_id = sp.piece_to_id("<|eos|>")
    
    # Vamos empacotar todas as conversas numa única fita longa (packing)
    buffer = []
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompt = data['prompt']
            response = data['response']
            
            # Converte texto para IDs numéricos
            prompt_ids = sp.encode(prompt)
            resp_ids = sp.encode(response)
            
            # A Mágica do Alinhamento: Nós colocamos as tags especiais em volta!
            # <|user|> Qual a capital do Brasil? <|assistant|> Brasília <|eos|>
            conversa = [user_id] + prompt_ids + [asst_id] + resp_ids + [eos_id]
            buffer.extend(conversa)
            
            # Para o seu PC não travar, limitamos a ler apenas 2 Milhões de tokens de conversa
            if len(buffer) > 2_000_000:
                break
                
    return torch.tensor(buffer, dtype=torch.long)

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print("Carregando o Tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

    SFT_CKPT_PATH = os.path.join(OUT_DIR, 'sft_ckpt.pt')
    start_iter = 0
    
    if os.path.exists(SFT_CKPT_PATH):
        print(f"\n[!] Cérebro ASSISTENTE encontrado ({SFT_CKPT_PATH}). Retomando SFT...")
        checkpoint = torch.load(SFT_CKPT_PATH, map_location=DEVICE, weights_only=False)
        config = checkpoint['config']
        model = LloaveceModel(config)
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        start_iter = checkpoint.get('iter_num', 0) + 1
        print(f"[!] SFT retomado a partir do passo {start_iter}.")
        
    else:
        print("Carregando o Cérebro Base da Fase 4...")
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Erro: Cérebro base ({CHECKPOINT_PATH}) não encontrado. Rode o train.py primeiro.")
            return

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        config = checkpoint['config']
        
        model = LloaveceModel(config)
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print("\n[!] Iniciando SFT do zero a partir do modelo base.")

    print("Carregando e processando o Dataset Alpaca...")
    tokens = load_sft_dataset(sp, config.block_size)
    print(f"Dataset de Fine-Tuning carregado com sucesso! ({len(tokens):,} tokens).")

    def get_batch():
        # Sorteia pedaços de conversa aleatórios
        ix = torch.randint(len(tokens) - config.block_size, (BATCH_SIZE,))
        x = torch.stack([tokens[i:i+config.block_size] for i in ix])
        y = torch.stack([tokens[i+1:i+1+config.block_size] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    print("\n=======================================================")
    print("=== INICIANDO O DESPERTAR (SUPERVISED FINE-TUNING) ===")
    print("=======================================================\n")
    
    X, Y = get_batch()
    t0 = time.time()
    
    for iter_num in range(start_iter, MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 and iter_num > 0:
            print(f"--> Salvando o Cérebro do ASSISTENTE em {OUT_DIR}/sft_ckpt.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'iter_num': iter_num,
            }, os.path.join(OUT_DIR, 'sft_ckpt.pt'))

        # Tentar prever a resposta do Assistente
        logits, loss = model(X, Y)
        
        if iter_num % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"Passo SFT {iter_num:4d} | Erro: {loss.item():.4f} | Tempo: {dt*1000:.2f}ms")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        X, Y = get_batch()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTreinamento SFT Interrompido em segurança!")
