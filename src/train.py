import os
import math
import time
import torch
import numpy as np
from torch.nn import functional as F

# Importando a Arquitetura que acabamos de criar
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel

# -----------------------------------------------------------------------------
# HIPERPARÂMETROS DO TREINO (Painel de Controle)
# -----------------------------------------------------------------------------
out_dir = 'checkpoints'
eval_interval = 50     # A cada 50 passos, o modelo pausa para medir o quão inteligente está
eval_iters = 10        # Usa 10 amostras de validação para medir o erro
log_interval = 10      # Imprime a velocidade no terminal a cada 10 passos
always_save_checkpoint = True 

# Tamanhos: 'micro', 'mini', ou '7B'
config_name = "micro"

batch_size = 4         # Lote de frases. Se seu PC travar/reiniciar, diminua isso para 2 ou 1.
learning_rate = 6e-4
max_iters = 10000      # Loop máximo para deixar rodando a madrugada toda.

# O código agora é inteligente: Se achar uma GPU (Colab), ele usa. Se não, usa a CPU do seu PC.
device = 'cuda' if torch.cuda.is_available() else 'cpu'        
# -----------------------------------------------------------------------------

config = LloaveceConfig.from_name(config_name)
block_size = config.block_size
data_dir = 'data/processed'

def get_batch(split):
    # Lendo o "Combustível Binário" (Memória Mapeada - Velocidade da luz, 0 RAM gasta)
    filename = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Sorteia posições aleatórias no arquivo de 1.6GB
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extrai o contexto e a resposta (futuro)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """ Função que roda sem treinar (sem backpropagation) só para calcular o nível de erro atual. """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"\n=============================================")
    print(f"=== INICIALIZANDO O CÉREBRO LLOAVECE ({config_name.upper()}) ===")
    print(f"=============================================\n")
    
    model = LloaveceModel(config)
    model.to(device)
    
    # Mostrar quantos milhões de neurônios ele tem
    params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros : {params/1e6:.2f} Milhões")
    print(f"Treinando no hardware: {device.upper()}")
    print(f"Tamanho do Batch    : {batch_size}")
    print(f"Tamanho do Contexto : {block_size} tokens")

    # O Otimizador (AdamW) é o professor que corrige os erros (pesos) do modelo
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)
    
    # === SISTEMA DE RETOMADA (RESUME) ===
    start_iter = 0
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        print(f"\n[!] Cérebro existente encontrado ({ckpt_path}). Retomando memórias...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter_num'] + 1
        print(f"[!] Treinamento retomado a partir do passo {start_iter}.")
    else:
        print("\n[!] Nenhum cérebro salvo encontrado. Iniciando do absoluto zero.")
    
    print("\nIniciando ingestão da memória mapeada e loop de otimização...")
    X, Y = get_batch('train')
    
    t0 = time.time()
    for iter_num in range(start_iter, max_iters):

        # A cada X passos, paramos para avaliar e salvar um backup no HD
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model)
            print(f"\n[Passo {iter_num}] Erro (Validação): {losses['val']:.4f} | Erro (Treino): {losses['train']:.4f}")
            
            if always_save_checkpoint:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'config': config,
                    'val_loss': losses['val']
                }
                print(f"--> Salvando 'foto' do cérebro em {out_dir}/ckpt.pt (Checkpoint)")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # Log de tempo real (pra você saber se seu PC tá sofrendo muito)
        if iter_num % log_interval == 0 and iter_num > 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"passo {iter_num} | erro (loss) {loss.item():.4f} | tempo: {dt*1000:.2f}ms por rodada")

        # 1. Tentar prever o futuro
        logits, loss = model(X, Y)
            
        # 2. Backpropagation: Calcular a diferença matemática do que ele errou
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 3. O professor atualiza a mente da rede neural
        optimizer.step()
        
        # 4. Puxa as próximas frases da Wikipedia (via Memmap instantâneo)
        X, Y = get_batch('train')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido pelo usuário! Não se preocupe, o último backup está salvo na pasta checkpoints/.")
