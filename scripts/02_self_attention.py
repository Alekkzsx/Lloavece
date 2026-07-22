import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparâmetros básicos
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # Dimensão dos embeddings vetoriais

# Texto de exemplo brasileiro
text = """O Brasil é o maior país da América do Sul.
Temos uma cultura muito diversa, do carnaval ao futebol.
Nossa língua é o português brasileiro, cheio de sotaques e gírias.
Aqui começamos a jornada da nossa IA Brasileira!""" * 50

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
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


# --- O CORAÇÃO DO TRANSFORMER ---
class Head(nn.Module):
    """ Uma única cabeça de Self-Attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 'tril' é a máscara triangular para evitar que tokens "vejam o futuro"
        # Em um decoder-only, a letra atual não pode olhar para as próximas letras
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Computa as pontuações de atenção ("afinidade" matemática entre os tokens)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        
        # Aplica a máscara para que o token 'i' só possa olhar para os tokens <= 'i'
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Agrega os Valores (Values) de acordo com a atenção gerada
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Identidade: cada token vira um vetor de tamanho n_embd (32 posições)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 2. Posição: onde o token está? Para o Bigrama isso não importava,
        # mas a Atenção precisa saber quem vem antes e quem vem depois.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 3. A Cabeça de Self-Attention em ação
        self.sa_head = Head(n_embd)
        
        # 4. Projeção final para prever o vocabulário novamente
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        
        # O modelo junta a identidade da letra com sua posição (Contexto Espacial)
        x = tok_emb + pos_emb
        
        # MÁGICA: Os tokens se comunicam através da Self-Attention
        x = self.sa_head(x) 
        
        # Adivinha a próxima letra baseado na conversa entre os tokens
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Recorta o contexto para caber no limite da janela de atenção (block_size)
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    print(f"Treinando Mini-Transformer (Self-Attention) no device: {device}")
    
    model = AttentionLanguageModel()
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- Teste de Geração ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
