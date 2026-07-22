import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparâmetros - Agora um modelo um pouco maior!
batch_size = 32
block_size = 32 # Contexto maior (lê 32 letras de uma vez)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Parâmetros reais de um Transformer
n_embd = 64     # Dimensão do embedding (quão complexo é o vetor de cada letra)
n_head = 4      # Número de cabeças de atenção (Multi-Head)
n_layer = 3     # Número de Blocos Transformer sequenciais
dropout = 0.1   # Para evitar overfitting

# Texto de exemplo brasileiro (expandido um pouco para ter mais palavras)
text = """O Brasil é o maior país da América do Sul.
Temos uma cultura muito diversa, do carnaval ao futebol.
Nossa língua é o português brasileiro, cheio de sotaques e gírias.
Aqui começamos a jornada da nossa IA Brasileira!
O clima no Brasil é predominantemente tropical.
A capital do Brasil é Brasília. São Paulo é a cidade mais populosa.
A floresta amazônica é a maior floresta tropical do mundo.
Nós amamos comer feijoada e pão de queijo.
""" * 30

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


# --- INOVAÇÕES MODERNAS (Conforme seu PLAN.md) ---

class RMSNorm(nn.Module):
    """ RMSNorm: Mais rápido e estável que o antigo LayerNorm """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / norm)

class SwiGLUFFN(nn.Module):
    """ O Feed-Forward moderno (SwiGLU) usado no LLaMA e na nossa IA BR """
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x)) # Função de ativação Swish (SiLU)
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# --- MECANISMOS DE ATENÇÃO ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Múltiplas cabeças de atenção rodando em paralelo """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Cada cabeça aprende focar em coisas diferentes (vogais, verbos, pontuação...)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# --- O BLOCO TRANSFORMER ---

class Block(nn.Module):
    """ Bloco Transformer Completo """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = SwiGLUFFN(n_embd)
        self.rmsnorm1 = RMSNorm(n_embd)
        self.rmsnorm2 = RMSNorm(n_embd)

    def forward(self, x):
        # Arquitetura "Pre-Norm" e Conexões Residuais (o "x +" antes de tudo)
        x = x + self.sa(self.rmsnorm1(x))
        x = x + self.ffwd(self.rmsnorm2(x))
        return x


# --- O MODELO FINAL ---

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Empilhando vários blocos Transformers
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.rmsnorm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        x = tok_emb + pos_emb
        x = self.blocks(x)        # Passa por todas as camadas do Transformer
        x = self.rmsnorm_f(x)     # Normalização final
        logits = self.lm_head(x)  # Prevê as próximas letras
        
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    print(f"Treinando GPT Completo em miniatura no device: {device}")
    print(f"Parâmetros do modelo: {sum(p.numel() for p in TransformerLanguageModel().parameters())}")
    
    model = TransformerLanguageModel()
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- Teste de Geração Final ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=400)[0].tolist()))
