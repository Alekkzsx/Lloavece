import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model.config import LloaveceConfig

class RMSNorm(nn.Module):
    """ Normalização ultra rápida usada pelo LLaMA e Gemma. """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm

class SwiGLU(nn.Module):
    """ O cérebro lógico (Feed Forward) mais avançado do mundo open-source hoje. """
    def __init__(self, config: LloaveceConfig):
        super().__init__()
        # A matemática dita que a dimensão oculta é ~ 8/3 * dimensão normal
        hidden_dim = int(8 * config.n_embd / 3)
        # Arredonda para múltiplo de 256 (truque sujo para a GPU fazer as contas mais rápido)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Fórmula: Dropout( W2 * ( SiLU(W1 * x) * W3 * x ) )
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: LloaveceConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Em vez de criar 3 Lineares, criamos 1 gigante (3x) porque multiplicar matrizes de uma vez é mais rápido
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        
        # QKV de uma vez só!
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # O MODO DEUS: Flash Attention
        # O PyTorch 2.0 faz isso em código de máquina (C++/CUDA). É extremamente eficiente!
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=True # A máscara para não ler o futuro fica embutida aqui!
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, config: LloaveceConfig):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rmsnorm_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        # A mágica da conexão residual
        x = x + self.attn(self.rmsnorm_1(x))
        x = x + self.mlp(self.rmsnorm_2(x))
        return x

class LloaveceModel(nn.Module):
    """ A Grande Arquitetura Oficial da Lloavece """
    def __init__(self, config: LloaveceConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding normal. O modelo final de 7B usaria RoPE (Math pesada complexa). 
            # Mantemos simples aqui para garantir que roda bem na sua CPU.
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Truque de Ouro: Weight Tying.
        # A gente aponta a camada final de previsão para ter o mesmo peso matemático
        # que a camada inicial de Embeddings. Economizamos 30% da memória RAM com essa linha!
        self.transformer.wte.weight = self.lm_head.weight
        
        # Inicializa tudo com os devios-padrão corretos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Em modo inferência, não gastamos CPU computando o vocabulário para todos os tokens
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
