from dataclasses import dataclass
from typing import Optional

@dataclass
class LloaveceConfig:
    """
    Configuração Oficial da Arquitetura Lloavece-Coder.
    Baseada no estado da arte para modelos de código (RoPE + GQA + SwiGLU + RMSNorm).
    """
    vocab_size: int = 32768        # Vocabulário BPE Byte-Fallback otimizado para Código + PT-BR
    block_size: int = 4096         # Janela de bloco padrão para treino denso
    max_context_tokens: int = 1_000_000 # Janela de contexto expandida de 1M tokens (RoPE Dinâmico)
    n_layer: int = 24              # Número de blocos Transformer
    n_head: int = 14               # Cabeças de atenção para Query
    n_kv_head: int = 2             # Cabeças de atenção para Key/Value (Grouped Query Attention - GQA)
    n_embd: int = 896              # Dimensão dos embeddings (d_model)
    intermediate_size: Optional[int] = None # Dimensão interna do SwiGLU (se None, calculado automaticamente)
    rope_theta: float = 10000000.0 # Frequência base do RoPE escalada para contexto de 1M tokens
    norm_eps: float = 1e-6         # Epsilon numérico do RMSNorm
    dropout: float = 0.0           # Dropout (0.0 para pré-treino com dados maciços)
    bias: bool = False             # Sem bias em projeções lineares (estilo LLaMA / Mistral)
    tie_word_embeddings: bool = True # Compartilha pesos de embedding e lm_head

    @classmethod
    def from_name(cls, name: str):
        presets = {
            "test": cls(
                n_layer=2,
                n_head=4,
                n_kv_head=2,
                n_embd=128,
                block_size=256,
                max_context_tokens=1_000_000,
                vocab_size=1024,
            ),
            "micro": cls(
                n_layer=6,
                n_head=8,
                n_kv_head=2,
                n_embd=512,
                block_size=512,
                max_context_tokens=1_000_000,
                vocab_size=1024,
            ),
            "0.5B": cls(
                n_layer=24,
                n_head=14,
                n_kv_head=2,
                n_embd=896,
                block_size=4096,
                max_context_tokens=1_000_000,
                vocab_size=32768,
                rope_theta=10000000.0,
            ),
            "1.5B": cls(
                n_layer=28,
                n_head=16,
                n_kv_head=4,
                n_embd=1536,
                block_size=4096,
                max_context_tokens=1_000_000,
                vocab_size=32768,
                rope_theta=10000000.0,
            ),
            "3B": cls(
                n_layer=32,
                n_head=24,
                n_kv_head=8,
                n_embd=2048,
                block_size=8192,
                max_context_tokens=1_000_000,
                vocab_size=32768,
                rope_theta=10000000.0,
            ),
            "7B": cls(
                n_layer=32,
                n_head=32,
                n_kv_head=8,
                n_embd=4096,
                block_size=8192,
                max_context_tokens=1_000_000,
                vocab_size=32768,
                rope_theta=10000000.0,
            ),
        }

        if name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Preset '{name}' não existe. Escolha entre: {available}")

        return presets[name]
