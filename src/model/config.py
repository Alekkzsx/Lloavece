from dataclasses import dataclass

@dataclass
class LloaveceConfig:
    """ 
    O "DNA" do nosso modelo. 
    Aqui controlamos o tamanho do cérebro da IA alterando essas variáveis.
    """
    block_size: int = 1024       # Quantos tokens a IA consegue lembrar no contexto
    vocab_size: int = 16384      # Exatamente o tamanho do nosso dicionário BPE otimizado!
    n_layer: int = 12            # Quantas camadas de blocos Transformer empilhados
    n_head: int = 12             # Quantas "cabeças" de atenção paralelas
    n_embd: int = 768            # O tamanho da dimensão vetorial (a complexidade de cada token)
    dropout: float = 0.0         # Usamos > 0.0 para evitar que o modelo decore os dados
    bias: bool = False           # Falso (Moderno - LLaMA): Sem viés nas redes neurais. Deixa a rede mais magra e rápida.

    @classmethod
    def from_name(cls, name: str):
        """ Fábrica de configurações """
        if name == "micro":
            # Versão V2 Otimizada (~26 Milhões de parâmetros focados no Transformer).
            # Roda macio em CPU. D_model 512, 6 camadas.
            return cls(n_layer=6, n_head=8, n_embd=512, block_size=512, dropout=0.1)
            
        elif name == "mini":
            # Equivalente ao GPT-2 (~100 Milhões). Requer um PC razoável.
            return cls(n_layer=12, n_head=12, n_embd=768, block_size=512)
            
        elif name == "7B":
            # O MONSTRO OFICIAL DO PLANO. Requer centenas de GPUs de Nuvem para treinar.
            return cls(n_layer=32, n_head=32, n_embd=4096, block_size=4096)
            
        raise ValueError(f"Configuração {name} não reconhecida.")
