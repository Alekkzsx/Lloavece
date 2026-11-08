"""
Definição de Tokens Especiais Oficiais da Lloavece-Coder.
Inclui suporte a FIM (Fill-in-the-Middle), ChatML e controle de contexto de repositórios.
"""

# Tokens de Diálogo e Chat (Padrão ChatML)
CHAT_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|sys|>",
    "<|user|>",
    "<|assistant|>",
]

# Tokens FIM (Fill-in-the-Middle) — Essenciais para autocompletar código em IDEs
FIM_TOKENS = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

# Tokens de Estrutura de Código e Repositórios
CODE_STRUCTURE_TOKENS = [
    "<|repo_name|>",
    "<|file_sep|>",
    "<|code_block|>",
    "<|indent_2|>",
    "<|indent_4|>",
    "<|tab|>",
]

# Tokens de Extensibilidade Futura
EXTRA_SPECIAL_TOKENS = [f"<|ext_{i}|>" for i in range(1, 9)]

# Lista unificada de todos os símbolos especiais
ALL_SPECIAL_TOKENS = (
    CHAT_TOKENS +
    FIM_TOKENS +
    CODE_STRUCTURE_TOKENS +
    EXTRA_SPECIAL_TOKENS
)
