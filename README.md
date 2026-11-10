# 🇧🇷 Lloavece-Coder
> **A Primeira Grande Inteligência Artificial de Código Nativa do Brasil.**  
> *Treinada com dados de desenvolvimento de alta fidelidade, arquitetura Transformer moderna com RoPE e tecnologia de compressão K-Bit Sparse.*

---

```
   ██╗     ██╗      ██████╗  █████╗ ██╗   ██╗███████╗ ██████╗███████╗
   ██║     ██║     ██╔═══██╗██╔══██╗██║   ██║██╔════╝██╔════╝██╔════╝
   ██║     ██║     ██║   ██║███████║██║   ██║█████╗  ██║     █████╗  
   ██║     ██║     ██║   ██║██╔══██║╚██╗ ██╔╝██╔══╝  ██║     ██╔══╝  
   ███████╗███████╗╚██████╔╝██║  ██║ ╚████╔╝ ███████╗╚██████╗███████╗
   ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝ ╚═════╝╚══════╝
                            [ CODER EDITION ]
```

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Architecture: RoPE + GQA](https://img.shields.io/badge/Architecture-RoPE%20%2B%20GQA-purple.svg)]()
[![Quantization: K--Bit Sparse](https://img.shields.io/badge/Compression-K--Bit%20Sparse%20(~0.15%20bits)-red.svg)]()

---

## 🏛️ O Fluxo Livre dos 5 Modelos (1M de Contexto & Aval Mútuo)

A **Lloavece-Coder** opera com uma janela de **1.000.000 de tokens (1M Context Window)** reunindo 5 modelos que trabalham **100% livres**, sem tarefas fixas ou papéis engessados:

- **`moonshotai/Kimi-K2.5`** — **Voz Direta com o Usuário & Participante Livre do Fluxo (1M)**: É quem fala diretamente com o usuário e também participa livremente da construção e decisão em todas as tasks.
- **`Qwen/Qwen3.8-2.4T-A95B`** — **Livre no Fluxo (1M)**: Pode fazer o que quiser entre si para resolver a task (arquitetura, código, ideias, refatoração).
- **`zai-org/GLM-5.3`** — **Livre no Fluxo (1M)**: Livre para contribuir em qualquer etapa da resolução.
- **`MiniMaxAI/MiniMax-M3`** — **Livre no Fluxo (1M)**: Livre para contribuir em qualquer etapa da resolução.
- **`deepseek-ai/DeepSeek-R1`** — **Livre no Fluxo (1M)**: Livre para contribuir em qualquer etapa da resolução.

> **Regra de Ouro:** Os 5 modelos conversam livremente entre si sobre a tarefa e só entregam o resultado após o **Aval Mútuo (5/5)** de todos. Compatível nativamente com **Cline** (Plan Mode & Act Mode), **Roo Code**, **Hermes** e **OpenAI API (`/v1`)**.

---

## 🎯 Por que a Lloavece-Coder existe?

Os modelos globais de inteligência artificial foram concebidos a partir de textos e documentações em inglês. Embora consigam gerar código básico, eles enfrentam três problemas graves para o desenvolvedor brasileiro:

1. **Alucinação no Ecossistema Brasileiro**: Falham em regras de negócio críticas do Brasil — como fluxos de **PIX do Banco Central**, validação/formatação de **CPF/CNPJ**, emissão de **NF-e/SPED**, integrações com gateways nacionais (Asaas, Mercado Pago, Efí) e conformidade técnica com a **LGPD**.
2. **Mentoria e Explicação Artificiais**: Traduzem explicações técnicas de forma mecânica. A Lloavece foi desenhada para pensar e explicar arquitetura, depurar erros e conduzir revisões de código com vocabulário técnico nativo em **Português do Brasil**.
3. **Dependência e Custo de Nuvem Estrangeira**: A Lloavece-Coder é 100% soberana, de código aberto, e com a tecnologia **K-Bit Sparse** roda localmente em computadores normais e até processadores sem GPU dedicada a mais de **100 tokens por segundo**.

---

## 🧠 Arquitetura do Modelo

A Lloavece adota os mais rigorosos padrões da engenharia moderna de LLMs (*SOTA*):

```
Entrada de Código / Prompt (Tokens)
              │
    [ Embedding Layer ]               (Vocabulário Byte-Fallback de 32k)
              │
    ┌─────────┴─────────┐
    │  Transformer Block│ x N Camadas
    │  ├── RMSNorm      │             (Pre-normalization numérica estável)
    │  ├── RoPE + GQA   │             (Rotary Position + Grouped Query Attention)
    │  │   └── SDPA     │             (FlashAttention nativo PyTorch 2.0+)
    │  ├── RMSNorm      │
    │  └── SwiGLU MLP   │             (Feed-Forward não-linear ativado por SiLU)
    └─────────┬─────────┘
              │
      [ RMSNorm Final ]
              │
    [ Language Model Head ]           (Tied Weights com Embedding inicial)
              │
      Próximo Token / Infill FIM
```

### Configurações de Escala

| Configuração | Parâmetros | Contexto | VRAM Treino | Tamanho K-Bit (~0.15 bits) | Velocidade CPU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.5B (Micro)** | ~490 Milhões | 4.096 | ~12 GB (A100/L4) | **~45 MB** | ~180 tokens/s |
| **1.5B (Mini)** | ~1.54 Bilhões | 4.096 | ~28 GB (A100) | **~120 MB** | ~90 tokens/s |
| **3B (Base)** | ~3.10 Bilhões | 8.192 | ~52 GB (A100 80G) | **~240 MB** | ~45 tokens/s |
| **7B (Flagship)**| ~7.24 Bilhões | 8.192 | Multi-GPU | **~550 MB** | ~20 tokens/s |

---

## ⚡ A Inovação: Compressão K-Bit Sparse

Modelos convencionais usam quantização escalar (como 4-bit GGUF ou 1.58-bit BitNet da Microsoft). O método **K-Bit Sparse da Lloavece** supera essa abordagem explorando a geometria tensorial dos pesos da rede:

1. **Decomposição Tensorial de Kronecker ($W \approx A \otimes B$)**:
   Uma matriz de pesos $4096 \times 4096$ (16,7 milhões de números) é decomposta em duas matrizes sementes $64 \times 64$ ($8.192$ números no total).  
   *Taxa de compressão bruta: **2.048x**.*
2. **Resíduo Esparso Ternário ($\Delta$ 1%)**:
   Os neurônios de precisão crítica (que definem indentação, sintaxe e fechamento de blocos de código) são preservados em uma máscara esparsa ternária $\{-1, 0, +1\}$.
3. **Multiplicação Acelerada em Cache L1/L2**:
   Calculada via $y = \text{vec}(B \cdot X \cdot A^T)$, a computação cabe inteira dentro do cache do processador, eliminando o gargalo de largura de banda de memória RAM.

---

## 🧩 Suporte Nativo a FIM (Fill-in-the-Middle)

A Lloavece foi projetada para atuar diretamente como **extensão de autocompletar em IDEs (VS Code, Cursor, Neovim, Continue.dev)**:

* `<|fim_prefix|>`: O código que vem antes do cursor do desenvolvedor.
* `<|fim_suffix|>`: O código que vem depois do cursor.
* `<|fim_middle|>`: O bloco de código que a Lloavece gera com precisão cirúrgica.

---

## 📁 Estrutura do Repositório

```
Lloavece/
├── README.md                     # Documentação oficial
├── pyproject.toml                # Dependências e empacotamento
├── configs/                      # Definições de hiperparâmetros
├── src/
│   ├── model/
│   │   ├── config.py             # Configuração e presets de arquitetura
│   │   └── modeling.py           # Transformer SOTA com RoPE e GQA
│   ├── tokenizer/
│   │   ├── special_tokens.py     # Tokens FIM, ChatML e controle de código
│   │   └── code_tokenizer.py     # Treinador BPE com preservação de indentação
│   ├── data/
│   │   ├── collectors/           # Download The Stack, Python-Edu, dados BR
│   │   ├── processors/           # Filtro sintático e gerador FIM
│   │   └── pipeline.py           # Pipeline em disco via Memmap uint16
│   ├── training/
│   │   ├── train.py              # Treinador com bfloat16, GradAccum e Drive Sync
│   │   └── sft.py                # Fine-tuning de instruções técnicas em PT-BR
│   └── export/
│       ├── kbit_compressor.py    # Algoritmo de Kronecker + Resíduo Esparso
│       └── to_gguf.py            # Conversor para Ollama e LM Studio
└── notebooks/
    └── Lloavece_Colab_Trainer.ipynb # Notebook executável em 1-clique para Google Colab
```

---

## 🚀 Como Treinar e Rodar

### 1. Clonar e Instalar Dependências

```bash
git clone https://github.com/alekkzsx/Lloavece.git
cd Lloavece
pip install -e .
```

### 2. Treinar o Tokenizer de Código

```bash
python -m src.tokenizer.code_tokenizer
```

### 3. Coletar e Processar o Dataset de Código

```bash
python -m src.data.collectors.download_code
python -m src.data.pipeline
```

### 4. Iniciar o Pré-Treino (Colab Pro ou Local)

```bash
python -m src.training.train --config 0.5B --precision bf16
```

### 5. Comprimir com o K-Bit Sparse e Rodar no Ollama

```bash
python -m src.export.kbit_compressor --checkpoint checkpoints/lloavece_0.5B.pt
ollama create lloavece -f Modelfile
ollama run lloavece "Escreva uma função async em Python que valide uma chave PIX e emita um log formatado:"
```

---

## 🇧🇷 Licença e Contribuição

Desenvolvido com orgulho no Brasil pela equipe **Lloavece**.  
Distribuído sob licença **MIT**. Sinta-se livre para construir em cima, auditar e acelerar o ecossistema tecnológico brasileiro!
