# 🇧🇷 Lloavece Kode

**O melhor modelo de código com fluência nativa em português brasileiro**

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Model](https://img.shields.io/badge/model-Qwen3.8--27B-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)

---

## 🎯 Visão

Lloavece Kode é um modelo de IA especializado em **engenharia de software** com **fluência nativa em português brasileiro**, baseado no [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

### Diferenciais

- 🚀 **Top-tier em código**: Competitivo com Claude Sonnet e GPT-4o
- 🇧🇷 **Português fluente**: Explicações e documentação em PT-BR
- 💻 **100% local**: Roda na sua máquina, sem APIs externas
- 🔧 **Engenharia de software**: Focado em problemas reais de desenvolvimento
- 🔄 **Auto-aprimoramento**: Pipeline de self-improvement contínuo

---

## ⚡ Quick Start

**Comece em 5 minutos:**

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download do modelo base (~55GB)
python scripts/download_model.py

# 3. Coleta de dados
python scripts/collect_data.py

# 4. Curadoria e processamento
python scripts/curate_data.py
python scripts/process_data.py --all-splits

# 5. Treinamento
python training/scripts/train_cpt.py --train-data data/training/train.jsonl
```

📖 **Guia completo:** [QUICKSTART.md](QUICKSTART.md)

---

## 🏗️ Arquitetura

### Modelo Base
- **Qwen3.8-27B** (27B parâmetros)
- **Apache 2.0** (uso comercial liberado)
- **Contexto**: 262k tokens nativo, até 1M com YaRN

### Pipeline de Treinamento

```
Qwen3.8-27B (base)
    ↓
Continued Pretraining (CPT) - 2 semanas
    ↓
SFT com LoRA - 2 semanas
    ↓
DPO - 1 semana
    ↓
Self-Improvement Loop (5-6 iterações) - 6 semanas
    ↓
Quantização (GGUF/AWQ) - 1 semana
    ↓
Deploy (VS Code Extension) - 1 semana
```

**Timeline total:** ~3-4 meses

### Metas de Qualidade

| Benchmark | Base | Meta | Status |
|-----------|------|------|--------|
| **HumanEval** | 90% | 95-96% | 🔄 Em progresso |
| **SWE-bench Pro** | 61.7 | 70+ | 🔄 Em progresso |
| **LiveCodeBench** | 55% | 65-70% | 🔄 Em progresso |
| **MBPP** | 85% | 92-93% | 🔄 Em progresso |

---

## 📁 Estrutura do Projeto

```
Lloavece/
├── 📄 README.md              # Este arquivo
├── 📄 QUICKSTART.md          # Guia rápido
├── 📄 ARCHITECTURE.md        # Arquitetura técnica
├── 📄 DATA_PIPELINE.md       # Pipeline de dados
├── 📄 LICENSE                # Apache 2.0
│
├── 📂 scripts/               # Scripts auxiliares
│   ├── setup_environment.py  # Setup do ambiente
│   ├── download_model.py     # Download do modelo base
│   ├── collect_data.py       # Coleta de dados
│   ├── curate_data.py        # Curadoria e limpeza
│   └── process_data.py       # Processamento e tokenização
│
├── 📂 training/              # Treinamento
│   ├── scripts/
│   │   └── train_cpt.py     # Script de CPT
│   └── configs/
│       └── cpt.yaml         # Configuração de CPT
│
├── 📂 data/                  # Datasets
│   ├── raw/                 # Dados brutos
│   ├── processed/           # Dados curados
│   └── training/            # Dados prontos para treino
│
├── 📂 models/                # Modelos treinados
│   ├── base/                # Modelo base (Qwen3.8-27B)
│   ├── cpt/                 # Após CPT
│   ├── sft/                 # Após SFT
│   └── gguf/                # Quantizados
│
├── 📂 extension/             # VS Code Extension (futuro)
├── 📂 inference/             # Código de inferência (futuro)
├── 📂 docs/                  # Documentação
│   └── contributing.md      # Guia de contribuição
└── 📂 logs/                  # Logs de execução
```

---

## 🚀 Funcionalidades (Roadmap)

### v0.1 - Prova de Conceito (Atual)
- [x] Estrutura do projeto
- [x] Scripts de coleta de dados
- [x] Pipeline de curadoria
- [x] Pipeline de processamento
- [x] Script de treinamento CPT
- [ ] Download do modelo base
- [ ] Coleta de dados real
- [ ] Treinamento CPT

### v0.5 - MVP
- [ ] Modelo CPT treinado
- [ ] Avaliação inicial (HumanEval, MBPP)
- [ ] CLI básica para inferência
- [ ] Documentação técnica

### v1.0 - Release
- [ ] SFT com LoRA
- [ ] DPO
- [ ] Self-improvement loop (5-6 iterações)
- [ ] Quantização (GGUF/AWQ)
- [ ] VS Code Extension
- [ ] API local (FastAPI)
- [ ] Documentação completa

### v2.0 - Avançado
- [ ] Sistema RAG para conhecimento atualizado
- [ ] Múltiplos tamanhos (4B, 14B, 32B)
- [ ] Plugin JetBrains
- [ ] Multi-linguagem (PT-BR, EN, ES)
- [ ] Agentic coding (execução de comandos)

---

## 📊 Pipeline de Dados

```
┌─────────────────┐
│   Coleta        │  GitHub, Stack Overflow, Docs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Curadoria     │  Deduplicação, qualidade, idioma
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Processamento  │  Tokenização, packing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Treinamento   │  CPT → SFT → DPO → Self-Improvement
└─────────────────┘
```

📖 **Detalhes:** [DATA_PIPELINE.md](DATA_PIPELINE.md)

---

## 🛠️ Tecnologias

### Core
- **Modelo base**: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Framework**: PyTorch 2.2+, Transformers, PEFT
- **Treinamento**: DeepSpeed, LoRA/QLoRA
- **Quantização**: GGUF (llama.cpp), AWQ

### Dados
- **Coleta**: GitHub API, Stack Overflow API
- **Processamento**: datasketch (MinHash), langdetect
- **Formatos**: JSONL, Parquet

### Infraestrutura
- **GPU**: A100 (treinamento), CPU/GPU consumer (inferência)
- **Cloud**: Colab Pro, Google Drive 5TB
- **Monitoramento**: Weights & Biases

---

## 📦 Instalação

### Requisitos

- Python 3.11+
- CUDA 12.x (para GPU)
- 55GB de espaço livre (modelo base)
- 8GB+ RAM (inferência), 64GB+ RAM (treinamento)

### Setup Completo

```bash
# Clone o repositório
git clone https://github.com/Alekkzsx/Lloavece.git
cd Lloavece

# Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Configure ambiente
python scripts/setup_environment.py

# Baixe o modelo base
python scripts/download_model.py --verify
```

---

## 🎯 Uso

### Inferência (após treinamento)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carrega modelo
model_path = "models/cpt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Gera código
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### CLI (futuro)

```bash
# Gera código
lloavece generate "função para calcular fatorial"

# Explica código
lloavece explain "def quicksort(arr): ..."

# Review de código
lloavece review arquivo.py
```

### VS Code Extension (v1.0)

Instale a extensão e use:
- Autocomplete inteligente
- Explicações em PT-BR
- Code review
- Geração de código

---

## 📈 Performance Esperada

### Inferência (após quantização)

| Configuração | VRAM | Velocidade | Qualidade |
|--------------|------|------------|-----------|
| **GGUF Q4_K_M** | 18GB | 30-50 tok/s | 95% |
| **GGUF Q5_K_M** | 22GB | 25-40 tok/s | 97% |
| **AWQ 4-bit** | 16GB | 35-55 tok/s | 96% |
| **BF16 (sem quant)** | 55GB | 15-25 tok/s | 100% |

### Treinamento (A100 40GB)

| Fase | Tempo | VRAM | Batch Size |
|------|-------|------|------------|
| **CPT** | 1-2 semanas | 35GB | 1 (grad acc 8) |
| **SFT** | 1-2 semanas | 30GB | 2 (grad acc 4) |
| **DPO** | 1 semana | 30GB | 2 (grad acc 4) |

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja [docs/contributing.md](docs/contributing.md) para detalhes.

### Áreas Prioritárias

1. **Coleta de dados**: Mais repositórios brasileiros de qualidade
2. **Curadoria**: Melhorar filtros de qualidade
3. **Avaliação**: Implementar mais benchmarks
4. **Documentação**: Tutoriais e exemplos
5. **VS Code Extension**: Interface e UX

---

## 📚 Documentação

- **[QUICKSTART.md](QUICKSTART.md)** - Guia rápido para começar
- **[DATA_PIPELINE.md](DATA_PIPELINE.md)** - Pipeline completo de dados
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Arquitetura técnica detalhada
- **[docs/contributing.md](docs/contributing.md)** - Como contribuir

---

## 🎓 Aprendizado

### Papers e Referências

- [Qwen3.8 Technical Report](https://qwen.ai/blog?id=qwen3.8)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Self-Improvement in LLMs](https://arxiv.org/abs/2312.01552)

### Tutoriais

- [Fine-tuning LLMs com LoRA](https://huggingface.co/docs/peft)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

## 📄 Licença

Este projeto está licenciado sob a **Apache License 2.0** - veja o arquivo [LICENSE](LICENSE) para detalhes.

O modelo base **Qwen3.8-27B** também é Apache 2.0, permitindo uso comercial.

---

## 🙏 Agradecimentos

- **Qwen Team** - Pelo excelente modelo base Qwen3.8-27B
- **Hugging Face** - Pela infraestrutura e ferramentas
- **Comunidade Open Source** - Pelas contribuições inestimáveis

---

## 📞 Contato

- **GitHub**: [Alekkzsx/Lloavece](https://github.com/Alekkzsx/Lloavece)
- **Issues**: [Reportar bug ou sugerir feature](https://github.com/Alekkzsx/Lloavece/issues)

---

## 🗺️ Roadmap Detalhado

### Q1 2026 (Atual)
- [x] Setup do projeto
- [x] Pipeline de dados
- [ ] CPT completo
- [ ] Avaliação inicial

### Q2 2026
- [ ] SFT + DPO
- [ ] Self-improvement
- [ ] Quantização
- [ ] VS Code Extension v1.0

### Q3 2026
- [ ] RAG system
- [ ] Múltiplos tamanhos
- [ ] API pública
- [ ] Plugin JetBrains

### Q4 2026
- [ ] Agentic coding
- [ ] Multi-linguagem
- [ ] v2.0 release

---

**Feito com ❤️ no Brasil 🇧🇷**

*Lloavece Kode - Programando o futuro, em português*
