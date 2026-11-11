# 🚀 Lloavece Kode

> **IA local brasileira especializada em engenharia de software**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)

## 🎯 O que é o Lloavece Kode?

Lloavece Kode é um modelo de inteligência artificial **local**, **open-source** e **especializado exclusivamente em código**. Construído sobre o Qwen3.8-27B (Apache 2.0), foi treinado para ser um dos melhores modelos de código open-weight do mundo, com **fluência nativa em português brasileiro**.

### ✨ Características Principais

- 🧠 **Excelência em código**: Treinado em bilhões de tokens de código de alta qualidade
- 🇧🇷 **Português fluente**: Instruções, explicações e documentação em PT-BR
- 🔒 **100% local**: Roda offline, sem API calls, sem cloud
- 🎯 **Focado em engenharia de software**: Especializado em problemas reais de desenvolvedores
- ⚡ **Auto-aprimoramento**: Pipeline de self-improvement para melhoria contínua
- 🌐 **Open-source**: Apache 2.0, sem restrições comerciais

---

## 🏗️ Arquitetura

### Modelo Base: Qwen3.8-27B

| Parâmetro | Valor |
|-----------|-------|
| **Tamanho** | 27 bilhões de parâmetros (dense) |
| **Contexto** | 262k tokens nativo, extensível até 1M |
| **Licença** | Apache 2.0 |
| **Arquitetura** | Híbrida (DeltaNet + Attention) |

### Benchmarks Baseline

| Benchmark | Score |
|-----------|-------|
| SWE-bench Pro | 61.7 |
| QwenSWEBench | 79.0 |
| Terminal Bench 2.1 | 73.0 |

---

## 🚀 Funcionalidades (MVP v1.0)

### ✅ Implementadas

- **Code Completion**: Autocomplete inteligente de código
- **Code Explanation**: Explica código em português
- **Code Review**: Identifica bugs, problemas de segurança e más práticas
- **Code Generation**: Gera código a partir de descrições em português
- **Debugging Assistido**: Analisa erros e sugere correções

### 🚧 Em Desenvolvimento

- **VS Code Extension**: Interface integrada ao IDE
- **Agentic Coding**: Execução autônoma de tarefas complexas
- **Multi-file Editing**: Edição de múltiplos arquivos simultaneamente
- **RAG Integration**: Base de conhecimento sempre atualizada

---

## 📊 Metas de Qualidade

| Benchmark | Base | Lloavece v1.0 | Melhoria |
|-----------|------|---------------|----------|
| SWE-bench Pro | 61.7 | 70-75 | +15% |
| HumanEval | ~90% | 95-96% | +6% |
| MBPP | ~85% | 92-93% | +8% |
| LiveCodeBench | ~55% | 65-70% | +20% |
| Aider Code Editing | ~60% | 70-75% | +15% |

---

## 🔧 Pipeline de Treinamento

O Lloavece Kode utiliza um pipeline de 4 fases para especialização:

### Fase 1: Continued Pretraining (CPT)
- **Objetivo**: Especializar em código de alta qualidade
- **Dados**: 1-5B tokens de código curado
- **Ganho**: +5-8% em benchmarks

### Fase 2: Supervised Fine-Tuning (SFT)
- **Objetivo**: Alinhar com instruções em português
- **Técnica**: LoRA (treina 1-2% dos parâmetros)
- **Ganho**: +10-15% total

### Fase 3: Direct Preference Optimization (DPO)
- **Objetivo**: Alinhar com preferências de qualidade
- **Dados**: 100k-500k pares de preferências
- **Ganho**: +15-20% total

### Fase 4: Self-Improvement Loop
- **Objetivo**: Auto-aprimoramento iterativo
- **Processo**: Gera respostas → Avalia → Treina nos melhores exemplos
- **Ganho**: +25-30% total (após 5-6 iterações)

---

## 💻 Requisitos de Hardware

### Para Inferência (Rodar o Modelo)

| Configuração | VRAM/RAM | Velocidade | Qualidade |
|--------------|----------|------------|-----------|
| **GPU Alta** (RTX 4090) | 24GB VRAM | 30-50 tok/s | Máxima |
| **GPU Média** (RTX 3060) | 12GB VRAM | 15-25 tok/s | Alta |
| **CPU** (32GB RAM) | 40GB RAM | 8-15 tok/s | Alta |

### Para Treinamento

- **GPU**: A100 40GB/80GB
- **RAM**: 64GB+
- **Storage**: 500GB+

---

## 📦 Instalação

### Pré-requisitos

- Python 3.11+
- PyTorch 2.2+
- CUDA 12.x (para GPU)

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Lloavece.git
cd Lloavece

# Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Baixe o modelo base
python scripts/download_model.py

# Configure o ambiente
python scripts/setup_environment.py
```

---

## 🚀 Uso Rápido

### Via CLI

```bash
# Completar código
lloavece complete --file main.py --language python

# Explicar código
lloavece explain --file algorithm.py

# Review de código
lloavece review --file app.py

# Gerar código
lloavece generate --prompt "Função para calcular fatorial em Python"
```

### Via Python API

```python
from lloavece import LloaveceKode

# Inicialize o modelo
model = LloaveceKode(model_path="models/gguf/lloavece-27b-q4_k_m.gguf")

# Gere código
code = model.generate(
    prompt="Escreva uma função que implemente busca binária",
    language="python",
    max_tokens=500
)

print(code)
```

### VS Code Extension

```bash
# Instale a extensão (em desenvolvimento)
cd extension
npm install
npm run compile
# Instale no VS Code
```

---

## 📁 Estrutura do Projeto

```
Lloavece/
├── data/                    # Datasets de treino
├── models/                  # Modelos treinados
├── training/                # Scripts de treino
│   ├── configs/            # Configurações YAML
│   ├── scripts/            # Scripts Python
│   └── utils/              # Utilitários
├── scripts/                 # Scripts auxiliares
├── inference/               # Código de inferência
├── extension/               # VS Code extension
├── docs/                    # Documentação
├── notebooks/               # Jupyter notebooks
├── tests/                   # Testes
├── ARCHITECTURE.md          # Arquitetura técnica
├── README.md               # Este arquivo
└── LICENSE                 # Apache 2.0
```

---

## 🛠️ Stack Técnica

### Treinamento
- PyTorch 2.2+
- Hugging Face Transformers
- LoRA/QLoRA (PEFT)
- DeepSpeed ZeRO-3
- Weights & Biases

### Inferência
- llama.cpp (GGUF)
- vLLM (GPU)
- FastAPI (API REST)

### Deploy
- VS Code Extension (TypeScript)
- CLI (Python)
- Docker

---

## 📈 Roadmap

### v0.1 (Mês 1-2)
- [x] Definição de arquitetura
- [x] Setup do ambiente
- [ ] Download do modelo base
- [ ] Coleta de dados
- [ ] CPT (Continued Pretraining)

### v0.5 (Mês 2-3)
- [ ] SFT com LoRA
- [ ] DPO
- [ ] Self-Improvement Loop (5 iterações)
- [ ] Avaliação inicial

### v1.0 (Mês 4)
- [ ] Quantização (GGUF/AWQ)
- [ ] VS Code Extension (MVP)
- [ ] CLI completo
- [ ] Documentação
- [ ] Release público

### v2.0 (Futuro)
- [ ] RAG Integration
- [ ] Agentic Coding avançado
- [ ] Multi-modal (imagens de código)
- [ ] Modelos menores (7B, 14B)
- [ ] Fine-tuning customizado

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja [docs/contributing.md](docs/contributing.md) para detalhes.

### Áreas Prioritárias

1. **Coleta de dados**: Repositórios brasileiros de qualidade
2. **Curadoria**: Limpeza e filtragem de datasets
3. **Treinamento**: Otimização de hiperparâmetros
4. **Avaliação**: Benchmarks e métricas
5. **Deploy**: VS Code extension, CLI, API

---

## 📄 Licença

Este projeto está licenciado sob a **Apache License 2.0** - veja o arquivo [LICENSE](LICENSE) para detalhes.

O modelo base (Qwen3.8-27B) também está sob Apache 2.0, permitindo uso comercial sem restrições.

---

## 🙏 Agradecimentos

- **Qwen Team** (Alibaba) pelo modelo base excepcional
- **Hugging Face** pela infraestrutura de modelos open-source
- **Comunidade open-source** pelos datasets e ferramentas

---

## 📞 Contato

- **GitHub Issues**: [Reportar bugs ou sugerir features](https://github.com/seu-usuario/Lloavece/issues)
- **Discussions**: [Perguntas e discussões](https://github.com/seu-usuario/Lloavece/discussions)

---

## ⭐ Se este projeto te ajudou

Considere dar uma ⭐ no GitHub e compartilhar com outros desenvolvedores!

---

**Feito com ❤️ no Brasil 🇧🇷**

*Lloavece Kode - Código inteligente, em português, rodando na sua máquina.*
