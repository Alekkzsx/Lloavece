# 🇧🇷 Lloavece Kode

> **A primeira IA local brasileira especializada em código.**  
> *100% offline, fluente em Português do Brasil, otimizada para desenvolvedores.*

---

```
   ██╗     ██╗      ██████╗  █████╗ ██╗   ██╗███████╗ ██████╗███████╗
   ██║     ██║     ██╔═══██╗██╔══██╗██║   ██║██╔════╝██╔════╝██╔════╝
   ██║     ██║     ██║   ██║███████║██║   ██║█████╗  ██║     █████╗  
   ██║     ██║     ██║   ██║██╔══██║╚██╗ ██╔╝██╔══╝  ██║     ██╔══╝  
   ███████╗███████╗╚██████╔╝██║  ██║ ╚████╔╝ ███████╗╚██████╗███████╗
   ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝ ╚═════╝╚══════╝
                            [ KODE EDITION ]
```

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Format: GGUF](https://img.shields.io/badge/Format-GGUF-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Status: Em Desenvolvimento](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)]()

---

## 🎯 O que é a Lloavece Kode?

**Lloavece Kode** é uma família de modelos de linguagem especializados **exclusivamente em código**, otimizados para desenvolvedores brasileiros.

### Princípios Fundamentais

1. **🔒 100% Local**: Roda offline, sem API calls, sem cloud, sem telemetria
2. **💻 Especializado em Código**: Não é assistente geral, só sabe programar (e programa muito bem)
3. **🇧🇷 Brasileiro Nativo**: Fluente em PT-BR, entende APIs brasileiras (PIX, NFe, SPED, LGPD)
4. **⚡ Acessível**: Roda em hardware comum (laptops, desktops, servidores)
5. **🆓 Open Source**: Apache 2.0, sem restrições comerciais

---

## 🏗️ Família de Modelos

| Modelo | Parâmetros | Tamanho (Q4_K_M) | RAM Mínima | Hardware Alvo |
|--------|------------|------------------|------------|---------------|
| **lloavece-kode-32b** | 32B | ~19 GB | 24GB+ | GPU (RTX 3090/4090, A100) |
| **lloavece-kode-14b** | 14B | ~8.7 GB | 12GB | Desktop (CPU/GPU) |
| **lloavece-kode-4b** | 4B | ~2.5 GB | 4-8GB | Laptop (CPU) ← **MVP** |

### Por que 3 Tamanhos?

- **32B**: Máxima qualidade, para quem tem hardware robusto
- **14B**: Equilíbrio perfeito entre qualidade e velocidade
- **4B**: **MVP principal**, roda em laptops brasileiros típicos (8GB RAM)

---

## 🚀 Funcionalidades (MVP v1.0)

### ✅ Code Completion
Autocompleta código enquanto você digita, integrado ao VS Code.

### ✅ Code Explanation
Explica código em **Português do Brasil** fluente, com vocabulário técnico natural.

### ✅ Code Review
Analisa código e aponta bugs, problemas de segurança, más práticas e sugestões de melhoria.

### ✅ Code Generation
Gera código a partir de descrições em PT-BR:
```
"Crie uma função Python que valide um CPF"
"Implemente um endpoint FastAPI que integre com PIX"
"Escreva um algoritmo de ordenação em TypeScript"
```

### ✅ Debugging Assistido
Analisa erros e sugere correções:
```
"Por que este código dá TypeError?"
"Corrija este bug: ..."
"Este código não funciona, por quê?"
```

---

## 🇧🇷 Diferencial Brasileiro

### O que torna a Lloavece "brasileira"?

1. **Português do Brasil fluente**
   - Comentários, explicações e documentação em PT-BR natural
   - Sem traduções mecânicas ou artificiais

2. **Conhecimento de APIs brasileiras**
   - PIX (Banco Central do Brasil)
   - NFe/SPED (Receita Federal)
   - LGPD (Lei Geral de Proteção de Dados)
   - Gateways de pagamento: Asaas, Mercado Pago, Efí, PagSeguro
   - Frameworks e bibliotecas nacionais

3. **Contexto de desenvolvimento brasileiro**
   - Padrões de código usados no Brasil
   - Boas práticas da comunidade brasileira
   - Problemas comuns no ecossistema nacional

---

## 🛠️ Stack Técnica

### Treinamento
- **Base**: Qwen2.5-Coder-32B-Instruct (Apache 2.0)
- **Framework**: PyTorch 2.2+, Transformers, DeepSpeed ZeRO-3
- **Técnicas**: Continued Pretraining, SFT, DPO, Knowledge Distillation
- **Hardware**: NVIDIA A100 (40GB ou 80GB)

### Inferência
- **Backend**: llama.cpp (GGUF)
- **API**: FastAPI + Uvicorn
- **CLI**: Click + Rich
- **VS Code Extension**: TypeScript + VS Code API

### Formatos Disponíveis
- **GGUF** (Q4_K_M, Q5_K_M, Q8_0, F16) — para llama.cpp, Ollama, LM Studio
- **Safetensors** — para uso direto com PyTorch/Transformers

---

## 📁 Estrutura do Projeto

```
Lloavece/
├── ARCHITECTURE.md          # Arquitetura técnica detalhada
├── README.md                # Este documento
├── LICENSE                  # Apache 2.0
│
├── data/                    # Datasets de treinamento
│   ├── raw/                 # Dados brutos (código, docs, etc)
│   ├── processed/           # Dados processados (limpos)
│   ├── sft/                 # Dataset SFT (instruções)
│   ├── dpo/                 # Dataset DPO (preferências)
│   └── distillation/        # Dataset de destilação
│
├── training/                # Scripts de treinamento
│   ├── continued_pretraining.py
│   ├── sft.py
│   ├── dpo.py
│   └── distillation.py
│
├── models/                  # Modelos treinados
│   ├── base/                # Modelo base (Qwen2.5-Coder-32B)
│   ├── cpt/                 # Modelo após continued pretraining
│   ├── sft/                 # Modelo após SFT
│   ├── dpo/                 # Modelo após DPO
│   └── gguf/                # Modelos quantizados (GGUF)
│
├── inference/               # Código de inferência
│   ├── server.py            # Servidor FastAPI
│   ├── cli.py               # Interface CLI
│   └── client.py            # Cliente Python
│
├── extension/               # VS Code Extension
│   ├── src/
│   │   ├── extension.ts
│   │   ├── provider.ts
│   │   └── client.ts
│   └── package.json
│
├── scripts/                 # Scripts auxiliares
│   ├── download_model.sh
│   ├── convert_gguf.sh
│   └── benchmark.py
│
└── docs/                    # Documentação
    ├── installation.md
    ├── usage.md
    ├── api.md
    └── contributing.md
```

---

## 🚀 Roadmap

### v0.1 — Prova de Conceito (4 semanas)
- [ ] Setup do ambiente (A100, PyTorch, DeepSpeed)
- [ ] Download do Qwen2.5-Coder-32B-Instruct
- [ ] Coleta de 1B tokens de código PT-BR
- [ ] Continued Pretraining (1 época)
- [ ] Avaliação inicial (perplexidade, qualidade)

### v0.2 — Fine-Tuning (6 semanas)
- [ ] Coleta de 50K instruções SFT
- [ ] Supervised Fine-Tuning (LoRA)
- [ ] Avaliação qualitativa (qualidade do código)
- [ ] Ajustes e iterações

### v1.0 — MVP (8 semanas)
- [ ] Dataset DPO (10K preferências)
- [ ] Direct Preference Optimization
- [ ] Quantização GGUF (Q4_K_M, Q5_K_M, Q8_0)
- [ ] Servidor FastAPI
- [ ] CLI funcional
- [ ] VS Code Extension (básica)

### v1.1 — Modelos Menores (4 semanas)
- [ ] Destilação para 14B
- [ ] Destilação para 4B
- [ ] Quantização GGUF (14B e 4B)
- [ ] Benchmark de performance

### v2.0 — Polimento (contínuo)
- [ ] Mais dados de treino
- [ ] Melhorias no SFT/DPO
- [ ] VS Code Extension avançada
- [ ] JetBrains Plugin
- [ ] API pública (opcional)

---

## 📊 Métricas de Sucesso

### Qualidade do Código
- **HumanEval** (code generation): >80% (32B), >70% (14B), >60% (4B)
- **MBPP** (code completion): >75% (32B), >65% (14B), >55% (4B)
- **CodeBLEU** (qualidade do código): >0.7 (32B), >0.6 (14B), >0.5 (4B)

### Performance
- **Velocidade de inferência**:
  - 32B (A100): >50 tokens/s
  - 14B (CPU): >10 tokens/s
  - 4B (CPU): >30 tokens/s
- **Latência** (primeiro token): <500ms
- **Uso de memória**: Dentro dos limites especificados

### Qualidade do Português
- **Fluência**: Avaliação humana (1-5, alvo: >4.5)
- **Precisão técnica**: Avaliação humana (1-5, alvo: >4.5)
- **Contexto brasileiro**: Avaliação humana (1-5, alvo: >4.0)

---

## 🔒 Licença

**Apache 2.0** — Licença permissiva que permite:
- ✅ Uso comercial
- ✅ Modificação
- ✅ Distribuição
- ✅ Uso privado
- ❌ Sem garantia (standard)

**Modelo Base**: Qwen2.5-Coder-32B-Instruct (Apache 2.0)

---

## 🤝 Contribuição

Quer contribuir? Veja [`docs/contributing.md`](docs/contributing.md)

### Áreas Prioritárias

1. **Coleta de dados**: Repositórios brasileiros, documentação PT-BR
2. **Dataset SFT**: Instruções de código em PT-BR
3. **Avaliação**: Testes de qualidade, benchmarks
4. **VS Code Extension**: Features, UI/UX
5. **Documentação**: Tutoriais, exemplos

---

## 📞 Contato

- **GitHub**: [github.com/alekkzsx/Lloavece](https://github.com/alekkzsx/Lloavece)
- **Discord**: (a definir)
- **Email**: (a definir)

---

## 📚 Referências

1. **Qwen2.5-Coder**: [github.com/QwenLM/Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)
2. **llama.cpp**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
3. **DeepSpeed**: [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
4. **HumanEval**: [github.com/openai/human-eval](https://github.com/openai/human-eval)
5. **MBPP**: [github.com/google-research/google-research/tree/master/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)

---

**Desenvolvido com orgulho no Brasil** 🇧🇷  
**Última atualização**: 11 de Novembro de 2026
