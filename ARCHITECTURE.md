# Lloavece Kode - Arquitetura Técnica

## 🎯 Visão

**Lloavece Kode** é um modelo de IA local especializado em engenharia de software, com fluência nativa em português brasileiro. O objetivo é ser um dos melhores modelos de código open-weight do mundo.

## 🧠 Modelo Base

### Qwen3.8-27B

| Aspecto | Especificação |
|---------|---------------|
| **Parâmetros** | 27B (dense) |
| **Licença** | Apache 2.0 (uso comercial liberado) |
| **Contexto Nativo** | 262,144 tokens |
| **Contexto Estendido** | Até 1,000,000 tokens (via YaRN) |
| **Arquitetura** | Causal LM com Vision Encoder |
| **Camadas** | 64 |
| **Hidden Dimension** | 5120 |
| **Vocabulário** | 248,320 tokens |

### Arquitetura Híbrida

```
16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
```

- **Gated DeltaNet**: Attention linear eficiente (48 heads V, 16 heads QK)
- **Gated Attention**: Attention tradicional (24 heads Q, 4 heads KV)
- **Feed Forward**: Intermediate dim = 17,408

### Benchmarks Baseline

| Benchmark | Score | Descrição |
|-----------|-------|-----------|
| SWE-bench Pro | 61.7 | Resolução de bugs reais |
| QwenSWEBench | 79.0 | Engenharia de software |
| Terminal Bench 2.1 | 73.0 | Coding agêntico |
| DeepSWE 1.1 | 42.2 | Tarefas complexas SWE |
| NL2Repo-Bench | 42.3 | Geração repo-level |

## 🎨 Features Especiais do Base

- **Thinking Mode**: Controle de raciocínio com `reasoning_effort`
- **Agentic Capabilities**: Planejamento e execução autônoma
- **Multimodal**: Compreensão de imagens e vídeos
- **Multi-Token Prediction**: Treinado com MTP

---

## 🏗️ Pipeline de Treinamento

### Fase 1: Continued Pretraining (CPT)

**Objetivo**: Especializar o modelo em código de alta qualidade

- **Duração**: 2-3 semanas
- **Dados**: 1-5B tokens de código curado
- **Técnica**: Causal LM training com learning rate baixo (1e-5 a 5e-5)
- **Ganho esperado**: +5-8% em benchmarks de código

**Fontes de dados**:
- GitHub repositories (Python, JavaScript, TypeScript, Java, Go, Rust)
- Stack Overflow (respostas de alta qualidade)
- Documentação técnica oficial
- Repositórios brasileiros de qualidade
- LeetCode, HackerRank, Codeforces

### Fase 2: Supervised Fine-Tuning (SFT)

**Objetivo**: Alinhar com instruções e português

- **Duração**: 2-3 semanas
- **Técnica**: LoRA (Low-Rank Adaptation)
  - r=16, alpha=32
  - Aplica em attention layers e MLP
  - Treina ~1-2% dos parâmetros
- **Dados**: 500k-1M exemplos de (instrução, resposta)
- **Learning rate**: 2e-4
- **Épocas**: 2-3
- **Ganho esperado**: +10-15% total

**Estrutura dos dados**:
```json
{
  "instruction": "Escreva uma função que calcule a sequência de Fibonacci",
  "input": "",
  "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
}
```

### Fase 3: Direct Preference Optimization (DPO)

**Objetivo**: Alinhar com preferências de qualidade

- **Duração**: 1 semana
- **Dados**: 100k-500k pares de (resposta preferida, resposta rejeitada)
- **Técnica**: DPO com β=0.1
- **Épocas**: 1-2
- **Ganho esperado**: +15-20% total

**Criação de preferências**:
- Múltiplas respostas por prompt (5-10)
- Avaliação por modelo juiz (GPT-4 ou similar)
- Filtro: diferença de score > 0.2

### Fase 4: Self-Improvement Loop

**Objetivo**: Auto-aprimoramento iterativo

- **Duração**: 5-6 semanas (5-6 iterações)
- **Processo**:
  1. Modelo gera respostas (n=5 por prompt)
  2. Avalia qualidade (testes + juiz LLM)
  3. Cria pares (melhor, pior)
  4. Treina DPO
  5. Repete
- **Ganho por iteração**: +2-4%
- **Ganho total**: +25-30% sobre base

**Critério de parada**:
- Melhoria < 1% por iteração
- Ou 6 iterações completadas

### Fase 5: Quantização

**Objetivo**: Otimizar para deployment

**Técnicas**:
- **GGUF Q4_K_M**: Padrão ouro, ~18GB
- **GGUF Q5_K_M**: Mais qualidade, ~22GB
- **AWQ 4-bit**: Alternativa rápida, ~16GB

**Hardware alvo**:
- GPU: 24GB+ VRAM (RTX 3090, RTX 4090)
- RAM: 40GB+ para inferência CPU

### Fase 6: Deploy

**Objetivo**: Integração com desenvolvedores

- **VS Code Extension**: Interface principal
- **CLI**: Para usuários avançados
- **API Local**: FastAPI para integrações

---

## 📊 Metas de Qualidade

### Benchmarks Alvo

| Benchmark | Base (Qwen3.8-27B) | Lloavece v1.0 | Melhoria |
|-----------|-------------------|---------------|----------|
| SWE-bench Pro | 61.7 | 70-75 | +15% |
| HumanEval | ~90% | 95-96% | +6% |
| MBPP | ~85% | 92-93% | +8% |
| LiveCodeBench | ~55% | 65-70% | +20% |
| Aider Code Editing | ~60% | 70-75% | +15% |

### Diferenciais

1. **Português fluente**: Instruções e explicações em PT-BR
2. **Código brasileiro**: Conhecimento de APIs e frameworks nacionais
3. **Local-first**: Roda 100% offline
4. **Open-source**: Apache 2.0, sem restrições

---

## 🛡️ Técnicas Anti-Forgetting

### 1. LoRA (Low-Rank Adaptation)

- Treina apenas adaptadores (1-2% dos parâmetros)
- Base fica congelada → preserva conhecimento geral
- Reduz catastrophic forgetting em 80%

### 2. Replay Buffer

- Mistura 70% dados de código + 30% dados gerais
- Previne que o modelo "esqueça" capacidades gerais
- Avalia capacidades gerais a cada checkpoint

### 3. Checkpoints Frequentes

- Salva modelo a cada 500 steps
- Permite rollback se qualidade degradar
- Mantém 10+ versões para seleção

### 4. Avaliação Dual

- Testa benchmarks de código (HumanEval, MBPP, etc)
- Testa capacidades gerais (MMLU, HellaSwag)
- Para se geral degradar > 5%

---

## 📁 Estrutura do Projeto

```
Lloavece/
├── data/
│   ├── raw/                    # Dados brutos
│   ├── processed/              # Dados limpos
│   ├── sft/                    # Instruções SFT
│   ├── dpo/                    # Preferências DPO
│   └── self_improvement/       # Dados do loop
│
├── models/
│   ├── base/                   # Qwen3.8-27B original
│   ├── cpt/                    # Após CPT
│   ├── sft/                    # Após SFT
│   ├── dpo/                    # Após DPO
│   ├── self_improvement/       # Iterações do loop
│   └── gguf/                   # Quantizados GGUF
│
├── training/
│   ├── configs/                # Configurações YAML
│   │   ├── cpt.yaml
│   │   ├── sft.yaml
│   │   ├── dpo.yaml
│   │   └── self_improvement.yaml
│   ├── scripts/
│   │   ├── train_cpt.py
│   │   ├── train_sft.py
│   │   ├── train_dpo.py
│   │   └── self_improvement_loop.py
│   └── utils/
│       ├── data_loader.py
│       ├── evaluation.py
│       └── metrics.py
│
├── scripts/
│   ├── setup_environment.py
│   ├── download_model.py
│   ├── collect_data.py
│   ├── curate_data.py
│   ├── evaluate_model.py
│   └── quantize_model.py
│
├── inference/
│   ├── llm_server.py           # API FastAPI
│   ├── code_completion.py      # Autocomplete
│   ├── code_review.py          # Review de código
│   └── code_explanation.py     # Explicação de código
│
├── extension/
│   ├── src/                    # VS Code extension
│   ├── package.json
│   └── README.md
│
├── docs/
│   ├── contributing.md
│   ├── training_guide.md
│   ├── deployment.md
│   └── benchmarks.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_evaluation_analysis.ipynb
│   └── 03_benchmark_results.ipynb
│
├── tests/
│   ├── test_inference.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── ARCHITECTURE.md
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔧 Stack Técnica

### Treinamento

- **Framework**: PyTorch 2.2+
- **Transformers**: Hugging Face Transformers 4.40+
- **PEFT**: LoRA, QLoRA
- **DeepSpeed**: ZeRO-3 (se múltiplas GPUs)
- **Accelerate**: Distributed training
- **W&B**: Monitoramento e logging

### Dados

- **Coleta**: GitHub API, Stack Exchange API, web scraping
- **Processamento**: pandas, numpy
- **Tokenização**: Transformers AutoTokenizer
- **Storage**: Parquet, JSONL

### Inferência

- **llama.cpp**: Para GGUF (CPU-friendly)
- **vLLM**: Para inferência rápida em GPU
- **FastAPI**: API REST
- **WebSocket**: Para streaming

### Deploy

- **VS Code Extension**: TypeScript
- **CLI**: Python (Click/Typer)
- **Docker**: Para deployment containerizado

---

## 📈 Timeline

### Mês 1: Setup e CPT

- Semana 1: Setup do ambiente, download do modelo
- Semana 2-3: Coleta e curadoria de dados
- Semana 3-4: CPT (Continued Pretraining)

### Mês 2: SFT e DPO

- Semana 1-2: SFT com LoRA
- Semana 3: DPO
- Semana 4: Avaliação inicial

### Mês 3: Self-Improvement

- Semana 1-6: 5-6 iterações de self-improvement
- Semana 7: Seleção do melhor checkpoint

### Mês 4: Deploy

- Semana 1: Quantização
- Semana 2: VS Code Extension (MVP)
- Semana 3: CLI e API
- Semana 4: Release v1.0

**Total**: 4 meses para v1.0

---

## 💰 Recursos Necessários

### Hardware

- **GPU**: A100 40GB/80GB (para treino)
- **RAM**: 64GB+ (para data processing)
- **Storage**: 500GB+ (modelos + dados)
- **Cloud**: Colab Pro (para tarefas auxiliares)

### Tempo

- **Treinamento**: ~3 meses
- **Deploy**: ~1 mês
- **Iteração contínua**: ongoing

### Custo Estimado

- **Cloud (A100)**: ~$500-1000/mês (se alugar)
- **Colab Pro**: $10/mês
- **Google Drive**: Gratuito (5TB disponível)
- **Total**: ~$500-1500 para v1.0

---

## 🎯 Métricas de Sucesso

### Qualidade

- SWE-bench Pro > 70
- HumanEval > 95%
- Aider > 70%
- Português fluente (avaliação humana)

### Performance

- Inferência > 30 tok/s (GPU)
- Inferência > 10 tok/s (CPU)
- Latência < 500ms (autocomplete)

### Adoção

- 1000+ downloads no primeiro mês
- 100+ usuários ativos
- 4.5+ estrelas no GitHub

---

## 🚀 Próximos Passos

1. ✅ Setup do ambiente
2. ✅ Download do Qwen3.8-27B
3. ⏳ Coleta de dados de código
4. ⏳ CPT (Continued Pretraining)
5. ⏳ SFT com LoRA
6. ⏳ DPO
7. ⏳ Self-Improvement Loop
8. ⏳ Quantização
9. ⏳ Deploy (VS Code Extension)

---

**Versão**: 1.0  
**Última atualização**: 2026-11-11  
**Status**: Em desenvolvimento
