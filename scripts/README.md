# Lloavece Kode - Scripts

Este diretório contém todos os scripts para o pipeline completo do Lloavece Kode.

## 📋 Índice

1. [Setup e Instalação](#setup-e-instalação)
2. [Coleta de Dados](#coleta-de-dados)
3. [Processamento de Dados](#processamento-de-dados)
4. [Treinamento](#treinamento)
5. [Avaliação](#avaliação)
6. [Deploy](#deploy)
7. [Utilitários](#utilitários)

---

## 🛠️ Setup e Instalação

### `setup_environment.py`

Configura o ambiente de desenvolvimento.

**Uso:**
```bash
python scripts/setup_environment.py
```

**O que faz:**
- Verifica Python 3.11+
- Detecta GPUs e VRAM disponível
- Verifica espaço em disco
- Valida dependências
- Cria estrutura de diretórios
- Configura arquivo `.env`

---

### `download_model.py`

Baixa o modelo base Qwen3.8-27B.

**Uso:**
```bash
# Download normal
python scripts/download_model.py

# Com verificação de integridade
python scripts/download_model.py --verify

# Com token do GitHub (maior rate limit)
python scripts/download_model.py --token SEU_TOKEN --verify

# Caminho customizado
python scripts/download_model.py --output-dir /caminho/custom/models
```

**Opções:**
- `--output-dir`: Diretório de saída (padrão: `models/base`)
- `--token`: Token do GitHub para maior rate limit
- `--verify`: Verifica integridade após download
- `--model-id`: ID do modelo no HuggingFace (padrão: `Qwen/Qwen3.8-27B`)

**Tempo estimado:** 2-6 horas (depende da conexão)

**Espaço necessário:** ~55GB

---

## 📥 Coleta de Dados

### `collect_data.py`

Coleta dados de múltiplas fontes para treinamento.

**Uso:**
```bash
# Coleta de todas as fontes
python scripts/collect_data.py

# Com token do GitHub (recomendado)
python scripts/collect_data.py --github-token SEU_TOKEN

# Apenas GitHub
python scripts/collect_data.py --sources github --max-repos 500

# Apenas Stack Overflow
python scripts/collect_data.py --sources stackoverflow --max-questions 5000

# Apenas repositórios brasileiros
python scripts/collect_data.py --sources brazilian --max-repos 200

# Configuração customizada
python scripts/collect_data.py \
    --sources github stackoverflow \
    --max-repos 1000 \
    --max-questions 2000 \
    --github-token SEU_TOKEN
```

**Fontes disponíveis:**
- `github`: Repositórios populares de código
- `stackoverflow`: Perguntas e respostas de alta qualidade
- `brazilian`: Repositórios brasileiros

**Opções:**
- `--sources`: Fontes a coletar (múltiplas)
- `--max-repos`: Máximo de repositórios (padrão: 1000)
- `--max-questions`: Máximo de questões do Stack Overflow (padrão: 5000)
- `--github-token`: Token do GitHub para API
- `--output-dir`: Diretório de saída (padrão: `data/raw`)

**Saída:**
- `data/raw/github_repos.json`: Metadados dos repositórios
- `data/raw/code_files.jsonl`: Arquivos de código
- `data/raw/stackoverflow.jsonl`: Q&A do Stack Overflow
- `data/raw/brazilian_repos.json`: Repositórios brasileiros

**Tempo estimado:** 1-3 horas

---

## 🔄 Processamento de Dados

### `curate_data.py`

Curadoria e limpeza dos dados coletados.

**Uso:**
```bash
# Curadoria padrão
python scripts/curate_data.py

# Com opções customizadas
python scripts/curate_data.py \
    --min-length 50 \
    --max-length 10000 \
    --similarity-threshold 0.7 \
    --languages python javascript java

# Apenas deduplicação
python scripts/curate_data.py --only-dedup

# Apenas filtragem
python scripts/curate_data.py --only-filter
```

**O que faz:**
1. **Deduplicação** (MinHash + exata)
2. **Filtros de qualidade**:
   - Comprimento mínimo/máximo
   - Diversidade de vocabulário
   - Detecção de repetição
3. **Detecção de idioma** (PT-BR + EN)
4. **Filtro de toxicidade**
5. **Classificação de conteúdo** (código, QA, docs)
6. **Split train/validation/test** (90/5/5)

**Opções:**
- `--input-dir`: Diretório de entrada (padrão: `data/raw`)
- `--output-dir`: Diretório de saída (padrão: `data/processed`)
- `--min-length`: Comprimento mínimo (padrão: 50)
- `--max-length`: Comprimento máximo (padrão: 10000)
- `--similarity-threshold`: Threshold de similaridade para dedup (padrão: 0.8)
- `--languages`: Idiomas a manter (padrão: python, javascript, typescript, java, go, rust)
- `--only-dedup`: Apenas deduplicação
- `--only-filter`: Apenas filtragem

**Saída:**
- `data/processed/train.jsonl`: Dados de treino
- `data/processed/validation.jsonl`: Dados de validação
- `data/processed/test.jsonl`: Dados de teste
- `data/processed/curate_stats.json`: Estatísticas detalhadas

**Tempo estimado:** 30-60 minutos

---

### `process_data.py`

Tokenização e preparação para treinamento.

**Uso:**
```bash
# Processamento padrão
python scripts/process_data.py

# Com opções customizadas
python scripts/process_data.py \
    --model-path models/base/Qwen3.8-27B \
    --max-seq-length 4096 \
    --packing \
    --all-splits

# Apenas train split
python scripts/process_data.py --split train

# Sem packing (mais simples)
python scripts/process_data.py --no-packing
```

**O que faz:**
1. **Tokenização** com tokenizer do modelo base
2. **Packing** de sequências curtas (eficiência 2-3x)
3. **Formatação** baseada em tipo de conteúdo
4. **Criação** de input_ids e labels
5. **Estatísticas** de tokens

**Opções:**
- `--model-path`: Caminho do modelo base (padrão: `models/base/Qwen3.8-27B`)
- `--input-dir`: Diretório de entrada (padrão: `data/processed`)
- `--output-dir`: Diretório de saída (padrão: `data/training`)
- `--max-seq-length`: Comprimento máximo de sequência (padrão: 4096)
- `--packing`: Ativa packing de sequências
- `--split`: Split específico a processar (train, validation, test)
- `--all-splits`: Processa todos os splits

**Saída:**
- `data/training/train.jsonl`: Dados tokenizados para treino
- `data/training/validation.jsonl`: Dados tokenizados para validação
- `data/training/test.jsonl`: Dados tokenizados para teste
- `data/training/train_process_stats.json`: Estatísticas

**Tempo estimado:** 20-40 minutos

---

## 🎓 Treinamento

### `train_cpt.py` (Continued Pretraining)

Treinamento CPT para adaptar o modelo ao domínio de código.

**Uso:**
```bash
# Treinamento básico
python training/scripts/train_cpt.py \
    --model-name-or-path models/base/Qwen3.8-27B \
    --train-data data/training/train.jsonl \
    --val-data data/training/validation.jsonl \
    --output-dir models/cpt

# Com LoRA (recomendado)
python training/scripts/train_cpt.py \
    --model-name-or-path models/base/Qwen3.8-27B \
    --train-data data/training/train.jsonl \
    --output-dir models/cpt \
    --use-lora \
    --lora-r 64 \
    --lora-alpha 128 \
    --num-train-epochs 2

# Com QLoRA (menos VRAM)
python training/scripts/train_cpt.py \
    --model-name-or-path models/base/Qwen3.8-27B \
    --train-data data/training/train.jsonl \
    --output-dir models/cpt \
    --use-qlora \
    --lora-r 64

# Com DeepSpeed
python training/scripts/train_cpt.py \
    --model-name-or-path models/base/Qwen3.8-27B \
    --train-data data/training/train.jsonl \
    --output-dir models/cpt \
    --deepspeed training/configs/deepspeed_zero2.json

# Com arquivo de configuração YAML
python training/scripts/train_cpt.py training/configs/cpt.yaml
```

**Opções principais:**
- `--model-name-or-path`: Modelo base
- `--train-data`: Dados de treino
- `--val-data`: Dados de validação
- `--output-dir`: Diretório de saída
- `--use-lora`: Usa LoRA (recomendado)
- `--use-qlora`: Usa QLoRA (4-bit)
- `--lora-r`: Rank do LoRA (padrão: 64)
- `--lora-alpha`: Alpha do LoRA (padrão: 128)
- `--num-train-epochs`: Número de épocas (padrão: 2)
- `--learning-rate`: Learning rate (padrão: 2e-4)
- `--per-device-train-batch-size`: Batch size (padrão: 2)
- `--gradient-accumulation-steps`: Gradient accumulation (padrão: 4)
- `--gradient-checkpointing`: Ativa gradient checkpointing
- `--use-wandb`: Log para Weights & Biases

**Tempo estimado:** 1-2 semanas (A100)

---

### `train_sft.py` (Supervised Fine-Tuning)

Treinamento SFT com instruções de código em PT-BR.

**Uso:**
```bash
# SFT com LoRA
python training/scripts/train_sft.py \
    --model-name-or-path models/cpt \
    --train-data data/sft/train.jsonl \
    --val-data data/sft/validation.jsonl \
    --output-dir models/sft \
    --use-lora \
    --num-train-epochs 3

# Com QLoRA
python training/scripts/train_sft.py \
    --model-name-or-path models/cpt \
    --train-data data/sft/train.jsonl \
    --output-dir models/sft \
    --use-qlora \
    --num-train-epochs 3

# Com configuração YAML
python training/scripts/train_sft.py training/configs/sft.yaml
```

**Opções principais:**
- `--model-name-or-path`: Modelo base (pós-CPT)
- `--train-data`: Dados de treino (instruções)
- `--val-data`: Dados de validação
- `--output-dir`: Diretório de saída
- `--use-lora`: Usa LoRA (recomendado)
- `--use-qlora`: Usa QLoRA
- `--lora-r`: Rank do LoRA (padrão: 64)
- `--lora-alpha`: Alpha do LoRA (padrão: 128)
- `--num-train-epochs`: Número de épocas (padrão: 3)
- `--learning-rate`: Learning rate (padrão: 2e-4)
- `--max-seq-length`: Comprimento máximo (padrão: 4096)
- `--train-on-inputs`: Treina também nos inputs

**Tempo estimado:** 1-2 semanas (A100)

---

### `train_dpo.py` (Direct Preference Optimization)

Treinamento DPO com pares de preferências.

**Uso:**
```bash
# DPO com LoRA
python training/scripts/train_dpo.py \
    --model-name-or-path models/sft \
    --train-data data/dpo/train.jsonl \
    --val-data data/dpo/validation.jsonl \
    --output-dir models/dpo \
    --use-lora \
    --num-train-epochs 2 \
    --beta 0.1

# Com configuração YAML
python training/scripts/train_dpo.py training/configs/dpo.yaml
```

**Formato dos dados (JSONL):**
```json
{
    "prompt": "Escreva uma função para calcular fibonacci...",
    "chosen": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "rejected": "def fibonacci(n):\n    return n"
}
```

**Opções principais:**
- `--model-name-or-path`: Modelo base (pós-SFT)
- `--train-data`: Pares de preferência
- `--val-data`: Dados de validação
- `--output-dir`: Diretório de saída
- `--use-lora`: Usa LoRA (recomendado)
- `--num-train-epochs`: Número de épocas (padrão: 2)
- `--beta`: Parâmetro beta do DPO (padrão: 0.1)
- `--learning-rate`: Learning rate (padrão: 5e-6)
- `--loss-type`: Tipo de loss (padrão: sigmoid)
- `--max-prompt-length`: Comprimento máximo do prompt (padrão: 2048)
- `--max-completion-length`: Comprimento máximo da resposta (padrão: 2048)

**Tempo estimado:** 3-7 dias (A100)

---

### `self_improvement.py` (Self-Improvement Loop)

Loop iterativo de auto-aprimoramento.

**Uso:**
```bash
# Self-improvement padrão
python training/scripts/self_improvement.py \
    --model-path models/dpo \
    --prompts-file data/self_improvement/prompts.jsonl \
    --output-dir models/self_improvement \
    --max-iterations 6 \
    --min-improvement 1.0

# Com opções customizadas
python training/scripts/self_improvement.py \
    --model-path models/dpo \
    --prompts-file data/self_improvement/prompts.jsonl \
    --output-dir models/self_improvement \
    --max-iterations 6 \
    --n-samples 5 \
    --min-improvement 1.5

# Com arquivo de configuração
python training/scripts/self_improvement.py \
    --config training/configs/self_improvement.json \
    --prompts-file data/self_improvement/prompts.jsonl
```

**Formato dos prompts (JSONL):**
```json
{
    "prompt": "Implemente uma função para ordenar uma lista usando quicksort em Python",
    "tests": ["assert quicksort([3,1,2]) == [1,2,3]"],
    "language": "python"
}
```

**O que faz:**
1. Gera N respostas por problema
2. Avalia qualidade (testes + métricas)
3. Cria pares (melhor, pior)
4. Treina DPO nos pares
5. Avalia melhoria
6. Repete até saturação

**Opções:**
- `--model-path`: Modelo base
- `--prompts-file`: Arquivo com prompts
- `--output-dir`: Diretório de saída
- `--max-iterations`: Máximo de iterações (padrão: 6)
- `--min-improvement`: Melhoria mínima % (padrão: 1.0)
- `--n-samples`: Respostas por problema (padrão: 5)
- `--temperature`: Temperatura de geração (padrão: 0.8)
- `--train-epochs`: Épocas por iteração (padrão: 1)

**Tempo estimado:** 5-6 semanas (6 iterações)

---

## 📊 Avaliação

### `evaluate_model.py`

Avalia o modelo em benchmarks padrão.

**Uso:**
```bash
# Avaliação completa
python scripts/evaluate_model.py \
    --model-path models/dpo \
    --output models/evaluations/results.json

# Avaliação rápida (limitada)
python scripts/evaluate_model.py \
    --model-path models/dpo \
    --output models/evaluations/results.json \
    --sample-limit 50

# Benchmarks específicos
python scripts/evaluate_model.py \
    --model-path models/dpo \
    --output models/evaluations/results.json \
    --benchmarks humaneval

# Forçar CPU
python scripts/evaluate_model.py \
    --model-path models/dpo \
    --output models/evaluations/results.json \
    --device cpu
```

**Benchmarks disponíveis:**
- `humaneval`: 164 problemas de Python
- `mbpp`: 974 problemas de Python
- `all`: Todos os benchmarks

**Opções:**
- `--model-path`: Modelo a avaliar
- `--output`: Arquivo de saída
- `--benchmarks`: Benchmarks a rodar (padrão: all)
- `--sample-limit`: Limite de amostras (para testes rápidos)
- `--device`: Device (cuda, cpu, auto)

**Saída:**
```json
{
    "model": "models/dpo",
    "benchmarks": {
        "humaneval": {
            "total": 164,
            "passed": 156,
            "pass@1": 95.12
        },
        "mbpp": {
            "total": 974,
            "passed": 892,
            "pass@1": 91.58
        }
    }
}
```

**Tempo estimado:** 30-60 minutos (completo)

---

## 🚀 Deploy

### `quantize_model.py`

Quantiza o modelo para formatos eficientes.

**Uso:**
```bash
# GGUF Q4_K_M (recomendado)
python scripts/quantize_model.py \
    --model-path models/dpo \
    --output-dir models/quantized \
    --method gguf \
    --gguf-quant-type Q4_K_M

# GGUF Q5_K_M (mais qualidade)
python scripts/quantize_model.py \
    --model-path models/dpo \
    --output-dir models/quantized \
    --method gguf \
    --gguf-quant-type Q5_K_M

# AWQ 4-bit
python scripts/quantize_model.py \
    --model-path models/dpo \
    --output-dir models/quantized \
    --method awq \
    --awq-bits 4 \
    --awq-group-size 128

# GPTQ 4-bit
python scripts/quantize_model.py \
    --model-path models/dpo \
    --output-dir models/quantized \
    --method gptq \
    --gptq-bits 4 \
    --gptq-group-size 128

# Com calibração customizada
python scripts/quantize_model.py \
    --model-path models/dpo \
    --output-dir models/quantized \
    --method awq \
    --calibration-samples 1024 \
    --calibration-dataset code
```

**Métodos disponíveis:**

#### GGUF (para llama.cpp)
- `Q2_K`: ~2-bit, menor tamanho, menor qualidade
- `Q3_K_S`: ~3-bit pequeno
- `Q3_K_M`: ~3-bit médio (bom equilíbrio)
- `Q4_K_S`: ~4-bit pequeno
- `Q4_K_M`: ~4-bit médio (**recomendado**)
- `Q5_K_S`: ~5-bit pequeno
- `Q5_K_M`: ~5-bit médio (alta qualidade)
- `Q6_K`: ~6-bit (quase sem perda)
- `Q8_0`: ~8-bit (mínima perda)

#### AWQ (Active-aware Weight Quantization)
- 4-bit ou 8-bit
- Rápido em GPU
- Boa qualidade

#### GPTQ (Post-Training Quantization)
- 2, 3, 4, ou 8-bit
- Alternativa ao AWQ

**Opções:**
- `--model-path`: Modelo a quantizar
- `--output-dir`: Diretório de saída
- `--method`: Método (gguf, awq, gptq)
- `--gguf-quant-type`: Tipo de quantização GGUF
- `--awq-bits`: Bits para AWQ (4 ou 8)
- `--awq-group-size`: Group size para AWQ
- `--gptq-bits`: Bits para GPTQ
- `--gptq-group-size`: Group size para GPTQ
- `--calibration-samples`: Amostras de calibração
- `--calibration-dataset`: Dataset de calibração

**Tamanhos esperados (27B modelo):**
- Q4_K_M: ~18GB
- Q5_K_M: ~20GB
- AWQ 4-bit: ~16GB
- GPTQ 4-bit: ~16GB

**Tempo estimado:** 1-3 horas

---

### `cli.py`

Interface de linha de comando para uso do modelo.

**Uso:**
```bash
# Modo chat interativo
python scripts/cli.py \
    --model-path models/quantized/lloavece-kode-q4_k_m.gguf \
    --mode chat

# Prompt único
python scripts/cli.py \
    --model-path models/dpo \
    --mode chat \
    --prompt "Explique o que é uma closure em Python"

# Code completion
python scripts/cli.py \
    --model-path models/dpo \
    --mode complete \
    --prompt "def fibonacci(n):" \
    --language python

# Code explanation
python scripts/cli.py \
    --model-path models/dpo \
    --mode explain \
    --prompt "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)" \
    --language python

# Code review
python scripts/cli.py \
    --model-path models/dpo \
    --mode review \
    --prompt "def process_data(data):\n    result = []\n    for i in range(len(data)):\n        result.append(data[i] * 2)\n    return result" \
    --language python

# Code generation
python scripts/cli.py \
    --model-path models/dpo \
    --mode generate \
    --prompt "Crie uma função que valida se um email é válido" \
    --language python

# Batch processing
python scripts/cli.py \
    --model-path models/dpo \
    --mode batch \
    --input-file data/batch/input.jsonl \
    --output-file data/batch/output.jsonl
```

**Modos disponíveis:**
- `chat`: Chat interativo
- `complete`: Completar código
- `explain`: Explicar código
- `review`: Revisar código
- `generate`: Gerar código
- `batch`: Processamento em lote

**Comandos no chat interativo:**
- `sair`/`exit`: Encerra o programa
- `limpar`: Limpa histórico
- `ajuda`: Mostra comandos

---

## 🔧 Utilitários

### Estrutura de Diretórios

```
Lloavece/
├── scripts/                    # Scripts auxiliares
│   ├── setup_environment.py    # Setup do ambiente
│   ├── download_model.py       # Download do modelo
│   ├── collect_data.py         # Coleta de dados
│   ├── curate_data.py          # Curadoria
│   ├── process_data.py         # Processamento
│   ├── evaluate_model.py       # Avaliação
│   ├── quantize_model.py       # Quantização
│   └── cli.py                  # Interface CLI
│
├── training/
│   └── scripts/
│       ├── train_cpt.py        # CPT
│       ├── train_sft.py        # SFT
│       ├── train_dpo.py        # DPO
│       └── self_improvement.py # Self-improvement
│
├── data/
│   ├── raw/                    # Dados brutos
│   ├── processed/              # Dados curados
│   └── training/               # Dados tokenizados
│
├── models/
│   ├── base/                   # Modelo base
│   ├── cpt/                    # Modelo pós-CPT
│   ├── sft/                    # Modelo pós-SFT
│   ├── dpo/                    # Modelo pós-DPO
│   └── quantized/              # Modelos quantizados
│
└── configs/                    # Configurações
    ├── cpt.yaml
    ├── sft.yaml
    └── dpo.yaml
```

---

## 📊 Fluxo Completo

```
1. setup_environment.py
   └── Configura ambiente

2. download_model.py
   └── Baixa Qwen3.8-27B

3. collect_data.py
   └── Coleta dados brutos

4. curate_data.py
   └── Curadoria e limpeza

5. process_data.py
   └── Tokenização

6. train_cpt.py
   └── Continued Pretraining

7. evaluate_model.py
   └── Avalia CPT

8. train_sft.py
   └── Supervised Fine-Tuning

9. evaluate_model.py
   └── Avalia SFT

10. train_dpo.py
    └── Direct Preference Optimization

11. evaluate_model.py
    └── Avalia DPO

12. self_improvement.py
    └── Self-Improvement Loop

13. evaluate_model.py
    └── Avalia Final

14. quantize_model.py
    └── Quantiza para deploy

15. cli.py
    └── Testa modelo final
```

---

## 🐛 Troubleshooting

### Problema: "CUDA out of memory"

**Solução:**
```bash
# Use QLoRA em vez de LoRA
python training/scripts/train_cpt.py --use-qlora

# Reduza batch size
python training/scripts/train_cpt.py --per-device-train-batch-size 1

# Aumente gradient accumulation
python training/scripts/train_cpt.py --gradient-accumulation-steps 8

# Ative gradient checkpointing
python training/scripts/train_cpt.py --gradient-checkpointing
```

### Problema: "Rate limit exceeded" (GitHub API)

**Solução:**
```bash
# Use token do GitHub
python scripts/collect_data.py --github-token SEU_TOKEN

# Reduza número de repositórios
python scripts/collect_data.py --max-repos 100
```

### Problema: "Modelo muito lento"

**Solução:**
```bash
# Use modelo quantizado
python scripts/quantize_model.py --method gguf --gguf-quant-type Q4_K_M

# Use device correto
python scripts/cli.py --device cuda
```

### Problema: "Qualidade baixa"

**Solução:**
```bash
# Aumente dados de treino
python scripts/collect_data.py --max-repos 2000

# Use mais épocas
python training/scripts/train_cpt.py --num-train-epochs 3

# Faça self-improvement
python training/scripts/self_improvement.py --max-iterations 6
```

---

## 📚 Recursos Adicionais

- [DATA_PIPELINE.md](../DATA_PIPELINE.md) - Guia completo do pipeline de dados
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Arquitetura técnica
- [QUICKSTART.md](../QUICKSTART.md) - Guia rápido
- [README.md](../README.md) - Visão geral do projeto

---

## 💡 Dicas

1. **Comece pequeno**: Teste com poucos dados primeiro
2. **Monitore sempre**: Use W&B para acompanhar métricas
3. **Salve checkpoints**: Mantenha múltiplas versões
4. **Avalie frequentemente**: Teste após cada fase
5. **Documente tudo**: Mantenha logs dos experimentos

---

## 🆘 Suporte

- **Issues**: https://github.com/Alekkzsx/Lloavece/issues
- **Documentação**: Veja arquivos `.md` no diretório raiz

---

**Lloavece Kode** - IA Local Brasileira para Código 🇧🇷
