# 📜 Scripts - Lloavece Kode

Este diretório contém todos os scripts auxiliares para o pipeline de dados e treinamento do Lloavece Kode.

## 🗂️ Scripts Disponíveis

### 1. `setup_environment.py`

**Propósito:** Configura o ambiente de desenvolvimento e verifica dependências.

**O que faz:**
- Verifica versão do Python (3.11+)
- Detecta GPUs e VRAM disponíveis
- Verifica espaço em disco
- Valida dependências instaladas
- Cria estrutura de diretórios
- Configura arquivo `.env`

**Como usar:**
```bash
python scripts/setup_environment.py
```

**Output:**
- Mensagens de status no terminal
- Cria diretórios necessários
- Gera `.env` com configurações

---

### 2. `download_model.py`

**Propósito:** Baixa o modelo base Qwen3.8-27B do Hugging Face.

**O que faz:**
- Download do modelo completo (~55GB)
- Suporte a resume de download interrompido
- Verificação de integridade dos arquivos
- Atualização automática do `.env`

**Como usar:**
```bash
# Download básico
python scripts/download_model.py

# Com token do Hugging Face (recomendado)
python scripts/download_model.py --token hf_SEU_TOKEN

# Com verificação de integridade
python scripts/download_model.py --verify

# Especificando diretório de saída
python scripts/download_model.py --output-dir models/base/Qwen3.8-27B
```

**Opções:**
- `--token`: Token do Hugging Face para acesso
- `--verify`: Verifica integridade após download
- `--output-dir`: Diretório de destino
- `--resume`: Resume download interrompido

**Output:**
- Modelo salvo em `models/base/Qwen3.8-27B/`
- Arquivos: `config.json`, `tokenizer.json`, `*.safetensors`, etc.

---

### 3. `collect_data.py`

**Propósito:** Coleta dados brutos de múltiplas fontes (GitHub, Stack Overflow, etc).

**O que faz:**
- Coleta repositórios do GitHub
- Coleta perguntas/respostas do Stack Overflow
- Coleta repositórios brasileiros
- Gera estatísticas detalhadas
- Suporte a múltiplas fontes

**Como usar:**
```bash
# Coleta de todas as fontes
python scripts/collect_data.py

# Com token do GitHub (maior rate limit)
python scripts/collect_data.py --github-token SEU_TOKEN

# Apenas uma fonte específica
python scripts/collect_data.py --sources github

# Customizando limites
python scripts/collect_data.py \
    --sources github stackoverflow \
    --max-repos 1000 \
    --max-questions 5000
```

**Opções:**
- `--sources`: Fontes de dados (`github`, `stackoverflow`, `brazilian`)
- `--github-token`: Token do GitHub para API
- `--max-repos`: Número máximo de repositórios
- `--max-questions`: Número máximo de questões do SO
- `--output-dir`: Diretório de saída

**Output:**
- `data/raw/github_*.json`
- `data/raw/stackoverflow_*.json`
- `data/raw/collection_stats.json`

**Exemplo de output:**
```json
{
  "timestamp": "2026-11-11T...",
  "total_repos": 500,
  "total_questions": 2000,
  "total_size_mb": 250.5
}
```

---

### 4. `curate_data.py`

**Propósito:** Limpa e filtra os dados brutos aplicando múltiplos critérios de qualidade.

**O que faz:**
- Deduplicação (exata + MinHash aproximada)
- Filtro de qualidade (comprimento, vocabulário, repetição)
- Filtro de idioma (mantém PT-BR e EN)
- Filtro de toxicidade (spam, conteúdo ofensivo)
- Classificação de tipo (código, QA, documentação)
- Split train/validation/test (90/5/5)

**Como usar:**
```bash
# Curadoria padrão
python scripts/curate_data.py

# Customizando parâmetros
python scripts/curate_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --similarity-threshold 0.85 \
    --min-length 100 \
    --max-length 50000 \
    --languages pt en
```

**Opções:**
- `--input-dir`: Diretório de entrada (padrão: `data/raw`)
- `--output-dir`: Diretório de saída (padrão: `data/processed`)
- `--similarity-threshold`: Threshold para MinHash (0.0-1.0, padrão: 0.8)
- `--min-length`: Comprimento mínimo em caracteres (padrão: 50)
- `--max-length`: Comprimento máximo em caracteres (padrão: 100000)
- `--languages`: Idiomas a manter (padrão: `pt en`)

**Output:**
- `data/processed/train.jsonl` + `train.parquet`
- `data/processed/validation.jsonl` + `validation.parquet`
- `data/processed/test.jsonl` + `test.parquet`
- `data/processed/*_stats.json` (estatísticas detalhadas)

**Estatísticas geradas:**
```json
{
  "total_samples": 100000,
  "duplicates_removed": 15000,
  "quality_filtered": 8000,
  "language_filtered": 2000,
  "toxic_filtered": 500,
  "final_samples": 74500
}
```

---

### 5. `process_data.py`

**Propósito:** Transforma dados curados em formato pronto para treinamento (tokenização, packing).

**O que faz:**
- Tokenização com tokenizer do modelo
- Formatação baseada em tipo (código, QA, docs)
- Packing de sequências curtas (eficiência)
- Criação de input_ids e labels
- Estatísticas de tokens e sequências

**Como usar:**
```bash
# Processa apenas train
python scripts/process_data.py --split train

# Processa todos os splits
python scripts/process_data.py --all-splits

# Customizando
python scripts/process_data.py \
    --input-dir data/processed \
    --output-dir data/training \
    --model-name Qwen/Qwen3.8-27B \
    --max-seq-length 4096 \
    --all-splits

# Sem packing
python scripts/process_data.py --no-packing --split train
```

**Opções:**
- `--input-dir`: Diretório de entrada (padrão: `data/processed`)
- `--output-dir`: Diretório de saída (padrão: `data/training`)
- `--split`: Split para processar (`train`, `validation`, `test`)
- `--all-splits`: Processa todos os splits
- `--model-name`: Modelo para tokenizer (padrão: `Qwen/Qwen3.8-27B`)
- `--max-seq-length`: Comprimento máximo de sequência (padrão: 4096)
- `--no-packing`: Desabilita packing de sequências

**Output:**
- `data/training/train.jsonl` (sequências tokenizadas)
- `data/training/validation.jsonl`
- `data/training/test.jsonl`
- `data/training/*_process_stats.json`

**Estatísticas geradas:**
```json
{
  "split": "train",
  "num_sequences": 50000,
  "total_tokens": 25000000,
  "avg_tokens_per_sample": 500,
  "sequences_by_length": {
    "min": 100,
    "max": 4096,
    "mean": 3800,
    "median": 3900
  }
}
```

---

## 🔄 Fluxo Completo

```bash
# 1. Setup
python scripts/setup_environment.py

# 2. Download do modelo
python scripts/download_model.py --verify

# 3. Coleta de dados
python scripts/collect_data.py --github-token SEU_TOKEN

# 4. Curadoria
python scripts/curate_data.py

# 5. Processamento
python scripts/process_data.py --all-splits

# 6. Treinamento (ver training/scripts/train_cpt.py)
python training/scripts/train_cpt.py --train-data data/training/train.jsonl
```

---

## 📊 Verificando Progresso

### Durante Coleta
```bash
# Ver logs
tail -f logs/collect_data.log

# Ver estatísticas
cat data/raw/collection_stats.json
```

### Durante Curadoria
```bash
# Ver logs
tail -f logs/curate_data.log

# Ver estatísticas
cat data/processed/curate_stats.json
cat data/processed/train_stats.json
```

### Durante Processamento
```bash
# Ver logs
tail -f logs/process_data.log

# Ver estatísticas
cat data/training/train_process_stats.json
```

---

## 🎯 Dicas de Uso

### 1. Coleta de Dados

**Use token do GitHub:**
```bash
export GITHUB_TOKEN=seu_token_aqui
python scripts/collect_data.py --github-token $GITHUB_TOKEN
```

**Limite para testes:**
```bash
# Apenas 50 repositórios para teste rápido
python scripts/collect_data.py --sources github --max-repos 50
```

### 2. Curadoria

**Dataset pequeno (<100k amostras):**
```bash
python scripts/curate_data.py --similarity-threshold 0.7 --min-length 20
```

**Dataset grande (>1M amostras):**
```bash
python scripts/curate_data.py --similarity-threshold 0.9 --min-length 100
```

### 3. Processamento

**Para A100 40GB:**
```bash
python scripts/process_data.py --max-seq-length 4096 --all-splits
```

**Para A100 80GB (contexto maior):**
```bash
python scripts/process_data.py --max-seq-length 8192 --all-splits
```

---

## ⚠️ Troubleshooting

### Problema: "Out of Memory" durante curadoria

**Solução:** Reduza o tamanho do batch ou aumente swap
```bash
# Processe em chunks menores
python scripts/curate_data.py --max-length 20000
```

### Problema: "Tokenizer não encontrado"

**Solução:** Baixe o modelo primeiro
```bash
python scripts/download_model.py
python scripts/process_data.py --all-splits
```

### Problema: "Poucos dados após curadoria"

**Solução:** Reduza filtros
```bash
python scripts/curate_data.py \
    --similarity-threshold 0.7 \
    --min-length 20 \
    --max-length 200000
```

### Problema: "Deduplicação muito lenta"

**Solução:** Use apenas deduplicação exata
```bash
# Edite curate_data.py e comente a seção de MinHash
# Ou reduza num_perm
python scripts/curate_data.py --similarity-threshold 0.95
```

---

## 📚 Dependências

Todos os scripts usam as seguintes bibliotecas:

```bash
# Core
pip install transformers torch pandas numpy tqdm

# Opcionais (recomendados)
pip install datasketch langdetect pyarrow

# Para DeepSpeed
pip install deepspeed
```

---

## 🔧 Configuração Avançada

### Variáveis de Ambiente

Crie um arquivo `.env`:
```bash
GITHUB_TOKEN=seu_token_aqui
HF_TOKEN=seu_token_huggingface
WANDB_API_KEY=sua_chave_wandb
MODEL_PATH=models/base/Qwen3.8-27B
```

### Logs

Todos os scripts geram logs em `logs/`:
- `logs/setup_environment.log`
- `logs/download_model.log`
- `logs/collect_data.log`
- `logs/curate_data.log`
- `logs/process_data.log`

---

## 🚀 Próximos Scripts (Em Desenvolvimento)

- [ ] `evaluate_model.py` - Avaliação em benchmarks
- [ ] `quantize_model.py` - Quantização GGUF/AWQ
- [ ] `merge_lora.py` - Merge de adaptadores LoRA
- [ ] `export_gguf.py` - Export para formato GGUF

---

## 📞 Suporte

Para problemas ou dúvidas:
1. Verifique os logs em `logs/`
2. Consulte [DATA_PIPELINE.md](../DATA_PIPELINE.md)
3. Abra uma issue no GitHub

---

**Última atualização:** Novembro 2026  
**Versão:** 1.0
