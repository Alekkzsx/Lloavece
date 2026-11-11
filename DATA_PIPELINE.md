# 🔄 Pipeline de Dados - Lloavece Kode

Este documento descreve o fluxo completo de preparação de dados para treinamento do Lloavece Kode.

## 📋 Visão Geral

```
┌─────────────────┐
│   Coleta        │  collect_data.py
│   (Raw Data)    │  → GitHub, Stack Overflow, etc
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Curadoria     │  curate_data.py
│   (Limpeza)     │  → Deduplicação, qualidade, idioma
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Processamento  │  process_data.py
│  (Tokenização)  │  → Tokenização, packing, formatos
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Treinamento   │  train_cpt.py
│   (CPT/SFT)     │  → Modelo final
└─────────────────┘
```

## 🚀 Como Usar

### Passo 1: Coleta de Dados

Coleta dados brutos de múltiplas fontes:

```bash
# Coleta de todas as fontes
python scripts/collect_data.py

# Com token do GitHub (recomendado)
python scripts/collect_data.py --github-token SEU_TOKEN

# Apenas uma fonte específica
python scripts/collect_data.py --sources github --max-repos 500

# Customizando limites
python scripts/collect_data.py \
    --sources github stackoverflow \
    --max-repos 1000 \
    --max-questions 10000
```

**Output:** `data/raw/` com arquivos JSON/JSONL

### Passo 2: Curadoria

Limpa e filtra os dados brutos:

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

**O que faz:**
- ✅ Deduplicação (exata + MinHash aproximada)
- ✅ Filtro de qualidade (comprimento, vocabulário, repetição)
- ✅ Filtro de idioma (mantém PT-BR e EN)
- ✅ Filtro de toxicidade (spam, conteúdo ofensivo)
- ✅ Classificação de tipo (código, QA, documentação)
- ✅ Split train/validation/test (90/5/5)

**Output:** `data/processed/` com:
- `train.jsonl` + `train.parquet`
- `validation.jsonl` + `validation.parquet`
- `test.jsonl` + `test.parquet`
- `*_stats.json` com estatísticas

### Passo 3: Processamento

Transforma dados em formato de treinamento:

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
```

**O que faz:**
- ✅ Tokenização com tokenizer do modelo
- ✅ Formatação baseada em tipo (código, QA, docs)
- ✅ Packing de sequências curtas (eficiência)
- ✅ Criação de input_ids e labels
- ✅ Estatísticas de tokens e sequências

**Output:** `data/training/` com:
- `train.jsonl` (sequências tokenizadas)
- `validation.jsonl`
- `test.jsonl`
- `*_process_stats.json`

### Passo 4: Treinamento

```bash
# CPT (Continued Pretraining)
python training/scripts/train_cpt.py \
    --config training/configs/cpt.yaml \
    --data-dir data/training

# SFT (Supervised Fine-Tuning) - futuro
python training/scripts/train_sft.py \
    --config training/configs/sft.yaml \
    --data-dir data/training
```

## 📊 Estatísticas e Monitoramento

### Durante Coleta

```bash
# Ver progresso
tail -f logs/collect_data.log

# Ver estatísticas
cat data/raw/collection_stats.json
```

### Durante Curadoria

```bash
# Ver progresso
tail -f logs/curate_data.log

# Ver estatísticas
cat data/processed/curate_stats.json

# Ver estatísticas por split
cat data/processed/train_stats.json
cat data/processed/validation_stats.json
cat data/processed/test_stats.json
```

### Durante Processamento

```bash
# Ver progresso
tail -f logs/process_data.log

# Ver estatísticas
cat data/training/train_process_stats.json
```

## ⚙️ Configurações Importantes

### Curadoria (`curate_data.py`)

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--similarity-threshold` | 0.8 | Threshold para MinHash (0.0-1.0) |
| `--min-length` | 50 | Comprimento mínimo (caracteres) |
| `--max-length` | 100000 | Comprimento máximo (caracteres) |
| `--languages` | pt en | Idiomas a manter |

**Ajustes recomendados:**
- **Dataset pequeno (<100k):** `--similarity-threshold 0.7` (mais permissivo)
- **Dataset grande (>1M):** `--similarity-threshold 0.9` (mais rigoroso)
- **Foco em código:** `--min-length 200` (códigos curtos são ruins)
- **Contexto longo:** `--max-length 200000` (mas cuidado com VRAM)

### Processamento (`process_data.py`)

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--max-seq-length` | 4096 | Comprimento máximo de sequência |
| `--no-packing` | false | Desabilita packing |
| `--model-name` | Qwen/Qwen3.8-27B | Modelo para tokenizer |

**Ajustes recomendados:**
- **A100 40GB:** `--max-seq-length 4096` (balanceado)
- **A100 80GB:** `--max-seq-length 8192` (contexto maior)
- **Dataset com muitos textos curtos:** manter packing ativo
- **Dataset com textos longos:** `--no-packing`

## 🎯 Dicas de Qualidade

### 1. Qualidade > Quantidade

Melhor ter **100k exemplos excelentes** do que **10M exemplos ruins**.

**Como verificar qualidade:**
```bash
# Ver amostras aleatórias
python -c "
import json
with open('data/processed/train.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        data = json.loads(line)
        print(f'--- Amostra {i+1} ---')
        print(data['content'][:500])
        print()
"
```

### 2. Balanceamento de Tipos

Verifique se tem diversidade:

```bash
# Ver distribuição de tipos
python -c "
import json
from collections import Counter

types = []
with open('data/processed/train.jsonl') as f:
    for line in f:
        data = json.loads(line)
        types.append(data.get('content_type', 'unknown'))

print('Distribuição de tipos:')
for type, count in Counter(types).most_common():
    print(f'  {type}: {count:,} ({count/len(types)*100:.1f}%)')
"
```

**Distribuição ideal:**
- Código: 40-60%
- QA: 20-30%
- Documentação: 15-25%
- Geral: 5-15%

### 3. Verificar Duplicatas

Mesmo após curadoria, vale verificar:

```bash
# Ver duplicatas exatas
python -c "
import json
import hashlib

hashes = set()
duplicates = 0

with open('data/processed/train.jsonl') as f:
    for line in f:
        data = json.loads(line)
        h = hashlib.md5(data['content'].encode()).hexdigest()
        if h in hashes:
            duplicates += 1
        hashes.add(h)

print(f'Duplicatas encontradas: {duplicates}')
print(f'Taxa de duplicação: {duplicates/len(hashes)*100:.2f}%')
"
```

### 4. Estatísticas de Tokens

```bash
# Ver estatísticas de tokens
cat data/training/train_process_stats.json | python -m json.tool
```

**O que olhar:**
- `total_tokens`: deve ser >1M para CPT efetivo
- `avg_tokens_per_sample`: ideal entre 1000-3000
- `sequences_by_length.mean`: deve estar próximo de `max_seq_length`

## 🔧 Troubleshooting

### Problema: Poucos dados após curadoria

**Causa:** Filtros muito rigorosos

**Solução:**
```bash
python scripts/curate_data.py \
    --similarity-threshold 0.7 \
    --min-length 20 \
    --max-length 200000
```

### Problema: Out of Memory durante processamento

**Causa:** Sequências muito longas

**Solução:**
```bash
python scripts/process_data.py --max-seq-length 2048
```

### Problema: Tokenizer não encontrado

**Causa:** Modelo não baixado ainda

**Solução:**
```bash
# Baixe o modelo primeiro
python scripts/download_model.py

# Ou use modelo menor para teste
python scripts/process_data.py --model-name Qwen/Qwen2.5-7B
```

### Problema: Packing ineficiente

**Causa:** Todas as sequências já são longas

**Solução:**
```bash
# Desabilite packing
python scripts/process_data.py --no-packing
```

## 📈 Métricas de Sucesso

Após o pipeline completo, você deve ter:

| Métrica | Valor Esperado | Como Verificar |
|---------|----------------|----------------|
| **Total de amostras** | >100k | `data/processed/curate_stats.json` |
| **Total de tokens** | >1M | `data/training/train_process_stats.json` |
| **Taxa de remoção** | 30-70% | `curate_stats.json` → stats |
| **Diversidade** | 3+ tipos | Verificar distribuição |
| **Qualidade** | Amostras limpas | Inspeção manual |

## 🚀 Próximos Passos

Após dados prontos:

1. **Iniciar CPT:**
   ```bash
   python training/scripts/train_cpt.py --config training/configs/cpt.yaml
   ```

2. **Monitorar treino:**
   ```bash
   # Se usando W&B
   wandb login
   # O treino vai logar automaticamente
   ```

3. **Avaliar modelo:**
   ```bash
   python scripts/evaluate_model.py --checkpoint models/cpt/best_model.pt
   ```

## 📚 Referências

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Data Quality in LLM Training](https://arxiv.org/abs/2401.06785)
- [The RefinedWeb Dataset](https://arxiv.org/abs/2306.01116)
- [Dolma Dataset](https://blog.eleuther.ai/dolma/)

---

**Última atualização:** Novembro 2026  
**Versão:** 1.0
