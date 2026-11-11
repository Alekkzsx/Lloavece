# 🚀 Quick Start - Lloavece Kode

## ⚡ Começando em 5 Passos

### 1️⃣ Setup do Ambiente (2 minutos)

```bash
cd Documents/Github/Lloavece

# Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2️⃣ Download do Modelo Base (2-6 horas)

```bash
# Download do Qwen3.8-27B (~55GB)
python scripts/download_model.py --verify

# Com token do GitHub (recomendado)
python scripts/download_model.py --token SEU_TOKEN --verify
```

**Onde baixar:** O modelo será salvo em `models/base/Qwen3.8-27B/`

### 3️⃣ Coleta de Dados (1-2 horas)

```bash
# Coleta dados de múltiplas fontes
python scripts/collect_data.py --github-token SEU_TOKEN

# Ou apenas uma fonte
python scripts/collect_data.py --sources github --max-repos 500
```

**Onde estão:** `data/raw/`

### 4️⃣ Curadoria e Processamento (30 minutos)

```bash
# Curadoria (limpeza e filtros)
python scripts/curate_data.py

# Processamento (tokenização)
python scripts/process_data.py --all-splits
```

**Onde estão:**
- `data/processed/` - dados curados
- `data/training/` - dados prontos para treino

### 5️⃣ Treinamento CPT (1-2 semanas)

```bash
# Treinamento com configuração padrão
python training/scripts/train_cpt.py \
    --train-data data/training/train.jsonl \
    --val-data data/training/validation.jsonl \
    --output-dir models/cpt

# Com configuração customizada
python training/scripts/train_cpt.py \
    --train-data data/training/train.jsonl \
    --config training/configs/cpt.yaml
```

**Onde está:** `models/cpt/`

## 📊 Verificando Progresso

### Durante Coleta
```bash
tail -f logs/collect_data.log
cat data/raw/collection_stats.json
```

### Durante Curadoria
```bash
tail -f logs/curate_data.log
cat data/processed/curate_stats.json
```

### Durante Treinamento
```bash
tail -f logs/train_cpt.log
# Ou veja no Weights & Biases
wandb login
```

## 🎯 Comandos Úteis

### Ver estatísticas do dataset
```bash
python -c "
import json
with open('data/processed/train.jsonl') as f:
    lines = [json.loads(l) for l in f]
print(f'Total de amostras: {len(lines)}')
print(f'Total de tokens: {sum(len(l[\"input_ids\"]) for l in lines):,}')
"
```

### Testar modelo rapidamente
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = 'models/cpt'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

prompt = 'def fibonacci(n):'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
"
```

### Ver amostras do dataset
```bash
python -c "
import json
with open('data/processed/train.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        data = json.loads(line)
        print(f'--- Amostra {i+1} ---')
        print(data['content'][:300])
        print()
"
```

## ⚠️ Troubleshooting Rápido

### "Out of Memory" durante treino
```bash
# Reduza batch size ou aumente gradient accumulation
python training/scripts/train_cpt.py \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --train-data data/training/train.jsonl
```

### "Dataset muito pequeno"
```bash
# Reduza filtros de curadoria
python scripts/curate_data.py \
    --similarity-threshold 0.7 \
    --min-length 20
```

### "Tokenizer não encontrado"
```bash
# Baixe o modelo primeiro
python scripts/download_model.py
```

### Treino interrompido - como continuar
```bash
# Encontre o último checkpoint
ls models/cpt/checkpoint-*

# Resume do checkpoint
python training/scripts/train_cpt.py \
    --train-data data/training/train.jsonl \
    --resume-from-checkpoint models/cpt/checkpoint-1000
```

## 📚 Documentação Completa

- **Pipeline de Dados:** `DATA_PIPELINE.md`
- **Arquitetura:** `ARCHITECTURE.md`
- **Contribuição:** `docs/contributing.md`

## 🎓 Próximos Passos

Após o CPT completo:

1. **Avaliação:** `python scripts/evaluate_model.py`
2. **SFT:** `python training/scripts/train_sft.py`
3. **DPO:** `python training/scripts/train_dpo.py`
4. **Self-Improvement:** `python training/scripts/self_improvement.py`
5. **Quantização:** `python scripts/quantize_model.py`
6. **Deploy:** Ver `extension/` para VS Code

## 💡 Dicas Pro

1. **Use Colab Pro** para testes rápidos antes de rodar na A100
2. **Monitore com W&B** - configure uma vez, use sempre
3. **Salve checkpoints** - você vai querer voltar
4. **Teste com 1 época** antes de rodar 2-3 épocas completas
5. **Valide os dados** - inspecione manualmente 10-20 amostras

## 🆘 Precisa de Ajuda?

1. Verifique os logs em `logs/`
2. Veja o troubleshooting em `DATA_PIPELINE.md`
3. Abra uma issue no GitHub
4. Consulte a documentação do Qwen3.8-27B

---

**Pronto para começar? Execute o passo 1!** 🚀
