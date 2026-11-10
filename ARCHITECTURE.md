# Lloavece Kode — Arquitetura Técnica

## 🇧🇷 Visão Geral

**Lloavece Kode** é uma família de modelos de linguagem especializados em código, otimizados para desenvolvedores brasileiros. O diferencial é fluência em **Português do Brasil (PT-BR)** e conhecimento profundo do ecossistema de desenvolvimento brasileiro.

### Princípios Fundamentais

1. **100% Local**: Roda offline, sem API calls, sem cloud
2. **Especializado em Código**: Não é assistente geral, só sabe programar
3. **Brasileiro Nativo**: PT-BR fluente + contexto de código brasileiro
4. **Acessível**: Roda em hardware comum (laptops, desktops, servidores)
5. **Open Source**: Apache 2.0, sem restrições comerciais

---

## 🏗️ Arquitetura da Família

### Tamanhos do Modelo

| Modelo | Params | Tamanho GGUF (Q4_K_M) | RAM Mínima | Hardware Alvo |
|--------|--------|----------------------|------------|---------------|
| **lloavece-kode-32b** | 32B | ~19 GB | 24GB+ | GPU (RTX 3090/4090) ou A100 |
| **lloavece-kode-14b** | 14B | ~8.7 GB | 12GB | Desktop (CPU/GPU) |
| **lloavece-kode-4b** | 4B | ~2.5 GB | 4-8GB | Laptop (CPU) ← **MVP** |

### Por que 3 Tamanhos?

- **32B**: Máxima qualidade, para quem tem hardware
- **14B**: Equilíbrio qualidade/velocidade, para devs avançados
- **4B**: **MVP principal**, roda em laptops brasileiros típicos (8GB RAM)

---

## 🎯 Pipeline de Treinamento

### Fase 1: Base Model

**Modelo Base**: `Qwen2.5-Coder-32B-Instruct`
- Licença: Apache 2.0 (permissiva)
- Params: 32B (denso)
- Context: 32K tokens
- Já treinado em código de alta qualidade

**Por que não treinar do zero?**
- Treinar 32B do zero custa US$ 1-5M em GPUs
- Requer cluster de H100/A100 (não viável para projeto independente)
- Qwen2.5-Coder já é SOTA em código, melhor partir dele

### Fase 2: Continued Pretraining (CPT)

**Objetivo**: Adicionar conhecimento de código em PT-BR

**Dados de Treinamento** (estimativa: 10-50B tokens):

```
1. Código aberto brasileiro (GitHub)
   - Repositórios com README em PT-BR
   - Comentários em português
   - Variáveis com nomes em PT-BR

2. Documentação técnica brasileira
   - Tutoriais em português
   - Blog posts de devs brasileiros
   - Stack Overflow em PT-BR

3. APIs e frameworks brasileiros
   - PIX (BCB)
   - NFe/SPED (Receita Federal)
   - LGPD
   - Gateways: Asaas, Mercado Pago, Efí, PagSeguro
   - Frameworks nacionais

4. Livros e cursos em PT-BR
   - Livros de programação traduzidos
   - Cursos online brasileiros
   - Documentação oficial traduzida
```

**Técnicas de Treinamento**:
- **DeepSpeed ZeRO-3**: Distribui modelo em múltiplas GPUs (se necessário)
- **FSDP (Fully Sharded Data Parallel)**: PyTorch nativo, otimizado
- **Gradient Checkpointing**: Reduz uso de memória (trade-off: mais lento)
- **Mixed Precision (BF16)**: Acelera treino, reduz memória
- **Learning Rate Schedule**: Cosine annealing com warmup

**Hardware**: A100 (40GB ou 80GB)
- Tempo estimado: 1-4 semanas (depende do dataset)
- Custo: ~US$ 500-2000 (cloud) ou uso próprio

### Fase 3: Supervised Fine-Tuning (SFT)

**Objetivo**: Ensinar o modelo a seguir instruções em PT-BR

**Formato de Instruções**:

```json
{
  "instruction": "Crie uma função Python que valide um CPF",
  "input": "",
  "output": "def validar_cpf(cpf: str) -> bool:\n    # Remove caracteres não numéricos\n    cpf = ''.join(filter(str.isdigit, cpf))\n    \n    # Verifica se tem 11 dígitos\n    if len(cpf) != 11:\n        return False\n    \n    # Verifica se todos os dígitos são iguais\n    if cpf == cpf[0] * 11:\n        return False\n    \n    # Calcula primeiro dígito verificador\n    soma = sum(int(cpf[i]) * (10 - i) for i in range(9))\n    digito1 = (soma * 10) % 11\n    digito1 = 0 if digito1 == 10 else digito1\n    \n    # Verifica primeiro dígito\n    if int(cpf[9]) != digito1:\n        return False\n    \n    # Calcula segundo dígito verificador\n    soma = sum(int(cpf[i]) * (11 - i) for i in range(10))\n    digito2 = (soma * 10) % 11\n    digito2 = 0 if digito2 == 10 else digito2\n    \n    # Verifica segundo dígito\n    return int(cpf[10]) == digito2"
}
```

**Dataset SFT** (estimativa: 50K-200K exemplos):

```
1. Code Generation (40%)
   - "Crie uma função que..."
   - "Implemente um algoritmo de..."
   - "Escreva um código que..."

2. Code Explanation (20%)
   - "Explique este código: ..."
   - "O que esta função faz?"
   - "Como este algoritmo funciona?"

3. Code Review (20%)
   - "Revise este código e aponte problemas"
   - "Esta função tem algum bug?"
   - "Como melhorar este código?"

4. Debugging (15%)
   - "Por que este código dá erro?"
   - "Corrija este bug: ..."
   - "Este código não funciona, por quê?"

5. Contexto Brasileiro (5%)
   - "Como integrar PIX no meu sistema?"
   - "Como validar CPF/CNPJ?"
   - "Como emitir NFe?"
```

**Técnicas**:
- **Full Fine-Tuning**: Atualiza todos os parâmetros (melhor qualidade, mais lento)
- **LoRA (Low-Rank Adaptation)**: Treina apenas ~1% dos parâmetros (mais rápido)
- **QLoRA**: LoRA com quantização 4-bit (mais eficiente em memória)

**Recomendação**: Começar com **LoRA** para iterar rápido, depois fazer **Full Fine-Tuning** na versão final.

### Fase 4: DPO (Direct Preference Optimization)

**Objetivo**: Melhorar qualidade através de preferências humanas

**Formato**:

```json
{
  "prompt": "Crie uma função que calcule a média de uma lista",
  "chosen": "def calcular_media(numeros: list[float]) -> float:\n    if not numeros:\n        raise ValueError('Lista vazia')\n    return sum(numeros) / len(numeros)",
  "rejected": "def media(lista):\n    soma = 0\n    for n in lista:\n        soma += n\n    return soma / len(lista)"
}
```

**Por que DPO?**
- Melhora qualidade sem precisar de modelo de recompensa (RLHF é complexo)
- Ensina o modelo a preferir código limpo, tipado, com tratamento de erros
- Dataset menor que SFT (5K-20K exemplos)

### Fase 5: Quantização GGUF

**Objetivo**: Reduzir tamanho do modelo para rodar em hardware comum

**Formatos GGUF**:

| Quantização | Tamanho | Perplexidade | Uso |
|-------------|---------|--------------|-----|
| **Q4_K_M** | 40% | +2-3% | Principal (melhor equilíbrio) |
| **Q5_K_M** | 50% | +1-2% | Qualidade superior |
| **Q8_0** | 80% | +0.5% | Alta qualidade |
| **F16** | 100% | 0% | Referência (sem compressão) |

**Processo**:
```bash
# Converter para GGUF
python convert.py \
    --model lloavece-kode-32b \
    --outfile lloavece-kode-32b-f16.gguf \
    --outtype f16

# Quantizar
./quantize lloavece-kode-32b-f16.gguf lloavece-kode-32b-q4_k_m.gguf Q4_K_M
./quantize lloavece-kode-32b-f16.gguf lloavece-kode-32b-q5_k_m.gguf Q5_K_M
./quantize lloavece-kode-32b-f16.gguf lloavece-kode-32b-q8_0.gguf Q8_0
```

### Fase 6: Distilação (para modelos menores)

**Objetivo**: Criar versões 14B e 4B a partir do 32B

**Técnica**: Knowledge Distillation
- O modelo 32B (teacher) gera respostas de alta qualidade
- O modelo menor (student) aprende a imitar o teacher
- Loss: combinação de cross-entropy (ground truth) + KL divergence (teacher)

**Pipeline**:
```
Lloavece-Kode-32B (teacher)
    ↓ Gera respostas para dataset de treino
Dataset Destilado
    ↓ Treina modelo menor
Lloavece-Kode-14B ou Lloavece-Kode-4B (student)
```

---

## 🛠️ Stack Técnica

### Treinamento

```python
# Frameworks
PyTorch 2.2+
Transformers (Hugging Face)
DeepSpeed ZeRO-3
FSDP (PyTorch nativo)
Accelerate
PEFT (LoRA/QLoRA)

# Dados
datasets (Hugging Face)
tiktoken / sentencepiece (tokenização)

# Monitoramento
Weights & Biases (W&B)
TensorBoard

# Hardware
NVIDIA A100 (40GB ou 80GB)
CUDA 12.x
cuDNN 9.x
```

### Inferência

```python
# Backend
llama.cpp (GGUF)
llama-cpp-python (bindings Python)

# API
FastAPI (servidor REST)
Uvicorn (ASGI server)

# CLI
Click (framework CLI)
Rich (formatação terminal)
```

### VS Code Extension

```javascript
// Linguagem
TypeScript

// Framework
VS Code Extension API
webpack (build)

// Integração
llama-cpp-python (via API local)
```

---

## 📁 Estrutura do Projeto

```
Lloavece/
├── ARCHITECTURE.md          # Este documento
├── README.md                # Visão geral do projeto
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
│   ├── distillation.py
│   └── configs/
│       ├── cpt.yaml
│       ├── sft.yaml
│       └── dpo.yaml
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
│   │   ├── extension.ts     # Extensão principal
│   │   ├── provider.ts      # Provedor de completions
│   │   └── client.ts        # Cliente da API local
│   ├── package.json
│   └── webpack.config.js
│
├── scripts/                 # Scripts auxiliares
│   ├── download_model.sh    # Download do modelo base
│   ├── convert_gguf.sh      # Conversão para GGUF
│   ├── benchmark.py         # Benchmark de performance
│   └── collect_data.py      # Coleta de dados
│
├── notebooks/               # Notebooks Jupyter
│   ├── exploratory.ipynb    # Análise exploratória
│   ├── training.ipynb       # Visualização do treino
│   └── evaluation.ipynb     # Avaliação do modelo
│
└── docs/                    # Documentação
    ├── installation.md      # Como instalar
    ├── usage.md             # Como usar
    ├── api.md               # Documentação da API
    └── contributing.md      # Como contribuir
```

---

## 🚀 Roadmap

### v0.1 — Prova de Conceito (4 semanas)

**Objetivo**: Provar que a pipeline funciona

- [ ] Setup do ambiente (A100, PyTorch, DeepSpeed)
- [ ] Download do Qwen2.5-Coder-32B-Instruct
- [ ] Coleta de 1B tokens de código PT-BR
- [ ] Continued Pretraining (1 época)
- [ ] Avaliação inicial (perplexidade, qualidade)

**Entrega**: Modelo base funcional, rodando em A100

### v0.2 — Fine-Tuning (6 semanas)

**Objetivo**: Modelo especializado em código brasileiro

- [ ] Coleta de 50K instruções SFT
- [ ] Supervised Fine-Tuning (LoRA)
- [ ] Avaliação qualitativa (qualidade do código)
- [ ] Ajustes e iterações

**Entrega**: Lloavece-Kode-32B funcional, gerando código em PT-BR

### v1.0 — MVP (8 semanas)

**Objetivo**: Produto mínimo viável

- [ ] Dataset DPO (10K preferências)
- [ ] Direct Preference Optimization
- [ ] Quantização GGUF (Q4_K_M, Q5_K_M, Q8_0)
- [ ] Servidor FastAPI
- [ ] CLI funcional
- [ ] VS Code Extension (básica)

**Entrega**: Lloavece-Kode-32B completo, rodando localmente

### v1.1 — Modelos Menores (4 semanas)

**Objetivo**: Acessibilidade (laptops comuns)

- [ ] Destilação para 14B
- [ ] Destilação para 4B
- [ ] Quantização GGUF (14B e 4B)
- [ ] Benchmark de performance

**Entrega**: Família completa (32B, 14B, 4B)

### v2.0 — Polimento (contínuo)

**Objetivo**: Qualidade e features

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

- **Velocidade de inferência** (tokens/s):
  - 32B (A100): >50 tok/s
  - 14B (CPU): >10 tok/s
  - 4B (CPU): >30 tok/s
- **Latência** (primeiro token): <500ms
- **Uso de memória**: Dentro dos limites especificados

### Qualidade do Português

- **Fluência**: Avaliação humana (1-5, alvo: >4.5)
- **Precisão técnica**: Avaliação humana (1-5, alvo: >4.5)
- **Contexto brasileiro**: Avaliação humana (1-5, alvo: >4.0)

---

## 🔒 Licença e Uso

**Licença**: Apache 2.0
- ✅ Uso comercial permitido
- ✅ Modificação permitida
- ✅ Distribuição permitida
- ✅ Uso privado permitido
- ❌ Sem garantia (standard)

**Modelo Base**: Qwen2.5-Coder-32B-Instruct (Apache 2.0)
- Compatível com nossa licença
- Sem restrições adicionais

---

## 🤝 Contribuição

Quer contribuir? Veja `docs/contributing.md`

Áreas prioritárias:
1. **Coleta de dados**: Repositórios brasileiros, documentação PT-BR
2. **Dataset SFT**: Instruções de código em PT-BR
3. **Avaliação**: Testes de qualidade, benchmarks
4. **VS Code Extension**: Features, UI/UX
5. **Documentação**: Tutoriais, exemplos

---

## 📞 Contato

- **GitHub**: github.com/seu-usuario/lloavece
- **Discord**: (a definir)
- **Email**: (a definir)

---

## 📚 Referências

1. **Qwen2.5-Coder**: https://github.com/QwenLM/Qwen2.5-Coder
2. **llama.cpp**: https://github.com/ggerganov/llama.cpp
3. **DeepSpeed**: https://github.com/microsoft/DeepSpeed
4. **HumanEval**: https://github.com/openai/human-eval
5. **MBPP**: https://github.com/google-research/google-research/tree/master/mbpp

---

**Última atualização**: 11 de Novembro de 2026
