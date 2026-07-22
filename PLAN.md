# 🇧🇷 IA BRASILEIRA — PLANO COMPLETO DE CONSTRUÇÃO
> Uma Inteligência Artificial criada do zero, com alma brasileira, capaz de rivalizar com os melhores modelos do mundo.

---

## Índice

1. [Visão e Filosofia](#1-visão-e-filosofia)
2. [O que é uma IA de verdade](#2-o-que-é-uma-ia-de-verdade)
3. [Objetivo e Benchmarks-Alvo](#3-objetivo-e-benchmarks-alvo)
4. [FASE 0 — Fundação Científica](#fase-0--fundação-científica-meses-12)
5. [FASE 1 — Dados Brasileiros](#fase-1--dados-brasileiros-meses-16)
6. [FASE 2 — Tokenizer Próprio](#fase-2--tokenizer-próprio-meses-23)
7. [FASE 3 — Arquitetura do Modelo](#fase-3--arquitetura-do-modelo-meses-25)
8. [FASE 4 — Pré-Treinamento](#fase-4--pré-treinamento-meses-610)
9. [FASE 5 — Fine-Tuning e Alinhamento](#fase-5--fine-tuning-e-alinhamento-meses-1014)
10. [FASE 6 — Avaliação e Benchmarks](#fase-6--avaliação-e-benchmarks-meses-1215)
11. [FASE 7 — Produto e Deploy](#fase-7--produto-e-deploy-meses-1418)
12. [FASE 8 — Escala e Evolução](#fase-8--escala-e-evolução-meses-18)
13. [Infraestrutura de Computação](#infraestrutura-de-computação)
14. [Stack Tecnológico](#stack-tecnológico)
15. [Equipe e Papéis](#equipe-e-papéis)
16. [Financiamento e Custos](#financiamento-e-custos)
17. [Riscos e Mitigações](#riscos-e-mitigações)
18. [Cronograma Geral](#cronograma-geral)
19. [Definição de Sucesso](#definição-de-sucesso)
20. [Estrutura de Pastas do Projeto](#estrutura-de-pastas-do-projeto)
21. [Papers Obrigatórios](#papers-obrigatórios)
22. [Próximos Passos Imediatos](#próximos-passos-imediatos)

---

## 1. Visão e Filosofia

Esta IA não é um wrapper de ChatGPT. Não é uma ferramenta que usa API de terceiros. Não é uma cópia de nada.

**É uma inteligência artificial construída do zero:**
- Arquitetura desenhada por nós
- Tokenizer treinado em português do Brasil
- Dados coletados, limpos e curados por nós
- Treinamento executado em nossa infraestrutura
- Identidade, valores e cultura 100% brasileiros

### Filosofia Central

```
"Uma IA que pensa em português não apenas traduz —
 ela compreende o Brasil de dentro pra fora."
```

A língua portuguesa, especialmente o português brasileiro, não é apenas palavras.
É ritmo, gíria, ironia, regionalismo, história, direito, cultura, diversidade.
Um modelo treinado em inglês e "traduzido" jamais captará isso. **Nós vamos capturar.**

### Princípios

1. **Soberania Tecnológica**: nenhuma dependência crítica de APIs estrangeiras
2. **Transparência**: arquitetura e dados documentados publicamente (quando possível)
3. **Identidade**: a IA sabe que é brasileira, e isso é uma vantagem, não uma limitação
4. **Responsabilidade**: alinhada com a CF/88, LGPD e valores democráticos brasileiros
5. **Escalabilidade**: construída para crescer, não para ser um protótipo eterno

---

## 2. O que é uma IA de Verdade

Para evitar confusão: o que estamos construindo e o que não estamos.

### O que NÃO é uma "IA de verdade" nesse contexto:
- Um chatbot com regras hard-coded
- Um sistema de busca com respostas pré-escritas
- Um wrapper que chama GPT/Gemini por baixo
- Um modelo fine-tuned em cima de LLaMA sem entender nada
- Um classificador simples ou modelo de ML clássico

### O que É uma "IA de verdade":

Um **Large Language Model (LLM)** treinado do zero:

```
Texto de entrada (tokens)
        |
   [Embedding Layer]     — Converte tokens em vetores densos
        |
   [32x Transformer]     — Cada bloco processa relações entre tokens
   [  Block          ]   — Atencao Multi-Head + Feed-Forward
        |
   [RMSNorm Final]       — Normalizacao
        |
   [Language Model Head] — Projeta de volta ao vocabulario
        |
Distribuicao de probabilidade sobre proximo token
        |
Amostragem (temperature, top-p)
        |
     Resposta
```

Um LLM genuíno:
- Aprende representações do mundo a partir de texto bruto
- Generaliza para tarefas que nunca viu explicitamente
- Raciocina por composição de conceitos aprendidos
- Mantém coerência em contextos longos
- Pode aprender novas tarefas com poucos exemplos (few-shot)

**Nosso modelo terá 7 bilhões de parâmetros** — cada parâmetro é um número aprendido
durante o treinamento. São 7.000.000.000 pesos numa rede neural que aprende a modelar a linguagem humana.

---

## 3. Objetivo e Benchmarks-Alvo

### Meta Principal
> Superar o **Gemini 2 Flash** em benchmarks de português brasileiro, e competir de
> igual para igual nos benchmarks gerais em inglês.

### Por que o Gemini 2 Flash é a referência?
- É um modelo proprietário do Google (~8B parâmetros efetivos)
- Altamente otimizado, rápido, amplamente usado
- Forte em múltiplas línguas
- Representa o estado da arte em modelos "flash" (rápidos e eficientes)
- Bater ele prova que chegamos ao nível de produção de classe mundial

### Benchmarks Quantitativos

| Benchmark | Gemini 2 Flash | Nossa Meta | Nossa Vantagem |
|-----------|---------------|------------|----------------|
| MMLU (57 áreas) | ~78% | >75% | — |
| HumanEval (código) | ~74% | >70% | — |
| GSM8K (matemática) | ~80% | >75% | — |
| MT-Bench (diálogo) | ~8.2/10 | >8.0/10 | — |
| ARC-Challenge | ~77% | >75% | — |
| TruthfulQA | ~68% | >70% | NOSSA |
| **ENEM histórico** | ~78% | **>92%** | **NOSSA** |
| **ASSIN2 (PT-BR NLI)** | ~71% | **>88%** | **NOSSA** |
| **Portuguese SQUAD** | ~74% | **>90%** | **NOSSA** |
| **OAB Exame** | ~65% | **>85%** | **NOSSA** |
| **Avaliação humana PT-BR** | baseline | **>60% preferência** | **NOSSA** |

### Estratégia de Competição

1. **Dominar o PT-BR** completamente — aqui nenhum modelo estrangeiro nos bate
2. **Paridade em benchmarks gerais** — competir de igual no MMLU, HumanEval, etc.
3. **Eficiência** — ser tão bom quanto Gemini Flash com custo de inferência menor
4. **Confiabilidade** — menos alucinações sobre contexto brasileiro

---

## FASE 0 — Fundação Científica (Meses 1-2)

### Objetivo
Construir o conhecimento técnico necessário, montar a equipe e configurar o ambiente de desenvolvimento.

### 0.1 Dominar os Fundamentos Matemáticos

Sem isso, você não implementa nem depura nada. É inegociável.

**Álgebra Linear:**
- Multiplicação de matrizes (operação central de todo Transformer)
- Produto interno, normas, autovalores/autovetores
- Decomposição SVD (usada em LoRA)
- Operações em tensores multidimensionais

**Cálculo:**
- Derivadas parciais
- Regra da cadeia (É o backpropagation)
- Gradiente, jacobiano, hessiano
- Otimização por gradiente descendente

**Probabilidade e Estatística:**
- Distribuições de probabilidade
- Entropia cruzada (nossa função de loss)
- KL Divergence (usada no RLHF/DPO)
- Teorema de Bayes

**Livros (todos gratuitos online):**
- "Mathematics for Machine Learning" (Deisenroth et al.)
- "Deep Learning" (Goodfellow, Bengio, Courville)
- "The Matrix Calculus You Need For Deep Learning" (Parr & Howard)

### 0.2 Dominar PyTorch Profundo

Não apenas usar — **entender o que acontece por baixo**.

```python
# Você precisa entender profundamente:
# - autograd rastreia operações num grafo computacional
# - quando usar .detach(), .clone(), .contiguous()
# - memory layout: contiguous vs. non-contiguous tensors
# - como debugar NaN em gradientes
# - broadcasting rules
# - como escrever custom autograd functions
# - como nn.Module funciona internamente
```

### 0.3 Implementar Transformers do Zero (Exercício Obrigatório)

Sequência crescente de complexidade — NÃO pular etapas:
1. Bigramas simples — prever próximo caractere
2. MLP simples para linguagem
3. Self-attention do zero
4. Multi-head attention do zero
5. Transformer block completo
6. Mini-GPT completo (treinar em texto pequeno)
7. Escalar para modelo real

**Referência principal:** Andrej Karpathy — "Let's build GPT from scratch"
(YouTube + repositório github.com/karpathy/nanoGPT)

### 0.4 Configuração do Ambiente

```bash
# Sistema: Ubuntu 22.04 LTS + CUDA 12.1+ + Python 3.11+

python -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton deepspeed wandb
pip install numpy pandas tqdm rich pytest black isort
```

---

## FASE 1 — Dados Brasileiros (Meses 1-6)

> Esta é a fase mais importante do projeto inteiro.
> **Dados ruins = modelo ruim, independente da arquitetura.**

### 1.1 Meta de Dados

**~500 Bilhões de tokens PT-BR de alta qualidade**

### 1.2 Fontes de Dados

#### Texto Web (maior volume)

| Fonte | Tokens Estimados | Estratégia |
|-------|-----------------|------------|
| Common Crawl BR (filtrado) | ~200B | Pipeline rigoroso de filtragem |
| Arquivos de notícias BR | ~30B | Filtragem por data e domínio |
| Blogs e artigos técnicos | ~15B | Score de qualidade |

**Filtrar Common Crawl:**
```
1. Baixar WARC files do S3 público (s3://commoncrawl/)
2. Extrair texto (trafilatura)
3. Classificar idioma (fastText lid.176.bin)
4. Manter apenas pt-BR com score > 0.85
5. Aplicar pipeline completo de qualidade
```

#### Conteúdo Enciclopédico

| Fonte | Tokens | Como Obter |
|-------|--------|------------|
| Wikipedia PT-BR completa | ~2B | dumps.wikimedia.org |
| Wikicionário PT | ~200M | dumps.wikimedia.org |
| Wikisource PT | ~500M | dumps.wikimedia.org |

#### Conteúdo Oficial e Institucional

| Fonte | Tokens | Como Obter |
|-------|--------|------------|
| Diário Oficial da União (histórico) | ~20B | Portal Imprensa Nacional |
| Legislação Federal | ~5B | API dados.camara.leg.br |
| Jurisprudência STF | ~10B | Portal STF dados abertos |
| Jurisprudência STJ | ~8B | Portal STJ |
| Jurisprudência TST | ~5B | Portal TST |
| Outros tribunais | ~10B | Portais públicos |
| Diários Oficiais Estaduais | ~15B | Portais estaduais |

#### Conteúdo Acadêmico

| Fonte | Tokens | Como Obter |
|-------|--------|------------|
| SciELO (artigos científicos BR) | ~8B | API SciELO |
| CAPES (teses e dissertações) | ~15B | Portal CAPES |
| BDTD (Biblioteca Digital) | ~10B | protocolo OAI-PMH |

#### Livros e Literatura

| Fonte | Tokens | Como Obter |
|-------|--------|------------|
| BN Digital (Biblioteca Nacional) | ~5B | Portal BN |
| Domínio Público MEC | ~2B | dominiopublico.gov.br |
| Gutenberg PT | ~1B | gutenberg.org |
| Literatura brasileira clássica | ~2B | Curadoria manual + OCR |

#### Código e Tecnologia

| Fonte | Tokens | Como Obter |
|-------|--------|------------|
| GitHub (repos com PT-BR) | ~20B | GitHub Archive |
| Stack Overflow PT-BR | ~3B | Dump público |
| Documentações técnicas BR | ~2B | Curadoria |

#### Dados Sintéticos Controlados

| Tipo | Tokens | Método |
|------|--------|--------|
| Pares instrução-resposta | ~50B | Geração + filtragem rigorosa |
| Cadeia de raciocínio (CoT) | ~20B | Geração estruturada |
| Diálogos simulados | ~15B | Geração + revisão humana |
| Código com comentários PT-BR | ~10B | Geração controlada |
| Dados matemáticos | ~5B | Geração + verificação |

**TOTAL META: ~500B tokens limpos e únicos**

### 1.3 Pipeline de Processamento

```
COLETA RAW
    |
    v
[1. EXTRACAO DE TEXTO]
    - HTML -> texto puro (trafilatura)
    - PDF -> texto (pdfminer)
    - DOCX -> texto (python-docx)
    |
    v
[2. DETECCAO DE IDIOMA]
    - fastText lid.176.bin
    - Manter: pt-BR score > 0.85
    |
    v
[3. DEDUPLICACAO]
    - MinHash LSH (Jaccard similarity)
    - Threshold: 0.8
    - Near-duplicate em nivel de paragrafo
    |
    v
[4. FILTRAGEM DE QUALIDADE]
    - Remover: < 50 palavras
    - Remover: > 50% numeros/simbolos
    - Remover: repeticoes excessivas
    - Score de qualidade via modelo de ML
    |
    v
[5. FILTRAGEM DE SEGURANCA]
    - Remover: CPF, RG, telefone (regex)
    - Remover: conteudo adulto explicito
    - Remover: spam e bots
    |
    v
[6. NORMALIZACAO]
    - Encoding: tudo UTF-8
    - Normalizar espacos e quebras de linha
    |
    v
[7. TOKENIZACAO]
    - Aplicar tokenizer BPE proprio
    - Salvar em binario (.bin)
    |
    v
CORPUS FINAL (~500B tokens)
```

### 1.4 Infraestrutura de Dados

```
Storage:
- Dataset raw: ~50TB
- Dataset processado: ~5TB (tokens binários)
- Formato: numpy memmap, 1000 shards de ~500M tokens

Processamento:
- Apache Spark 3.x para deduplicação em escala
- Ray para processamento paralelo
- Redis para orquestração de filas
```

---

## FASE 2 — Tokenizer Próprio (Meses 2-3)

### 2.1 Por que um tokenizer próprio é crítico

**Problema com tokenizers americanos em PT-BR:**
- GPT-4 usa 40% mais tokens para português que para inglês
- Custo de inferência 40% maior para textos equivalentes
- "acentuação" pode virar ["acent", "ua", "ção"] — ineficiente
- "ção" (sufixo muito comum) não é um token único nesses tokenizers

**Nossa solução:** BPE treinado 100% em corpus PT-BR.
**Objetivo:** fertilidade < 1.3 para PT-BR (vs. ~1.8 do GPT-4)

### 2.2 Algoritmo BPE (Byte-Pair Encoding)

```
1. Iniciar com vocabulário de caracteres individuais
2. Contar frequência de pares consecutivos no corpus
3. Mesclar o par mais frequente em um novo token
4. Repetir até atingir tamanho de vocabulário alvo
5. Resultado: tokens frequentes em PT-BR são tokens únicos

Exemplo esperado:
"configuração"    -> ["configuração"]      (1 token)
"ção"             -> ["ção"]              (1 token — sufixo frequente)
"desenvolvimento" -> ["desenvolvimento"]   (1 token — palavra técnica)
```

### 2.3 Especificações

```python
VOCAB_SIZE = 65536       # 2^16 — suficiente para PT-BR rico

SPECIAL_TOKENS = {
    "<|bos|>": 0,        # início de sequência
    "<|eos|>": 1,        # fim de sequência
    "<|pad|>": 2,        # padding
    "<|unk|>": 3,        # token desconhecido
    "<|sep|>": 4,        # separador
    "<|sys|>": 5,        # mensagem de sistema
    "<|user|>": 6,       # mensagem do usuário
    "<|assistant|>": 7,  # resposta do assistente
    # 8 tokens extras para extensibilidade futura
}
```

### 2.4 Dados para Treinar o Tokenizer

- 10B tokens representativos do corpus completo
- Inclui: texto formal, informal, código, matemática, jurídico
- Distribuído proporcionalmente aos domínios do pré-treinamento

### 2.5 Implementação — Módulos em `src/tokenizer/`

- `bpe_trainer.py` — treina o BPE no corpus
- `tokenizer.py` — encode/decode eficiente
- `vocab_builder.py` — constrói e serializa o vocabulário
- `special_tokens.py` — gerenciamento de tokens especiais

---

## FASE 3 — Arquitetura do Modelo (Meses 2-5)

### 3.1 Tipo de Arquitetura

**Decoder-only Transformer** (autoregressive language model)

Por quê? Todos os top models usam isso (GPT-4, LLaMA, Gemini, Mistral, Claude).
Ideal para geração de texto. Escala comprovada para bilhões de parâmetros.
Treinamento simples: next-token prediction (cross-entropy loss).

### 3.2 Hiperparâmetros — Modelo 7B

```python
MODEL_CONFIG = {
    # Dimensoes
    "vocab_size":    65536,    # nosso tokenizer proprio
    "d_model":        4096,    # dimensao dos embeddings
    "n_layers":         32,    # numero de camadas Transformer
    "n_heads":          32,    # heads de atencao
    "n_kv_heads":        8,    # heads KV — Grouped Query Attention
    "d_ffn":         14336,    # dimensao da FFN com SwiGLU

    # Contexto
    "max_seq_len":   32768,    # 32k tokens de contexto

    # Atencao
    "rope_theta":   500000,    # RoPE base frequency (estendido para PT-BR)

    # Normalizacao
    "rms_norm_eps":   1e-5,

    # Parametros totais: ~7.1B
}

# Calculo de parametros:
# Embedding:        65536 x 4096           =  268M
# Por camada:
#   Atencao (GQA):                         ~  42M
#   FFN (SwiGLU):   4096 x 14336 x 3      = 176M
#   x 32 camadas                           ~  6.8B
# LM Head:          4096 x 65536           =  268M
# Total:                                   ~ 7.1B parametros
```

### 3.3 Componentes Técnicos Detalhados

#### A. RoPE — Rotary Position Embedding

Melhor que embeddings posicionais absolutos:
- Generaliza para sequências mais longas que vistas no treinamento
- Encoda posição relativa diretamente na atenção
- Zero parâmetros adicionais (computado on-the-fly)

```
Para posicao m e dimensao d:
[cos(m*theta_d)  -sin(m*theta_d)] [q_2d  ]
[sin(m*theta_d)   cos(m*theta_d)] [q_2d+1]

theta_d = base^(-2d/D)  onde base = 500.000
(base maior = contexto efetivo maior)
```

#### B. Grouped Query Attention (GQA)

```
Standard MHA:  32 Q heads, 32 K heads, 32 V heads
GQA (nossa):   32 Q heads,  8 K heads,  8 V heads

Cada grupo de 4 Q heads compartilha 1 K head e 1 V head.
Resultado: KV cache 4x menor -> atende 4x mais usuarios por GPU.
Qualidade: equivalente ao MHA padrão.
```

#### C. Flash Attention 2

```
Atencao padrao:    O(n^2) memoria — impossivel para n=32768
Flash Attention 2: O(n) memoria efetiva — viavel para n=32768

Algoritmo:
1. Processar atencao em tiles que cabem na SRAM da GPU
2. Acumular resultado sem materializar matriz n^2 completa
3. Backward pass via recomputacao (sem guardar intermediarios)

Resultado: 2-4x mais rapido, mesma saida matematica.
```

#### D. SwiGLU Feed-Forward Network

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU: Swish(x @ W_gate) * (x @ W_up)
    Melhor que ReLU e GELU.
    Usado em: PaLM, LLaMA, Mistral, GPT-4, Qwen.
    """
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))   # Swish activation
        up   = self.up_proj(x)
        return self.down_proj(gate * up)    # elemento a elemento
```

#### E. RMSNorm

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Mais eficiente que LayerNorm: sem bias, sem subtracao da media.
    Normaliza apenas pela RMS da ativacao.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)
```

#### F. Transformer Block Completo

```
Input x
  |
  +-- RMSNorm -> Attention (GQA + RoPE + FlashAttn) -> + x  (residual)
  |
  +-- RMSNorm -> SwiGLU FFN -> + x  (residual)
  |
Output x

Pre-norm architecture: normaliza ANTES de cada sub-layer.
Mais estavel no treinamento que post-norm.
```

### 3.4 Inovações Próprias

#### I. BrazilianContextLayer — Camada de Contexto Cultural

```python
class BrazilianContextLayer(nn.Module):
    """
    Conceito: vetor de contexto cultural aprendido durante o treinamento.
    Representa "brasilidade" — cultura, valores, regionalismos brasileiros.
    Adicionado ao embedding inicial com intensidade aprendida (alpha).
    Permite que o modelo "lembre" que e brasileiro em toda a forward pass.
    Treinado com dados curados especificamente de cultura brasileira.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.context_vector = nn.Parameter(torch.randn(d_model))
        self.alpha = nn.Parameter(torch.tensor(0.1))  # controle de intensidade

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings + self.alpha * self.context_vector
```

#### II. HierarchicalAttention — Para Documentos Longos

```
Problema: atencao quadratica O(n^2) e cara para documentos longos.
Solucao: atencao hierarquica em 3 niveis.

Nivel 1 (sentenca):   atencao full dentro de cada sentenca (ate 512 tokens)
Nivel 2 (paragrafo):  atencao entre representacoes de sentencas
Nivel 3 (documento):  atencao esparsa entre representacoes de paragrafos

Resultado: O(n) efetivo vs. O(n^2) da atencao padrao.
Ativado automaticamente para inputs > 8k tokens.
Casos de uso: contratos, artigos longos, livros.
```

### 3.5 Modelos de Validação (escada antes do 7B)

| Modelo | Params | Propósito | Dados |
|--------|--------|-----------|-------|
| Nano | ~15M | Validar pipeline completo de código | 1B tokens |
| Small | ~125M | Validar decisões arquiteturais | 10B tokens |
| Medium | ~1B | Validar leis de escala | 50B tokens |
| Large | ~3B | Validar configuração antes do 7B | 150B tokens |
| **Full** | **~7B** | **Modelo de produção** | **500B tokens** |

**Regra:** cada modelo deve convergir e atingir perplexidade esperada antes de avançar.

---

## FASE 4 — Pré-Treinamento (Meses 6-10)

### 4.1 Recipe de Treinamento

```python
TRAINING_CONFIG = {
    # Batch
    "micro_batch_size":             4,   # por GPU
    "gradient_accumulation_steps": 32,   # acumular antes do step
    "n_gpus":                      64,   # em producao
    # tokens por step = 4 * 32 * 64 * 4096 = ~33M tokens

    # Otimizador
    "optimizer":       "AdamW",
    "lr":                    3e-4,       # peak learning rate
    "lr_min":                3e-5,       # 10% do peak
    "betas":            (0.9, 0.95),
    "weight_decay":          0.1,
    "grad_clip":             1.0,        # gradient clipping

    # Schedule de LR
    "warmup_steps":         2000,        # warmup linear
    "lr_decay":         "cosine",        # decaimento cossenoidal

    # Precisao
    "dtype":           "bfloat16",       # mais estavel que float16

    # Total
    "total_tokens": 500_000_000_000,     # 500B tokens
}
```

### 4.2 Curriculum de Dados (3 Fases)

```
Fase A — Fundacao (0 a 50B tokens):
  Wikipedia PT-BR:                   30%
  Livros e literatura:               30%
  Legislacao e documentos oficiais:  20%
  Artigos cientificos:               20%
  -> Objetivo: base solida de linguagem de alta qualidade

Fase B — Diversificacao (50B a 350B tokens):
  Corpus completo misturado:          proporcoes naturais
  + Common Crawl filtrado introduzido
  + Codigo introduzido
  -> Objetivo: diversidade e generalizacao

Fase C — Refinamento (350B a 500B tokens):
  Upsampling dados alta qualidade:    x3
  Downsampling Common Crawl:          /2
  + Dados sinteticos de raciocinio
  -> Objetivo: capacidade de raciocinio e refinamento final
```

### 4.3 Treinamento Distribuído — ZeRO-3

```
Problema: 7B params x 4 bytes (float32) = 28GB por GPU.
Uma A100 80GB aguenta, mas sem espaco para gradientes e otimizador.

Solucao ZeRO-3:
- Divide PARAMETROS entre as GPUs (cada GPU tem 1/N dos params)
- Divide GRADIENTES entre as GPUs
- Divide ESTADO DO OTIMIZADOR (Adam) entre as GPUs
- Resultado: memoria por GPU proporcional a 1/N do total

Com 8 GPUs: ~3.5GB de params por GPU — cabe ate em uma GPU de 16GB.
Comunicacao: NCCL (NVLink dentro do no, InfiniBand entre nos).
```

**Config DeepSpeed ZeRO-3:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000
  },
  "gradient_clipping": 1.0,
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": 4
}
```

### 4.4 Monitoramento em Tempo Real (Weights & Biases)

```
Metricas obrigatorias:
  - loss (cross-entropy) e perplexidade = exp(loss)
  - gradient norm — detecta instabilidades e explosoes
  - learning rate atual
  - tokens/segundo (throughput)
  - MFU (Model FLOP Utilization) — eficiencia real da GPU

Hardware:
  - GPU utilization (%)
  - GPU memory used/total
  - Temperatura das GPUs
  - Throughput de rede entre nos

Alertas automaticos:
  - Loss spike: aumento > 10% em 100 steps -> rollback automatico
  - NaN em loss ou gradientes -> parar e investigar
  - GPU utilization < 85% -> gargalo de dados ou comunicacao
```

### 4.5 Estratégia de Checkpoints

```
Frequencia:      salvar a cada 1.000 steps
Retencao:        manter ultimos 10 checkpoints
Arquivo:         checkpoint completo a cada 10.000 steps (permanente)

Rollback automatico:
  Se loss > media_100_steps_anteriores * 1.15:
    -> Carregar checkpoint anterior
    -> Reduzir LR em 10%
    -> Retomar treinamento
```

### 4.6 Custo de Computação

```
Formula de FLOPs para treinar transformer:
  FLOPs ≈ 6 x N x D
  N = parametros = 7B
  D = tokens = 500B
  FLOPs = 6 x 7e9 x 5e11 = 2.1 x 10^22

Tempo com diferentes configs (H100 SXM = 3.9 PFLOPs/s em BF16, 80% MFU):
  1x  H100:  ~778 dias   ❌ inviavel
  8x  H100:  ~97 dias
  32x H100:  ~24 dias
  64x H100:  ~12 dias    ✅ ideal
  64x A100:  ~24 dias    ✅ alternativa economica
```

---

## FASE 5 — Fine-Tuning e Alinhamento (Meses 10-14)

### 5.1 Por que Fine-Tuning é necessário

```
Apos pre-treinamento:
  O modelo sabe "falar" mas nao sabe "ajudar".
  Responde de forma estatisticamente provavel, nao de forma util.

Apos SFT + DPO:
  O modelo segue instrucoes, e util, honesto e seguro.
  Responde como um assistente, nao como um completador de texto.
```

### 5.2 Supervised Fine-Tuning (SFT)

**Formato ChatML:**
```
<|sys|>
Você é uma IA brasileira criada para ser útil, honesta e segura.
Você foi desenvolvida no Brasil e orgulhosamente fala português do Brasil.
<|user|>
Como funciona o sistema tributário brasileiro?
<|assistant|>
O sistema tributário brasileiro é um dos mais complexos do mundo...
[resposta completa aqui]
<|eos|>
```

**Dataset de SFT — 500k exemplos de alta qualidade:**

| Categoria | Volume |
|-----------|--------|
| Perguntas e respostas gerais PT-BR | 150k |
| Instrução e execução de tarefas | 100k |
| Raciocínio passo-a-passo (Chain of Thought) | 80k |
| Código com explicação em PT-BR | 60k |
| Conhecimento jurídico brasileiro | 30k |
| Matemática explicada em PT-BR | 20k |
| Saúde e medicina no contexto do SUS | 15k |
| Conversação multiturno em PT-BR | 100k |
| Redação e criação de texto | 40k |
| Cultura e história do Brasil | 25k |

**Hiperparâmetros SFT:**
```python
SFT_CONFIG = {
    "lr":                  2e-5,    # 10x menor que pre-treino
    "epochs":                 3,
    "batch_size":           128,
    "max_seq_len":         4096,
    "warmup_ratio":        0.03,
    "lr_decay":        "cosine",
    "weight_decay":         0.0,
    "response_only_loss":  True,    # loss APENAS nos tokens da resposta
}
```

### 5.3 Direct Preference Optimization (DPO)

**Por que DPO ao invés de RLHF clássico:**
```
RLHF clássico requer:
  1. Treinar reward model separado
  2. Treinar policy com PPO (instável)
  3. Manter reference model
  -> Complexo, caro, instável

DPO:
  - Reformulação matemática equivalente
  - Treina diretamente com pares (chosen, rejected)
  - Um único modelo, mais estável, mais simples
  - Resultados comparáveis ou melhores
```

**Formato do dataset de preferências:**
```json
{
  "prompt": "Explique habeas corpus em termos simples",
  "chosen": "Habeas corpus é um remédio constitucional previsto no art. 5º...[explicação clara e precisa]",
  "rejected": "Habeas corpus é uma coisa jurídica...[explicação vaga e imprecisa]"
}
```

**Quem anota (100k pares):**
- Advogados para questões jurídicas
- Médicos para questões de saúde
- Professores para questões de educação
- Jornalistas para linguagem e fatos
- Engenheiros para questões técnicas
- Cidadãos para linguagem coloquial e adequação cultural

### 5.4 Constituição da IA Brasileira

Conjunto de princípios inspirado no Constitutional AI (Anthropic):

```markdown
## Principios Invioaveis:
1. Nao produzir conteudo que incite violencia, discriminacao ou odio
2. Nao ajudar a infringir leis brasileiras vigentes
3. Nao produzir desinformacao intencional
4. Respeitar a privacidade do usuario (LGPD)
5. Ser honesta sobre ser uma IA quando perguntada

## Principios de Qualidade:
6. Responder sempre em portugues do Brasil (a menos que solicitado)
7. Usar linguagem acessivel, sem pedantismo desnecessario
8. Reconhecer incerteza e limitacoes quando existirem
9. Citar fontes e referências quando possível
10. Adaptar formalidade ao contexto da conversa

## Principios Culturais:
11. Respeitar e celebrar a diversidade brasileira
12. Conhecer corretamente a cultura, historia e legislacao do Brasil
13. Nao estereotipar regionalismos negativamente
14. Ser sensivel as desigualdades sociais brasileiras
15. Usar referencias culturais brasileiras de forma autentica
```

### 5.5 Modelos Filhos Especializados (pós v1.0)

| Modelo | Especialização | Dataset Principal |
|--------|---------------|-------------------|
| IA-BR-Juridico | Direito Brasileiro | OAB, tribunais, legislação |
| IA-BR-Medico | Medicina e Saúde | Protocolos SUS, literatura médica |
| IA-BR-Codigo | Programação | GitHub PT-BR, Stack Overflow PT |
| IA-BR-Edu | Educação | ENEM, vestibulares, didática |
| IA-BR-Negocios | Finanças e Gestão | Normas CVM, BACEN |

---

## FASE 6 — Avaliação e Benchmarks (Meses 12-15)

### 6.1 Avaliação Contínua desde o Pré-Treino

```
Durante pre-treino (a cada 10.000 steps):
  -> Perplexidade em validation set PT-BR
  -> Few-shot em tarefas simples
  -> Comparacao com checkpoint anterior

Apos SFT:
  -> Benchmarks automaticos completos
  -> Avaliacao humana preliminar

Apos DPO:
  -> Comparacao humana com Gemini Flash (nosso alvo)
  -> Suite completa de benchmarks
```

### 6.2 Benchmarks Automáticos

```python
BENCHMARKS = {
    # Benchmarks internacionais
    "mmlu":          {"few_shot": 5,  "target": 0.75},
    "arc_challenge": {"few_shot": 25, "target": 0.75},
    "gsm8k":         {"few_shot": 8,  "target": 0.75},
    "humaneval":     {"few_shot": 0,  "target": 0.70},
    "truthfulqa":    {"few_shot": 0,  "target": 0.70},

    # Benchmarks brasileiros (nossa vantagem)
    "enem":          {"years": "2009-2024", "target": 0.90},
    "oab":           {"years": "2010-2024", "target": 0.85},
    "assin2":        {"task": "entailment", "target": 0.88},
    "portuguese_squad": {"target": 0.90},

    # Benchmark proprio (cultura, girias, regionalismos BR)
    "benchmark_br":  {"target": 0.85},
}
```

### 6.3 Avaliação Humana Estruturada

```
Setup:
  - 50 avaliadores brasileiros de todas as regioes
  - Distribuicao por profissao e faixa etaria
  - Avaliacao cega (nao sabem qual modelo e qual)

Metodologia:
  Para cada prompt:
    - Nossa IA responde
    - Gemini Flash responde (via API)
    - GPT-4o responde (referencia adicional)
  
  Avaliador ve "Modelo A" e "Modelo B" sem identificacao.

Criterios (escala 1-5 para cada):
  1. Naturalidade do portugues
  2. Precisao das informacoes
  3. Utilidade pratica
  4. Adequacao cultural ao Brasil
  5. Clareza e organizacao

Decisao final: "Prefiro A" / "Prefiro B" / "Indiferente"
Meta: > 60% de preferencia sobre Gemini Flash em PT-BR
```

### 6.4 Red Teaming (Segurança)

Equipe dedicada a tentar "quebrar" o modelo:
- Jailbreak attempts (bypassar restricoes de seguranca)
- Extracao de informacoes privadas
- Inducao a geracao de desinformacao
- Inducao a vieses e preconceitos
- Prompt injection attacks

Cada vulnerabilidade encontrada -> patch via DPO ou SFT adicional.

---

## FASE 7 — Produto e Deploy (Meses 14-18)

### 7.1 Otimizações de Inferência

```python
INFERENCE_CONFIG = {
    # Quantizacao: reduz 7B BF16 (14GB) para 4-bit (~4GB)
    "quantization":    "AWQ",    # Activation-aware Weight Quantization
    "bits":               4,
    "quality_loss":    "<1%",    # perda negligivel em benchmarks

    # Serving com vLLM (PagedAttention)
    "engine":          "vLLM",   # 3x mais throughput que naive serving
    "max_batch_size":    256,    # usuarios simultaneos por GPU

    # Latencia alvo
    "ttft":     "< 100ms",       # Time To First Token
    "tpot":      "< 30ms",       # Time Per Output Token (medio)

    # Hardware minimo por instancia producao
    "gpu_min":  "RTX 4090",      # 24GB VRAM (modelo 4-bit)
    "gpu_ideal": "A100 40GB",    # margin de seguranca
}
```

### 7.2 API (compatível com formato OpenAI)

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "ia-br-7b",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true
}
```

Endpoints:
```
POST /v1/chat/completions   — chat principal
POST /v1/completions        — completar texto
POST /v1/embeddings         — vetores de embedding
GET  /v1/models             — listar modelos
GET  /v1/health             — status da API
```

### 7.3 SDKs

```python
# Python SDK
from ia_br import IABR

client = IABR(api_key="ia-br-sua-chave-aqui")

response = client.chat.complete(
    model="ia-br-7b",
    messages=[{"role": "user", "content": "Explique o Bolsa Família"}]
)

print(response.message.content)
```

```javascript
// JavaScript/TypeScript SDK
import { IABR } from 'ia-br';

const client = new IABR({ apiKey: 'ia-br-sua-chave-aqui' });

const response = await client.chat.complete({
  model: 'ia-br-7b',
  messages: [{ role: 'user', content: 'Explique o Bolsa Família' }]
});

console.log(response.message.content);
```

### 7.4 Interface Web

```
ia-br.com.br/
├── /chat          — interface de chat principal (estilo Gemini/Claude)
├── /playground    — controle de parametros (temperatura, top-p, max tokens)
├── /api           — documentacao interativa da API
├── /dashboard     — painel do desenvolvedor (uso, chaves, billing)
└── /blog          — atualizacoes, releases e pesquisa
```

---

## FASE 8 — Escala e Evolução (Meses 18+)

### Roadmap de Versões

| Versão | Params | Novidades | Prazo |
|--------|--------|-----------|-------|
| v1.0 | 7B | Modelo base BR, API pública, chat | Mês 18 |
| v1.5 | 7B | CoT aprimorado, alinhamento melhorado | Mês 21 |
| v2.0 | 13B | Maior capacidade, melhor raciocínio | Mês 24 |
| v2.5 | 13B | Multimodal (texto + imagem + PDF) | Mês 30 |
| v3.0 | 30B | Supera GPT-4o em PT-BR | Mês 36 |
| v3.5 | 70B | Estado da arte mundial em PT-BR | Mês 42 |

### Multimodalidade (v2.5)

```
Arquitetura multimodal:

Imagem -> [Vision Encoder (ViT)] -> vetores visuais
                                         |
                                  [Projetor MLP]  <- converte visual -> linguagem
                                         |
Texto -> [Tokenizer BR]          -> vetores de texto
                                         |
                               [LLM 13B unificado]
                                         |
                                     Resposta
```

### Open Source Estratégico

| Release | O que liberar | Quando |
|---------|--------------|--------|
| Tokenizer | Código + vocabulário completo | Imediato após treinamento |
| Modelo 1B | Pesos completos + código | Mês 12 |
| Modelo 3B | Pesos completos + código | Mês 18 |
| Subset de dados | Parte limpa do corpus BR | Mês 18 |
| Modelo 7B distilado | Versão comprimida | Mês 24 |

---

## Infraestrutura de Computação

### Desenvolvimento e Validação (Fases 0-3)

```
Minimo para desenvolvimento local:
  - 1x RTX 4090 (24GB VRAM) — suficiente ate ~1B parametros
  - 64GB RAM
  - 4TB NVMe SSD
  - Custo hardware: ~$2.500

Alternativa 100% cloud (evita CAPEX inicial):
  - RunPod / Vast.ai: ~$2/hora por RTX 4090
  - Lambda Labs: ~$1.50/hora por A100 40GB
```

### Treinamento do Modelo 7B (Fase 4)

```
Opcao A — Cloud rapido (recomendado para inicio):
  Provedor:    RunPod ou Lambda Labs
  Config:      8x H100 SXM 80GB
  Custo:       ~$25/hora
  Duracao 7B:  ~45 dias (estimativa com 8 GPUs)
  Total:       ~$27.000

Opcao B — Cloud escalado (mais rapido):
  Provedor:    Lambda Labs / SkyPilot
  Config:      32x A100 80GB
  Custo:       ~$50/hora
  Duracao 7B:  ~15 dias
  Total:       ~$18.000

Opcao C — Infraestrutura Nacional (mais barato a longo prazo):
  LNCC (Santos Dumont):  edital de acesso para pesquisa brasileira
  Custo:                 subsidiado para projetos nacionais
  Processo:              inscricao via portal LNCC

Opcao D — Parceria Estrategica:
  Embrapa, INPE, RNP:    acesso a infraestrutura existente
  Contrapartida:         modelo open-source para pesquisa
```

### Produção (Fase 7+)

```
Por instancia de inferencia:
  - 1x A100 40GB (modelo 4-bit quantizado com vLLM)
  - Custo cloud: ~$1.500/mes
  - Capacidade: ~200 usuarios simultaneos

Para escalar a 10.000 usuarios ativos:
  - ~50 instancias = $75.000/mes
  - Compensado por receita de assinaturas (meta: R$200k/mes)
```

---

## Stack Tecnológico

### Core do Modelo
```
Python 3.11+       linguagem principal
PyTorch 2.2+       deep learning framework
Triton 2.x         kernels GPU customizados (Flash Attention propria)
CUDA 12.1+         computacao GPU
cuDNN 8.9+         primitivas de deep learning otimizadas
```

### Treinamento
```
DeepSpeed 0.14+    ZeRO optimizer e treinamento distribuido
NCCL 2.x           comunicacao GPU-GPU (all-reduce)
Ray 2.x            orquestracao de jobs distribuidos
Weights & Biases   rastreamento de experimentos e metricas
```

### Dados
```
Apache Spark 3.x   processamento de grandes volumes de dados
Trafilatura        extracao de texto de HTML (web scraping limpo)
FastText           classificacao de idioma (lid.176.bin)
Pandas, NumPy      manipulacao e analise de dados
MinHash LSH        deduplicacao em escala (Jaccard similarity)
```

### Inferencia e Serving
```
vLLM               serving de alta performance (PagedAttention)
FastAPI            API REST
Uvicorn            ASGI server
Redis              cache de KV e sessoes
PostgreSQL         banco de dados principal
```

### Infraestrutura
```
Kubernetes         orquestracao de containers
Docker             containerizacao
Terraform          infraestrutura como codigo
Prometheus+Grafana monitoramento e alertas
MinIO              object storage (checkpoints, datasets)
```

### Frontend
```
Next.js 14+        interface web (App Router)
TypeScript         tipagem estatica
Tailwind CSS       estilizacao
```

---

## Equipe e Papéis

### Equipe Mínima

| Papel | Qtd | Responsabilidades |
|-------|-----|-------------------|
| Research Lead (ML) | 1 | Arquitetura, papers, decisoes tecnicas |
| ML Engineer (arquitetura) | 2 | Implementar modelo, kernels Triton, otimizacoes |
| Data Engineer | 2 | Pipeline de dados, coleta, limpeza, tokenizer |
| Infrastructure Engineer | 1 | GPUs, kubernetes, MLOps, custos |
| Alignment Researcher | 1 | SFT, DPO, Constituicao, seguranca |
| Product Engineer | 1 | API, interface web, SDKs |
| **TOTAL** | **8** | |

### Papéis de Suporte

| Papel | Quantidade | Forma |
|-------|-----------|-------|
| Anotadores de Dados (BR) | 20-50 | Contratados por tarefa |
| Red Team (segurança) | 3-5 | Consultores |
| Pesquisadores Academicos | parceria | UNICAMP, USP, UFMG |
| Advogados | 1-2 | Compliance LGPD e licencas |

### Parcerias Estratégicas Recomendadas

- **UNICAMP / USP / UFMG / UFSC** — pesquisa, GPUs, dados academicos
- **RNP (Rede Nacional de Pesquisa)** — infraestrutura de rede
- **LNCC** — acesso ao Santos Dumont (supercomputador nacional)
- **FAPESP / CNPq** — financiamento de pesquisa
- **BNDES** — financiamento de inovacao
- **EMBRAPII** — projetos empresa-universidade

---

## Financiamento e Custos

### Custo Total Estimado (24 meses)

| Item | Custo Estimado |
|------|---------------|
| Computacao — treinamento 7B | R$ 150.000–300.000 |
| Computacao — desenvolvimento e experimentos | R$ 50.000 |
| Computacao — inferencia em producao (12 meses) | R$ 200.000 |
| Equipe — 8 pessoas x 24 meses | R$ 2.000.000–4.000.000 |
| Dados — coleta, storage, anotacao humana | R$ 100.000 |
| Licencas, ferramentas, legal, misc | R$ 50.000 |
| **TOTAL** | **~R$ 2.5M – 4.7M** |

### Fontes de Financiamento Públicas (Brasil)

| Fonte | O que oferece | Como acessar |
|-------|--------------|--------------|
| FAPESP (SP) | Ate R$2M para inovacao | Edital FAPESP Inovacao |
| CNPq | Bolsas de pesquisa + infra | Portal CNPq |
| BNDES Garagem | Startups early-stage | Portal BNDES |
| Finep | Financiamento P&D | Editais Finep |
| EMBRAPII | Projetos empresa-universidade | Unidades EMBRAPII |
| BNDES Mais Inovacao | Credito para inovacao | Linha especial |

### Fontes Privadas

- Seed round com VCs brasileiros (Kaszek, Monashees, Astella, Spectra)
- Patrocinio de grandes empresas BR interessadas em IA nacional
- Pre-venda de creditos de API para early adopters

### Modelo de Receita

| Produto | Preco | Meta Mes 18 | Receita |
|---------|-------|-------------|---------|
| API (por token) | R$ 0,05/1k tokens | 500M tokens/mes | R$ 25.000 |
| Pro Pessoal | R$ 49/mes | 1.000 usuarios | R$ 49.000 |
| Business | R$ 499/mes | 200 empresas | R$ 99.800 |
| Enterprise | Negociado (~R$5k) | 5 clientes | R$ 25.000 |
| **TOTAL** | | | **~R$ 200.000/mes** |

---

## Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|:------------:|:-------:|-----------|
| Instabilidade no treinamento | Alta | Alto | Checkpoints frequentes; rollback automatico; warmup longo |
| Dados de baixa qualidade | Alta | Alto | Pipeline rigoroso; validacao humana periodica |
| Custo de GPU acima do orcado | Alta | Alto | Buffer de 30%; buscar LNCC e FAPESP |
| Equipe insuficiente | Alta | Alto | Parcerias com universidades; remote-first; equity |
| Atraso no cronograma | Alta | Medio | Buffer 20% por fase; MVP reduzido se necessario |
| Concorrencia (Maritaca, Sabia 3) | Media | Medio | Diferenciacao por produto e dados proprietarios |
| Mudanca regulatoria de IA | Baixa | Alto | Acompanhar PL da IA; privacidade by design |
| Roubo de IP | Baixa | Medio | Pesos proprietarios; API como produto principal |

---

## Cronograma Geral

```
2026
╔══════════════════════════════════════════════════════════╗
║ Jul  Ago  Set  Out  Nov  Dez                            ║
║ ████████░░░░░░░░░░░░░░░░░░░░░  FASE 0: Fundacao (2m)   ║
║ ████████████████████████████░  FASE 1: Dados (paralelo) ║
║           ████████████░░░░░░░  FASE 2: Tokenizer        ║
║           ████████████████████ FASE 3: Arquitetura      ║
╚══════════════════════════════════════════════════════════╝

2027
╔══════════════════════════════════════════════════════════════════════════════╗
║ Jan  Fev  Mar  Abr  Mai  Jun  Jul  Ago  Set  Out  Nov  Dez                ║
║ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Modelos 1B e 3B (validacao)        ║
║      ████████████████████████████████ FASE 4: Pre-treino 7B               ║
║                              ████████ FASE 5: Fine-tuning + Alinhamento   ║
╚══════════════════════════════════════════════════════════════════════════════╝

2028
╔══════════════════════════════════════════╗
║ Jan  Fev  Mar  Abr  Mai  Jun            ║
║ ████████░░░░░░░░░░  FASE 5: Conclusao  ║
║      ██████████████ FASE 6: Benchmarks ║
║               █████ FASE 7: Produto    ║
║                ████ LANCAMENTO v1.0 🚀 ║
╚══════════════════════════════════════════╝
```

**Marcos Críticos:**

| Marco | Mês | O que Valida |
|-------|-----|--------------|
| Tokenizer BR pronto | 3 | Pipeline de dados funcional |
| Modelo 125M convergindo | 5 | Arquitetura correta |
| Modelo 1B com boa perplexidade | 8 | Leis de escala confirmadas |
| Pre-treino 7B iniciado | 10 | Infraestrutura de GPU pronta |
| Pre-treino 7B concluido | 14 | **MARCO MAIS CRITICO** |
| SFT + DPO concluidos | 16 | Modelo util e alinhado |
| Beta privado | 18 | Produto funcional |
| Lancamento publico v1.0 | 20 | IA brasileira no mundo |

---

## Definição de Sucesso

### MVP — Mínimo Viável
- [ ] Modelo 7B funcionando com dados BR proprios
- [ ] API publica com latencia < 200ms
- [ ] Supera Gemini Flash em pelo menos 3 benchmarks PT-BR
- [ ] 100 usuarios ativos no beta

### Sucesso Esperado (v1.0 — Mês 20)
- [ ] Preferencia humana >= 60% vs. Gemini Flash em tarefas PT-BR
- [ ] ENEM historico > 90%
- [ ] MMLU > 72%
- [ ] ASSIN2 > 85%
- [ ] 10.000 usuarios ativos
- [ ] R$ 100.000/mes de receita
- [ ] Modelos menores (1B, 3B) open-source

### Sucesso Transformador (v2.0+ — Mes 24+)
- [ ] Melhor modelo de IA em portugues do mundo
- [ ] Supera GPT-4o em benchmarks PT-BR
- [ ] Adotado pelo governo brasileiro em sistemas publicos
- [ ] Referencia de IA soberana para a America Latina
- [ ] 1 milhao+ de usuarios
- [ ] Empresa sustentavel e autofinanciada
- [ ] Inspiracao para outros paises do Sul Global

---

## Estrutura de Pastas do Projeto

```
ia/
├── PLAN.md                             <- este arquivo
├── README.md
├── .gitignore
├── pyproject.toml                      <- configuracao Python
│
├── src/
│   ├── tokenizer/
│   │   ├── bpe_trainer.py              <- treinar BPE do zero
│   │   ├── tokenizer.py                <- encode/decode eficiente
│   │   ├── vocab_builder.py            <- construir vocabulario
│   │   └── special_tokens.py           <- tokens especiais
│   │
│   ├── model/
│   │   ├── config.py                   <- ModelConfig (dataclass)
│   │   ├── embedding.py                <- TokenEmbedding
│   │   ├── attention.py                <- GQA + RoPE + Flash Attention
│   │   ├── feedforward.py              <- SwiGLU FFN
│   │   ├── normalization.py            <- RMSNorm
│   │   ├── transformer_block.py        <- bloco Transformer completo
│   │   ├── model.py                    <- modelo LLM completo
│   │   └── layers/
│   │       ├── cultural_context.py     <- BrazilianContextLayer
│   │       └── hierarchical_attn.py    <- HierarchicalAttention
│   │
│   ├── kernels/                        <- kernels GPU customizados (Triton)
│   │   ├── flash_attention.py          <- Flash Attention em Triton
│   │   ├── fused_rmsnorm.py            <- RMSNorm fundido
│   │   └── rope.py                     <- RoPE otimizado
│   │
│   ├── training/
│   │   ├── dataset.py                  <- DataLoader eficiente (memmap)
│   │   ├── optimizer.py                <- AdamW + scheduler
│   │   ├── trainer.py                  <- training loop principal
│   │   ├── checkpoint.py               <- salvar/carregar/rollback
│   │   ├── distributed.py              <- config DeepSpeed/NCCL
│   │   └── monitoring.py               <- WandB + alertas
│   │
│   ├── data/
│   │   ├── collectors/
│   │   │   ├── common_crawl.py
│   │   │   ├── wikipedia.py
│   │   │   ├── legislacao.py
│   │   │   ├── scielo.py
│   │   │   └── github.py
│   │   ├── pipeline/
│   │   │   ├── language_filter.py      <- fastText PT-BR
│   │   │   ├── deduplication.py        <- MinHash LSH
│   │   │   ├── quality_scorer.py       <- score de qualidade ML
│   │   │   ├── pii_remover.py          <- remocao de dados pessoais
│   │   │   └── normalizer.py           <- normalizacao UTF-8
│   │   └── synthetic/
│   │       ├── instruction_gen.py      <- geracao de instrucoes
│   │       └── cot_gen.py              <- chain-of-thought
│   │
│   ├── alignment/
│   │   ├── sft_trainer.py              <- Supervised Fine-Tuning
│   │   ├── dpo_trainer.py              <- Direct Preference Optimization
│   │   ├── constitution.py             <- Constituicao da IA Brasileira
│   │   └── reward_model.py             <- modelo de recompensa (auxiliar)
│   │
│   ├── evaluation/
│   │   ├── benchmarks/
│   │   │   ├── mmlu.py
│   │   │   ├── humaneval.py
│   │   │   ├── gsm8k.py
│   │   │   ├── enem.py
│   │   │   ├── oab.py
│   │   │   ├── assin2.py
│   │   │   └── portuguese_squad.py
│   │   ├── evaluator.py                <- runner central de benchmarks
│   │   └── human_eval/                 <- framework avaliacao humana
│   │
│   └── serving/
│       ├── api.py                      <- FastAPI (OpenAI-compatible)
│       ├── engine.py                   <- vLLM wrapper
│       └── quantization.py             <- AWQ quantizacao
│
├── configs/
│   ├── model_7b.yaml                   <- hiperparametros 7B
│   ├── training.yaml                   <- config de treinamento
│   ├── deepspeed_zero3.json            <- config DeepSpeed
│   └── data_pipeline.yaml              <- config pipeline de dados
│
├── scripts/
│   ├── train.py                        <- script principal de treinamento
│   ├── finetune.py                     <- SFT e DPO
│   ├── evaluate.py                     <- rodar benchmarks
│   ├── collect_data.sh                 <- coleta de dados
│   └── deploy.sh                       <- deploy para producao
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model_forward.py
│   ├── test_attention.py
│   ├── test_data_pipeline.py
│   └── test_distributed.py
│
├── notebooks/
│   ├── 01_tokenizer_analysis.ipynb
│   ├── 02_architecture_exploration.ipynb
│   ├── 03_data_analysis.ipynb
│   ├── 04_scaling_laws.ipynb
│   └── 05_benchmark_results.ipynb
│
├── data/                               <- gitignored (dados reais)
│   ├── raw/
│   ├── processed/
│   └── tokenized/
│
├── checkpoints/                        <- gitignored (pesos do modelo)
│
└── docs/
    ├── architecture.md                 <- documentacao tecnica
    ├── data.md                         <- documentacao dos dados
    ├── training.md                     <- guia de treinamento
    ├── api.md                          <- documentacao da API
    └── benchmarks.md                   <- resultados e comparacoes
```

---

## Papers Obrigatórios

Leia nesta ordem. Nao pule. Cada paper constroi sobre o anterior.

### Bloco 1 — Arquitetura Base

| # | Paper | Ano | Link | O que ensina |
|---|-------|-----|------|--------------|
| 1 | **Attention Is All You Need** | 2017 | https://arxiv.org/abs/1706.03762 | O Transformer original. Leia cada equacao. |
| 2 | **Language Models are Unsupervised Multitask Learners** (GPT-2) | 2019 | https://openai.com/research/gpt-2 | Como pre-treinamento funciona |
| 3 | **Language Models are Few-Shot Learners** (GPT-3) | 2020 | https://arxiv.org/abs/2005.14165 | Escala e emergencia de capacidades |
| 4 | **LLaMA: Open and Efficient Foundation LMs** | 2023 | https://arxiv.org/abs/2302.13971 | Arquitetura moderna e eficiente |
| 5 | **Mistral 7B** | 2023 | https://arxiv.org/abs/2310.06825 | GQA, Sliding Window Attention |
| 6 | **Qwen2.5 Technical Report** | 2024 | https://arxiv.org/abs/2412.15115 | Melhor pratica atual |

### Bloco 2 — Escala e Otimização

| # | Paper | Ano | Link | O que ensina |
|---|-------|-----|------|--------------|
| 7 | **Chinchilla: Training Compute-Optimal LLMs** | 2022 | https://arxiv.org/abs/2203.15556 | ESSENCIAL: dados vs. params |
| 8 | **FlashAttention** | 2022 | https://arxiv.org/abs/2205.14135 | Atencao eficiente em memoria |
| 9 | **FlashAttention-2** | 2023 | https://arxiv.org/abs/2307.08691 | Versao otimizada |
| 10 | **ZeRO: Memory Optimizations** | 2020 | https://arxiv.org/abs/1910.02054 | Treinamento distribuido eficiente |

### Bloco 3 — Alinhamento

| # | Paper | Ano | Link | O que ensina |
|---|-------|-----|------|--------------|
| 11 | **InstructGPT** | 2022 | https://arxiv.org/abs/2203.02155 | RLHF — como fazer SFT + RM |
| 12 | **Direct Preference Optimization (DPO)** | 2023 | https://arxiv.org/abs/2305.18290 | Nossa abordagem de alinhamento |
| 13 | **Constitutional AI** | 2022 | https://arxiv.org/abs/2212.08073 | Base da Constituicao BR |
| 14 | **LoRA** | 2022 | https://arxiv.org/abs/2106.09685 | Fine-tuning eficiente |

### Bloco 4 — Dados

| # | Paper | Ano | Link | O que ensina |
|---|-------|-----|------|--------------|
| 15 | **Deduplicating Training Data** | 2022 | https://arxiv.org/abs/2107.06499 | Por que deduplicar e critico |
| 16 | **RefinedWeb for Falcon LLM** | 2023 | https://arxiv.org/abs/2306.01116 | Como filtrar Common Crawl |
| 17 | **Neural MT with Subword Units (BPE)** | 2016 | https://arxiv.org/abs/1508.07909 | Base do nosso tokenizer |

### Bloco 5 — Leis de Escala

| # | Paper | Ano | Link | O que ensina |
|---|-------|-----|------|--------------|
| 18 | **Scaling Laws for Neural Language Models** | 2020 | https://arxiv.org/abs/2001.08361 | Prever performance antes de treinar |
| 19 | **Emergent Abilities of Large LLMs** | 2022 | https://arxiv.org/abs/2206.07682 | Por que modelos maiores surgem capacidades |

---

## Próximos Passos Imediatos

```
SEMANA 1-2:
[ ] Ler papers 1 e 7 (Attention Is All You Need + Chinchilla)
[ ] Configurar ambiente: Python 3.11 + PyTorch + CUDA
[ ] Implementar bigramas simples (nivel 1 do exercicio obrigatorio)

SEMANA 3-4:
[ ] Ler papers 4 e 5 (LLaMA + Mistral)
[ ] Implementar self-attention do zero (nivel 3)
[ ] Implementar multi-head attention (nivel 4)

MES 2:
[ ] Implementar Transformer block completo (nivel 5)
[ ] Implementar mini-GPT e treinar em corpus pequeno (nivel 6)
[ ] Iniciar coleta da Wikipedia PT-BR (primeira fonte de dados)

MES 3:
[ ] Pipeline de limpeza de dados funcionando
[ ] Treinar primeiro tokenizer BPE no Wikipedia PT-BR
[ ] Treinar modelo Nano (15M params) para validar pipeline completo
[ ] Modelo Small (125M params) iniciado

MARCO DO MES 3: Tokenizer BR treinado e pipeline funcionando.
Este e o primeiro marco concreto e verificavel do projeto.
```

---

*Documento criado: Julho 2026 | Versao: 1.0*
*Projeto: IA Brasileira — do zero para o mundo*

---

> "O Brasil tem talento, tem dados, tem criatividade.
>  O que faltava era coragem de comecar. Hoje comeca."
