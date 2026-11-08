import os
import json
from typing import List, Dict

# Currículo Oficial da Tríade Lloavece: Matemática, Código em Inglês e Raciocínio em PT-BR
TRIAD_CURRICULUM: List[Dict[str, str]] = [
    # 1. MATEMÁTICA PURA & CÁLCULO
    {
        "prompt": "Explique passo a passo como calcular a derivada de f(x) = x^3 * e^(2x) utilizando a Regra do Produto e da Cadeia, e forneça uma implementação em Python para verificação numérica.",
        "response": r'''### Resolução Matemática Passo a Passo

Queremos calcular a derivada da função:
$$f(x) = u(x) \cdot v(x)$$
onde:
- $u(x) = x^3$
- $v(x) = e^{2x}$

#### 1. Derivadas Individuais:
- Pela regra do tombo: $u'(x) = 3x^2$
- Pela regra da cadeia: $v'(x) = e^{2x} \cdot \frac{d}{dx}(2x) = 2e^{2x}$

#### 2. Aplicação da Regra do Produto:
$$f'(x) = u'(x)v(x) + u(x)v'(x)$$
$$f'(x) = (3x^2)(e^{2x}) + (x^3)(2e^{2x})$$

#### 3. Fatoração e Simplificação:
Colocando o termo comum $x^2 e^{2x}$ em evidência:
$$f'(x) = x^2 e^{2x} (3 + 2x)$$

---

### Implementação Numérica e Simbólica em Python

```python
import math

def f(x: float) -> float:
    """Função analítica f(x) = x^3 * e^(2x)"""
    return (x ** 3) * math.exp(2 * x)

def f_derivada_analitica(x: float) -> float:
    """Derivada exata f'(x) = x^2 * e^(2x) * (3 + 2x)"""
    return (x ** 2) * math.exp(2 * x) * (3 + 2 * x)

def f_derivada_numerica(x: float, h: float = 1e-7) -> float:
    """Aproximação por Diferenças Finitas Centradas O(h^2)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Teste para x = 2.0
x_val = 2.0
exata = f_derivada_analitica(x_val)
numerica = f_derivada_numerica(x_val)
print(f"Analítica: {exata:.6f} | Numérica: {numerica:.6f}")
assert abs(exata - numerica) < 1e-4
```'''
    },

    # 2. ÁLGEBRA LINEAR & ATENÇÃO DO TRANSFORMER
    {
        "prompt": "Como funciona matematicamente o mecanismo de Atenção Scaled Dot-Product do Transformer? Explique as matrizes Q, K, V e escreva o algoritmo em Python com PyTorch.",
        "response": r'''### Fundamentação Matemática da Atenção

O mecanismo de atenção mapeia uma consulta (**Query**) e um conjunto de pares chave-valor (**Key-Value**) para uma saída vetorial.

Dadas as matrizes de entrada:
- $Q \in \mathbb{R}^{B \times T \times d_k}$ (Queries)
- $K \in \mathbb{R}^{B \times T \times d_k}$ (Keys)
- $V \in \mathbb{R}^{B \times T \times d_v}$ (Values)

A equação canônica de Vaswani et al. (2017) é:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

#### Por que dividir por $\sqrt{d_k}$?
Para valores grandes de $d_k$, o produto escalar $Q K^T$ cresce em magnitude. Isso empurra a função $\text{Softmax}$ para regiões com gradientes infinitesimais (vanishing gradients). O fator de escala $\frac{1}{\sqrt{d_k}}$ mantém a variância unitária ($\text{Var} = 1$).

---

### Implementação Vetorizada em Python (PyTorch)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True
) -> torch.Tensor:
    """
    Computa a Atenção Scaled Dot-Product de forma eficiente.
    Dimensões esperadas: (Batch, Heads, Seq_Len, Head_Dim)
    """
    d_k = q.size(-1)
    scale = 1.0 / (d_k ** 0.5)
    
    # 1. Produto Escalar das Consultas com as Chaves: (B, H, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # 2. Máscara Causal para Auto-regressão (impede ler o futuro)
    if is_causal:
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1)
        scores = scores + causal_mask
        
    # 3. Probabilidades de Atenção normalizadas via Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. Multiplicação ponderada pelos Valores: (B, H, T, d_v)
    output = torch.matmul(attn_weights, v)
    return output
```'''
    },

    # 3. ALGORITMOS & ESTRUTURAS DE DADOS (TEORIA DOS GRAFOS)
    {
        "prompt": "Implemente o Algoritmo de Dijkstra para encontrar o caminho mais curto em um grafo ponderado com pesos não-negativos. Analise a complexidade de tempo com Heap Binário.",
        "response": r'''### Análise de Complexidade Assintótica

O algoritmo de Dijkstra calcula a menor distância de um vértice de origem para todos os outros em um grafo direcionado $G = (V, E)$.

Utilizando uma Fila de Prioridades baseada em **Min-Heap (Heap Binário)**:
- **Extração do Mínimo:** Cada vértice é extraído no máximo uma vez: $|V| \times O(\log |V|)$
- **Relaxamento de Arestas:** Cada aresta é analisada uma vez, atualizando distâncias: $|E| \times O(\log |V|)$
- **Complexidade Total de Tempo:** $\mathcal{O}((|V| + |E|) \log |V|)$
- **Complexidade de Espaço:** $\mathcal{O}(|V| + |E|)$

---

### Implementação Canônica em Python (Inglês Técnico)

```python
import heapq
from typing import Dict, List, Tuple, Optional

def dijkstra_shortest_path(
    graph: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str
) -> Tuple[float, List[str]]:
    """
    Finds the shortest path and minimum cost from source to target node using Dijkstra's algorithm.
    """
    # Priority Queue stores tuples of (current_cost, current_node)
    pq: List[Tuple[float, str]] = [(0.0, source)]
    distances: Dict[str, float] = {source: 0.0}
    predecessors: Dict[str, Optional[str]] = {source: None}
    visited = set()

    while pq:
        curr_dist, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v, weight in graph.get(u, []):
            if v in visited:
                continue
            new_dist = curr_dist + weight
            if new_dist < distances.get(v, float("inf")):
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))

    if target not in distances:
        return float("inf"), []

    # Reconstruct the optimal path from target to source
    path = []
    curr: Optional[str] = target
    while curr is not None:
        path.append(curr)
        curr = predecessors[curr]
    path.reverse()

    return distances[target], path
```'''
    },

    # 4. PROGRAMAÇÃO DINÂMICA & TEORIA DA COMPUTAÇÃO
    {
        "prompt": "Como resolver o Problema da Mochila 0/1 (0-1 Knapsack Problem) usando Programação Dinâmica? Explique a relação de recorrência e forneça a solução com otimização de espaço O(W).",
        "response": r'''### Relação de Recorrência da Programação Dinâmica

Seja $DP[i][w]$ o valor máximo possível considerando os primeiros $i$ itens com uma capacidade residual $w$:

1. Se o peso do item atual for maior que a capacidade ($peso[i-1] > w$):
   $$DP[i][w] = DP[i-1][w]$$
2. Se o item couber ($peso[i-1] \le w$), escolhemos o máximo entre não levar ou levar o item:
   $$DP[i][w] = \max\left(DP[i-1][w], \quad valor[i-1] + DP[i-1][w - peso[i-1]]\right)$$

#### Otimização de Espaço $\mathcal{O}(W)$:
Como a linha $i$ depende apenas da linha anterior $i-1$, podemos compactar a matriz 2D em um vetor 1D de tamanho $W+1$, iterando a capacidade de trás para frente (da direita para a esquerda) para evitar reuso do mesmo item na mesma iteração.

---

### Solução Otimizada em Python

```python
from typing import List

def knapsack_01_space_optimized(
    weights: List[int],
    values: List[int],
    capacity: int
) -> int:
    """
    Computes optimal 0-1 Knapsack value in O(N * W) time and O(W) space complexity.
    """
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        # Iterate backwards to prevent item re-use
        for w in range(capacity, w_i - 1, -1):
            dp[w] = max(dp[w], v_i + dp[w - w_i])

    return dp[capacity]

# Exemplo de teste:
pesos = [2, 3, 4, 5]
valores = [3, 4, 5, 8]
cap = 5
resultado = knapsack_01_space_optimized(pesos, valores, cap)
print(f"Valor Máximo da Mochila: {resultado}")
assert resultado == 8  # Escolhe o item de peso 5 e valor 8
```'''
    },

    # 5. MATEMÁTICA DISCRETA & TEORIA DOS NÚMEROS (CRIPTO RSA)
    {
        "prompt": "Explique o Algoritmo de Euclides Estendido e como encontrar o inverso multiplicativo modular. Implemente em Python e demonstre a aplicação em criptografia RSA.",
        "response": r'''### Fundamentação Teórica

O Algoritmo de Euclides Estendido encontra os coeficientes inteiros de Bézout $x$ e $y$ tais que:
$$a \cdot x + b \cdot y = \gcd(a, b)$$

Quando $\gcd(a, m) = 1$, temos:
$$a \cdot x \equiv 1 \pmod{m}$$
onde $x$ é o **Inverso Multiplicativo Modular** de $a$ módulo $m$ (notação: $a^{-1} \pmod{m}$).

No algoritmo **RSA**, esse cálculo é fundamental para derivar a chave privada $d$ a partir do expoente público $e$ e do totiente de Euler $\phi(n) = (p-1)(q-1)$:
$$d \equiv e^{-1} \pmod{\phi(n)}$$

---

### Implementação em Python

```python
from typing import Tuple

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    Time Complexity: O(log(min(a, b)))
    """
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def modular_inverse(e: int, phi: int) -> int:
    """Calculates d such that (d * e) % phi == 1."""
    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for e={e} and phi={phi}")
    return x % phi

# Exemplo RSA com primos pequenos:
p, q = 61, 53
n = p * q            # 3233
phi = (p - 1) * (q - 1) # 3120
e = 17               # Chave pública coprima com phi

d = modular_inverse(e, phi)
print(f"Chave Privada d: {d}") # 2753
assert (d * e) % phi == 1

# Teste de Encriptação e Decriptação:
mensagem = 65  # Caractere 'A'
cifra = pow(mensagem, e, n)
recuperado = pow(cifra, d, n)
assert recuperado == mensagem
print(f"Mensagem Original: {mensagem} -> Criptografado: {cifra} -> Decriptado: {recuperado}")
```'''
    }
]

def export_triad_dataset(output_sft: str, output_pretrain: str):
    """Exporta o currículo de alta densidade da Tríade para os datasets."""
    with open(output_sft, "w", encoding="utf-8") as f_sft, open(output_pretrain, "w", encoding="utf-8") as f_pre:
        for item in TRIAD_CURRICULUM:
            f_sft.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_pre.write(json.dumps({"content": item["response"]}, ensure_ascii=False) + "\n")
            f_pre.write(json.dumps({"content": f"# Problema:\n{item['prompt']}\n\n# Solução:\n{item['response']}"}, ensure_ascii=False) + "\n")
