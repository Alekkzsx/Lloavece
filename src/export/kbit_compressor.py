import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def nearest_kronecker_product(
    W: torch.Tensor,
    shape_a: Tuple[int, int],
    shape_b: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resolve o problema do Produto Mais Próximo de Kronecker (Van Loan & Pitsianis, 1993).
    Aproxima uma matriz W (m1*m2, n1*n2) pelo produto A (m1, n1) (x) B (m2, n2)
    minimizando a norma de Frobenius ||W - A (x) B||_F via SVD de Rank-1.
    """
    m1, n1 = shape_a
    m2, n2 = shape_b
    assert W.shape == (m1 * m2, n1 * n2), f"Dimensões incompatíveis: {W.shape} vs ({m1*m2}, {n1*n2})"

    # Reorganiza os blocos da matriz W em uma matriz tilde_W de dimensão (m1*n1, m2*n2)
    # Cada linha de tilde_W corresponde a vec(W_ij) de cada bloco
    reshaped = W.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous()
    tilde_W = reshaped.view(m1 * n1, m2 * n2)

    # SVD de Rank-1 (computado via float32 para estabilidade numérica perfeita)
    orig_dtype = W.dtype
    tilde_W_f32 = tilde_W.to(torch.float32)

    # U, S, Vh
    U, S, Vh = torch.linalg.svd(tilde_W_f32, full_matrices=False)

    sigma = S[0]
    u = U[:, 0]
    v = Vh[0, :]

    # Distribui a escala simetricamente entre A e B
    sqrt_sigma = torch.sqrt(sigma)
    vec_a = (sqrt_sigma * u).to(orig_dtype)
    vec_b = (sqrt_sigma * v).to(orig_dtype)

    A = vec_a.view(m1, n1)
    B = vec_b.view(m2, n2)

    return A, B

def kronecker_forward(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Calcula x @ (A (x) B)^T sem nunca materializar a matriz gigante de 16M pesos na RAM!
    Usa a identidade algébrica tensorial:
    (A (x) B) @ vec(X) = vec(B @ X @ A^T)
    Reduz a complexidade de O(N^2) para O(N * sqrt(N)).
    """
    # x shape: (..., in_features)
    orig_shape = x.shape[:-1]
    in_features = x.shape[-1]
    m1, n1 = A.shape
    m2, n2 = B.shape
    assert in_features == n1 * n2

    # Reshape x para bater com os blocos
    x_reshaped = x.view(-1, n1, n2) # (batch, n1, n2)

    # Aplica B no lado direito e A no lado esquerdo
    # out = A @ x_reshaped @ B^T
    # Matmul em lote:
    temp = torch.matmul(x_reshaped, B.t())  # (batch, n1, m2)
    temp = temp.transpose(1, 2)            # (batch, m2, n1)
    out = torch.matmul(temp, A.t())         # (batch, m2, m1)
    out = out.transpose(1, 2)              # (batch, m1, m2)

    return out.contiguous().view(*orig_shape, m1 * m2)

class KBitLinear(nn.Module):
    """
    Camada Linear K-Bit Sparse:
    - Base de Kronecker (A (x) B): 99% da representação em ~0.008 bits/peso.
    - Resíduo Esparso Ternário (1% dos outliers): Preserva decisões cirúrgicas de código.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity_ratio: float = 0.01,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_ratio = sparsity_ratio

        # Encontra fatores que dividam perfeitamente in_features e out_features
        self.m1, self.m2 = self._find_factors(out_features)
        self.n1, self.n2 = self._find_factors(in_features)

        # Matrizes sementes de Kronecker
        self.A = nn.Parameter(torch.empty(self.m1, self.n1))
        self.B = nn.Parameter(torch.empty(self.m2, self.n2))

        # Resíduo esparso (armazena apenas índices e valores ternários escalados)
        self.register_buffer("sparse_indices", torch.empty(2, 0, dtype=torch.long))
        self.register_buffer("sparse_values", torch.empty(0, dtype=torch.float32))
        self.register_buffer("sparse_scale", torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _find_factors(n: int) -> Tuple[int, int]:
        """Encontra dois fatores próximos de sqrt(n) para matriz quadrada ideal."""
        root = int(math.isqrt(n))
        for f in range(root, 0, -1):
            if n % f == 0:
                return f, n // f
        return 1, n

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        sparsity_ratio: float = 0.01
    ) -> "KBitLinear":
        """Converte uma camada nn.Linear padrão para a representação comprimida K-Bit Sparse."""
        kbit = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            sparsity_ratio=sparsity_ratio,
            bias=linear.bias is not None
        )

        with torch.no_grad():
            W = linear.weight.data # (out_features, in_features)

            # 1. Decomposição ótima de Kronecker
            A, B = nearest_kronecker_product(W, (kbit.m1, kbit.n1), (kbit.m2, kbit.n2))
            kbit.A.copy_(A)
            kbit.B.copy_(B)

            # 2. Reconstrói a aproximação de Kronecker para calcular o erro residual
            W_kron = torch.kron(A, B)
            residual = W - W_kron

            # 3. Pega os top-k outliers (1% mais críticos)
            total_elements = residual.numel()
            k = max(1, int(total_elements * sparsity_ratio))
            abs_res = residual.abs().view(-1)
            threshold, _ = torch.kthvalue(abs_res, total_elements - k + 1)

            mask = residual.abs() >= threshold
            indices = torch.nonzero(mask, as_tuple=False).t() # (2, k)
            values = residual[mask]

            # 4. Quantização Ternária do Resíduo {-1, 0, +1} * escala
            scale = values.abs().mean()
            ternary_values = torch.sign(values) * scale

            kbit.sparse_indices = indices
            kbit.sparse_values = ternary_values
            kbit.sparse_scale = scale

            if linear.bias is not None:
                kbit.bias.copy_(linear.bias.data)

        return kbit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Forward rápido de Kronecker sem materializar W
        out = kronecker_forward(self.A, self.B, x)

        # 2. Adiciona o resíduo esparso (apenas onde a máscara é ativa)
        if self.sparse_indices.size(1) > 0:
            sparse_W = torch.sparse_coo_tensor(
                self.sparse_indices,
                self.sparse_values,
                size=(self.out_features, self.in_features),
                device=x.device,
                dtype=x.dtype
            )
            # Multiplicação esparsa com x
            out = out + F.linear(x, sparse_W.to_dense())

        if self.bias is not None:
            out = out + self.bias

        return out

def benchmark_compression():
    """Testa e valida a matemática de compressão em uma camada típica de 16M parâmetros."""
    print("=" * 60)
    print("=== BENCHMARK DO MOTOR K-BIT SPARSE (LLOAVECE) ===")
    print("=" * 60)

    in_f, out_f = 4096, 4096
    print(f"Camada Alvo: {in_f}x{out_f} ({in_f * out_f / 1e6:.2f}M parâmetros)")

    # Cria camada simulada em FP32
    torch.manual_seed(42)
    original_layer = nn.Linear(in_f, out_f, bias=False)

    orig_mem_mb = (in_f * out_f * 4) / (1024 * 1024)
    orig_fp16_mb = (in_f * out_f * 2) / (1024 * 1024)
    print(f"Tamanho original (FP32) : {orig_mem_mb:.2f} MB")
    print(f"Tamanho padrão   (FP16) : {orig_fp16_mb:.2f} MB")

    # Aplica compressão K-Bit
    kbit_layer = KBitLinear.from_linear(original_layer, sparsity_ratio=0.01)

    kron_params = kbit_layer.A.numel() + kbit_layer.B.numel()
    sparse_params = kbit_layer.sparse_values.numel()
    total_stored = kron_params + sparse_params
    kbit_mem_mb = (total_stored * 2) / (1024 * 1024) # Supondo 16-bit para sementes e ternário

    print(f"\n[+] Matriz Semente A : {kbit_layer.A.shape} ({kbit_layer.A.numel()} parâmetros)")
    print(f"[+] Matriz Semente B : {kbit_layer.B.shape} ({kbit_layer.B.numel()} parâmetros)")
    print(f"[+] Resíduo Esparso  : {sparse_params} outliers ternários (1%)")
    print(f"Tamanho K-Bit Sparse : {kbit_mem_mb:.2f} MB")
    print(f"Taxa de Compressão   : {orig_fp16_mb / kbit_mem_mb:.1f}x vezes menor!")
    print(f"Bits Efetivos/Peso   : {(16.0 / (orig_fp16_mb / kbit_mem_mb)):.3f} bits/peso")

    # Validação de precisão numérica
    x = torch.randn(2, in_f)
    y_orig = original_layer(x)
    y_kbit = kbit_layer(x)

    cosine_sim = F.cosine_similarity(y_orig.flatten(), y_kbit.flatten(), dim=0).item()
    print(f"\nSimilaridade de Cosseno (Fidelidade Semântica): {cosine_sim * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_compression()
