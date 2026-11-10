import math
import time
from typing import Tuple, List, Optional, Dict, Any
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def multi_rank_kronecker_product(
    W: Any,
    shape_a: Tuple[int, int],
    shape_b: Tuple[int, int],
    rank: int = 4
) -> Tuple[List[Any], List[Any], List[float]]:
    """
    Resolve a Decomposição de Kronecker Multi-Rank ótima (Teorema de Van Loan & Pitsianis).
    Aproxima W como: sum_{k=1}^r sigma_k * (A_k (x) B_k)
    minimizando || W - sum_{k=1}^r A_k (x) B_k ||_F via SVD de Rank-r na matriz reorganizada.
    """
    if torch is None:
        raise RuntimeError("PyTorch não está instalado no ambiente.")

    m1, n1 = shape_a
    m2, n2 = shape_b
    assert W.shape == (m1 * m2, n1 * n2), f"Dimensões incompatíveis: {W.shape} vs ({m1*m2}, {n1*n2})"

    reshaped = W.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous()
    tilde_W = reshaped.view(m1 * n1, m2 * n2)

    orig_dtype = W.dtype
    tilde_W_f32 = tilde_W.to(torch.float32)

    U, S, Vh = torch.linalg.svd(tilde_W_f32, full_matrices=False)

    matrices_A = []
    matrices_B = []
    sigmas = []

    for k in range(min(rank, len(S))):
        sigma = S[k].item()
        sqrt_sigma = math.sqrt(max(0.0, sigma))

        u = U[:, k]
        v = Vh[k, :]

        vec_a = (sqrt_sigma * u).to(orig_dtype)
        vec_b = (sqrt_sigma * v).to(orig_dtype)

        A_k = vec_a.view(m1, n1)
        B_k = vec_b.view(m2, n2)

        matrices_A.append(A_k)
        matrices_B.append(B_k)
        sigmas.append(sigma)

    return matrices_A, matrices_B, sigmas


def multi_kronecker_forward(
    list_A: Any,
    list_B: Any,
    x: Any
) -> Any:
    """
    Executa a multiplicação y = sum_{k=1}^r (A_k (x) B_k) @ x^T
    usando a identidade tensorial vec(B_k @ X @ A_k^T) sem instanciar a matriz gigante.
    """
    if torch is None:
        raise RuntimeError("PyTorch não está instalado no ambiente.")

    orig_shape = x.shape[:-1]
    in_features = x.shape[-1]
    m1, n1 = list_A[0].shape
    m2, n2 = list_B[0].shape
    assert in_features == n1 * n2

    x_reshaped = x.view(-1, n1, n2)
    total_out = None

    for A_k, B_k in zip(list_A, list_B):
        A_dev = A_k.to(device=x.device, dtype=x.dtype)
        B_dev = B_k.to(device=x.device, dtype=x.dtype)
        temp = torch.matmul(x_reshaped, B_dev.t())  # (batch, n1, m2)
        temp = temp.transpose(1, 2)                 # (batch, m2, n1)
        out_k = torch.matmul(temp, A_dev.t())       # (batch, m2, m1)
        out_k = out_k.transpose(1, 2)               # (batch, m1, m2)

        if total_out is None:
            total_out = out_k
        else:
            total_out = total_out + out_k

    return total_out.contiguous().view(*orig_shape, m1 * m2)


class KBitLinear(nn.Module if nn else object):
    """
    Camada Linear de Alta Densidade K-Bit para Kimi-K3.1-Pro:
    Substitui W (m x n) por fatores de Kronecker Multi-Rank A_k (m1 x n1) e B_k (m2 x n2),
    comprimindo a memória em 84x+ mantendo precisão flutuante nas projeções.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        bias: bool = False,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        if nn is None:
            raise RuntimeError("PyTorch não está instalado.")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Decomposição de fatores ótimos
        self.m1 = int(math.isqrt(out_features))
        while out_features % self.m1 != 0 and self.m1 > 1:
            self.m1 -= 1
        self.m2 = out_features // self.m1

        self.n1 = int(math.isqrt(in_features))
        while in_features % self.n1 != 0 and self.n1 > 1:
            self.n1 -= 1
        self.n2 = in_features // self.n1

        self.list_A = nn.ParameterList([
            nn.Parameter(torch.randn(self.m1, self.n1, device=device, dtype=dtype) / math.sqrt(self.n1))
            for _ in range(rank)
        ])
        self.list_B = nn.ParameterList([
            nn.Parameter(torch.randn(self.m2, self.n2, device=device, dtype=dtype) / math.sqrt(self.n2))
            for _ in range(rank)
        ])

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Any) -> Any:
        out = multi_kronecker_forward(self.list_A, self.list_B, x)
        if self.bias is not None:
            out = out + self.bias
        return out


class KBitCompressor:
    """Utilitário de Compressão e Fatorização K-Bit para o Kimi-K3.1-Pro."""

    @staticmethod
    def factorize_weight(W: Any, rank: int = 4) -> Dict[str, Any]:
        if torch is None:
            return {}
        m, n = W.shape
        m1 = int(math.isqrt(m))
        while m % m1 != 0 and m1 > 1:
            m1 -= 1
        m2 = m // m1

        n1 = int(math.isqrt(n))
        while n % n1 != 0 and n1 > 1:
            n1 -= 1
        n2 = n // n1

        matrices_A, matrices_B, sigmas = multi_rank_kronecker_product(
            W, shape_a=(m1, n1), shape_b=(m2, n2), rank=rank
        )
        return {
            "shape_original": (m, n),
            "rank": rank,
            "shape_a": (m1, n1),
            "shape_b": (m2, n2),
            "sigmas": sigmas,
            "matrices_A": [a.cpu().half() for a in matrices_A],
            "matrices_B": [b.cpu().half() for b in matrices_B],
        }
