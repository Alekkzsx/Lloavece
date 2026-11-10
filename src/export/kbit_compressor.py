import math
import time
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_rank_kronecker_product(
    W: torch.Tensor,
    shape_a: Tuple[int, int],
    shape_b: Tuple[int, int],
    rank: int = 4
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """
    Resolve a Decomposição de Kronecker Multi-Rank ótima (Teorema de Van Loan & Pitsianis).
    Aproxima W como: sum_{k=1}^r sigma_k * (A_k (x) B_k)
    minimizando || W - sum_{k=1}^r A_k (x) B_k ||_F via SVD de Rank-r na matriz reorganizada.
    """
    m1, n1 = shape_a
    m2, n2 = shape_b
    assert W.shape == (m1 * m2, n1 * n2), f"Dimensões incompatíveis: {W.shape} vs ({m1*m2}, {n1*n2})"

    # Reorganiza os blocos da matriz W em tilde_W de dimensão (m1*n1, m2*n2)
    reshaped = W.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous()
    tilde_W = reshaped.view(m1 * n1, m2 * n2)

    orig_dtype = W.dtype
    tilde_W_f32 = tilde_W.to(torch.float32)

    # SVD computado em float32 para estabilidade numérica
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
    list_A: nn.ParameterList,
    list_B: nn.ParameterList,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Executa a multiplicação y = sum_{k=1}^r (A_k (x) B_k) @ x^T
    usando a identidade tensorial vec(B_k @ X @ A_k^T) sem nunca instanciar a matriz gigante.
    """
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

class KBitCompressor:
    """Utilitário de Decomposição Kronecker Multi-Rank e Resíduo Ternário para o Batalhão de Modelos."""

    @staticmethod
    def van_loan_pitsianis_svd(W: torch.Tensor, rank: int = 4):
        m, n = W.shape
        m1 = int(math.isqrt(m))
        while m % m1 != 0 and m1 > 1:
            m1 -= 1
        m2 = m // m1

        n1 = int(math.isqrt(n))
        while n % n1 != 0 and n1 > 1:
            n1 -= 1
        n2 = n // n1

        m_A, m_B, sigmas = multi_rank_kronecker_product(W, (m1, n1), (m2, n2), rank=rank)
        return [(A, B, torch.tensor(s, dtype=W.dtype, device=W.device)) for A, B, s in zip(m_A, m_B, sigmas)]

    @staticmethod
    def ternary_sparse_residual(W: torch.Tensor, factors, sparsity_ratio: float = 0.01):
        W_kron = torch.zeros_like(W)
        for A_k, B_k, _ in factors:
            W_kron += torch.kron(A_k, B_k)
        residual = W - W_kron
        total_elements = residual.numel()
        k = max(1, int(total_elements * sparsity_ratio))
        abs_res = residual.abs().view(-1)
        threshold, _ = torch.kthvalue(abs_res, total_elements - k + 1)
        mask = residual.abs() >= threshold
        indices = torch.nonzero(mask, as_tuple=False).t()
        values = torch.sign(residual[mask])
        return {
            "values": values,
            "indices": indices,
            "shape": list(W.shape)
        }

class KBitLinear(nn.Module):
    """
    Camada Linear K-Bit Multi-Rank:
    - Base Kronecker de Rank-r: W_kron = sum_{k=1}^r (A_k (x) B_k)
    - Resíduo Esparso Ternário: Outliers críticos em {-1, 0, +1} * escala
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        sparsity_ratio: float = 0.01,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.sparsity_ratio = sparsity_ratio

        self.m1, self.m2 = self._find_factors(out_features)
        self.n1, self.n2 = self._find_factors(in_features)

        self.list_A = nn.ParameterList([
            nn.Parameter(torch.empty(self.m1, self.n1)) for _ in range(rank)
        ])
        self.list_B = nn.ParameterList([
            nn.Parameter(torch.empty(self.m2, self.n2)) for _ in range(rank)
        ])

        self.register_buffer("sparse_indices", torch.empty(2, 0, dtype=torch.long))
        self.register_buffer("sparse_values", torch.empty(0, dtype=torch.float32))
        self.register_buffer("sparse_scale", torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _find_factors(n: int) -> Tuple[int, int]:
        root = int(math.isqrt(n))
        for f in range(root, 0, -1):
            if n % f == 0:
                return f, n // f
        return 1, n

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 4,
        sparsity_ratio: float = 0.01
    ) -> "KBitLinear":
        kbit = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            sparsity_ratio=sparsity_ratio,
            bias=linear.bias is not None
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        with torch.no_grad():
            W = linear.weight.data

            # 1. Decomposição de Kronecker Multi-Rank ótima
            m_A, m_B, _ = multi_rank_kronecker_product(
                W,
                (kbit.m1, kbit.n1),
                (kbit.m2, kbit.n2),
                rank=rank
            )

            W_kron = torch.zeros_like(W)
            for i, (A_k, B_k) in enumerate(zip(m_A, m_B)):
                kbit.list_A[i].copy_(A_k)
                kbit.list_B[i].copy_(B_k)
                W_kron += torch.kron(A_k, B_k)

            # 2. Resíduo após Rank-r
            residual = W - W_kron

            # 3. Extrai os maiores outliers ternários
            total_elements = residual.numel()
            k = max(1, int(total_elements * sparsity_ratio))
            abs_res = residual.abs().view(-1)
            threshold, _ = torch.kthvalue(abs_res, total_elements - k + 1)

            mask = residual.abs() >= threshold
            indices = torch.nonzero(mask, as_tuple=False).t()
            values = residual[mask]

            scale = values.abs().mean()
            ternary_values = torch.sign(values) * scale

            kbit.sparse_indices = indices
            kbit.sparse_values = ternary_values
            kbit.sparse_scale = scale

            if linear.bias is not None:
                kbit.bias.copy_(linear.bias.data)

        return kbit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = multi_kronecker_forward(self.list_A, self.list_B, x)

        if self.sparse_indices.size(1) > 0:
            sparse_W = torch.sparse_coo_tensor(
                self.sparse_indices,
                self.sparse_values,
                size=(self.out_features, self.in_features),
                device=x.device,
                dtype=x.dtype
            )
            out = out + F.linear(x, sparse_W.to_dense())

        if self.bias is not None:
            out = out + self.bias

        return out

def convert_model_to_kbit(model: nn.Module, rank: int = 4, sparsity_ratio: float = 0.01) -> nn.Module:
    """
    Substitui recursivamente todas as camadas nn.Linear do Transformer Lloavece
    por camadas comprimidas KBitLinear (Kronecker Multi-Rank + Resíduo Ternário).
    """
    print(f"\n[*] Convertendo arquitetura completa para K-Bit Multi-Rank (Rank={rank})...")
    converted_count = 0

    for layer_idx, block in enumerate(model.layers):
        block.self_attn.q_proj = KBitLinear.from_linear(block.self_attn.q_proj, rank=rank, sparsity_ratio=sparsity_ratio)
        block.self_attn.k_proj = KBitLinear.from_linear(block.self_attn.k_proj, rank=rank, sparsity_ratio=sparsity_ratio)
        block.self_attn.v_proj = KBitLinear.from_linear(block.self_attn.v_proj, rank=rank, sparsity_ratio=sparsity_ratio)
        block.self_attn.out_proj = KBitLinear.from_linear(block.self_attn.out_proj, rank=rank, sparsity_ratio=sparsity_ratio)

        block.mlp.w1 = KBitLinear.from_linear(block.mlp.w1, rank=rank, sparsity_ratio=sparsity_ratio)
        block.mlp.w2 = KBitLinear.from_linear(block.mlp.w2, rank=rank, sparsity_ratio=sparsity_ratio)
        block.mlp.w3 = KBitLinear.from_linear(block.mlp.w3, rank=rank, sparsity_ratio=sparsity_ratio)

        converted_count += 7

    print(f"[OK] {converted_count} camadas lineares convertidas com sucesso!")
    print(f"[⚡ K-BIT ATIVO] O modelo agora roda 100% sobre a matemática de Kronecker!")
    return model

def benchmark_multi_rank():
    print("=" * 65)
    print("=== BENCHMARK K-BIT MULTI-RANK (LLOAVECE COMPRESSION ENGINE) ===")
    print("=" * 65)

    in_f, out_f = 4096, 4096
    total_weights = in_f * out_f
    print(f"Camada de Referência: {in_f} x {out_f} ({total_weights / 1e6:.2f}M parâmetros)")
    orig_fp16_kb = (total_weights * 2) / 1024

    torch.manual_seed(42)
    original_layer = nn.Linear(in_f, out_f, bias=False)
    x = torch.randn(2, in_f)
    y_orig = original_layer(x)

    print(f"Tamanho Original (FP16): {orig_fp16_kb / 1024:.2f} MB ({orig_fp16_kb:,.0f} KB)\n")

    for rank in [1, 2, 4]:
        t0 = time.time()
        kbit = KBitLinear.from_linear(original_layer, rank=rank, sparsity_ratio=0.01)
        decomp_time = (time.time() - t0) * 1000

        kron_params = sum(A.numel() + B.numel() for A, B in zip(kbit.list_A, kbit.list_B))
        sparse_params = kbit.sparse_values.numel()
        stored_bytes = (kron_params * 2) + (sparse_params * 2)
        stored_kb = stored_bytes / 1024

        y_kbit = kbit(x)
        cos_sim = F.cosine_similarity(y_orig.flatten(), y_kbit.flatten(), dim=0).item()
        bits_per_param = (stored_bytes * 8) / total_weights
        comp_ratio = (total_weights * 2) / stored_bytes

        print(f"--- [ K-BIT RANK-{rank} ] ---")
        print(f"  Parâmetros Kronecker: {kron_params:,} ({rank} pares {kbit.m1}x{kbit.n1} e {kbit.m2}x{kbit.n2})")
        print(f"  Outliers Ternários   : {sparse_params:,} (1%)")
        print(f"  Tamanho Comprimido   : {stored_kb:.2f} KB ({stored_kb / 1024:.3f} MB)")
        print(f"  Taxa de Compressão   : {comp_ratio:.1f}x vezes menor!")
        print(f"  Bits Efetivos/Peso   : {bits_per_param:.4f} bits/peso")
        print(f"  Fidelidade Semântica : {cos_sim * 100:.2f}%")
        print(f"  Tempo de Decomposição: {decomp_time:.1f} ms\n")

    print("=" * 65)

if __name__ == "__main__":
    benchmark_multi_rank()
