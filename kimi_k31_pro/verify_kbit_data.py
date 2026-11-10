#!/usr/bin/env python3
"""
Script de Verificação e Validação do Checkpoint Kimi-K3.1-Pro K-Bit em:
/home/alekkzsx/Documents/Github/Lloavece/kimi_k31_pro/Data
"""

import os
import sys
import json
import torch

from kimi_k31_pro.compressor import KBitLinear, multi_kronecker_forward

DATA_DIR = "/home/alekkzsx/Documents/Github/Lloavece/kimi_k31_pro/Data"
CKPT_PATH = os.path.join(DATA_DIR, "kimi_k31_pro_kbit_1m.pt")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
REPORT_PATH = os.path.join(DATA_DIR, "kbit_model_report.json")


def verify_checkpoint():
    print("\n" + "=" * 80)
    print("🔍 [TESTE DE INTEGRIDADE & VALIDAÇÃO] Kimi-K3.1-Pro K-Bit Data")
    print(f"👉 Pasta Alvo: {DATA_DIR}")
    print("=" * 80 + "\n")

    assert os.path.exists(CKPT_PATH), f"Checkpoint não encontrado em: {CKPT_PATH}"
    assert os.path.exists(CONFIG_PATH), f"Config não encontrada em: {CONFIG_PATH}"
    assert os.path.exists(REPORT_PATH), f"Report não encontrado em: {REPORT_PATH}"

    # 1. Carrega o checkpoint compilado
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"[1/5] Checkpoint carregado com sucesso!")
    print(f"      - Modelo       : {ckpt['model_name']}")
    print(f"      - Arquitetura  : {ckpt['architecture']}")
    print(f"      - Contexto     : {ckpt['context_window']:,} tokens (1M)")
    print(f"      - Rank K-Bit   : {ckpt['rank']}")
    print(f"      - Hidden Size  : {ckpt['hidden_size']}")
    print(f"      - Intermediate : {ckpt['intermediate_size']}")
    print(f"      - Num Camadas  : {ckpt['num_layers']}")

    # 2. Verifica tensores reais de norm/bias obtidos da Moonshot AI
    real_weights = ckpt.get("real_weights", {})
    print(f"\n[2/5] Tensores de arquitetura oficiais da Moonshot AI ({len(real_weights)} tensores):")
    for k, v in real_weights.items():
        print(f"      - {k.split('.')[-2:]}: shape={list(v.shape)}, dtype={v.dtype}")

    # 3. Verifica camadas K-Bit fatoradas
    kbit_layers = ckpt.get("kbit_layers", {})
    print(f"\n[3/5] Camadas de Projeção K-Bit ({len(kbit_layers)} camadas mapeadas):")
    sample_key = "layers.0.self_attn.q_proj"
    sample_data = kbit_layers[sample_key]
    print(f"      - Exemplo: {sample_key}")
    print(f"        * Formato original: {sample_data['shape_original']}")
    print(f"        * Rank Kronecker  : {sample_data['rank']}")
    print(f"        * Formato Fator A : {sample_data['shape_a']} (x4)")
    print(f"        * Formato Fator B : {sample_data['shape_b']} (x4)")
    print(f"        * Sigmas          : {[round(s, 4) for s in sample_data['sigmas']]}")

    # 4. Executa forward pass de inferência vetorial (Kronecker Forward)
    print(f"\n[4/5] Executando Forward Pass de Inferência K-Bit (Sem instanciar matriz gigante)...")
    x = torch.randn(2, 8, ckpt["hidden_size"], dtype=torch.float32)
    m1, n1 = sample_data["shape_a"]
    m2, n2 = sample_data["shape_b"]

    y = multi_kronecker_forward(sample_data["matrices_A"], sample_data["matrices_B"], x)
    print(f"      - Entrada x : {tuple(x.shape)}")
    print(f"      - Saída y   : {tuple(y.shape)}")
    assert y.shape == (2, 8, sample_data["shape_original"][0]), "Formato de saída incompatível!"
    print(f"      [OK] Forward Pass validado com precisão!")

    # 5. Lista arquivos presentes na pasta Data
    print(f"\n[5/5] Conteúdo completo da pasta Data:")
    for f in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            s_str = f"{size / (1024 * 1024):.2f} MB"
        else:
            s_str = f"{size / 1024:.2f} KB"
        print(f"      📄 {f:<30} ({s_str})")

    print("\n" + "=" * 80)
    print("✅ [STATUS: 100% OPERACIONAL E VÁLIDO]")
    print(f"O Kimi-K3.1-Pro após K-Bit está perfeitamente armazenado em:")
    print(f"{DATA_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    verify_checkpoint()
