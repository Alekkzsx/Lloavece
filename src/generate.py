import os
import argparse
import torch
import sentencepiece as spm
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel

def generate_code(
    checkpoint_path: str,
    tokenizer_path: str = "data/tokenized/lloavece_code_bpe.model",
    prompt: str = "",
    fim_prefix: str = "",
    fim_suffix: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando Lloavece-Coder em {device.upper()}...")

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = LloaveceModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Prepara input
    if fim_prefix or fim_suffix:
        # Modo Fill-In-The-Middle
        fim_p_id = sp.piece_to_id("<|fim_prefix|>")
        fim_s_id = sp.piece_to_id("<|fim_suffix|>")
        fim_m_id = sp.piece_to_id("<|fim_middle|>")

        tokens = [fim_p_id] + sp.encode(fim_prefix) + [fim_s_id] + sp.encode(fim_suffix) + [fim_m_id]
        print("\n=== MODO FIM (PREENCHIMENTO DE CÓDIGO) ===")
        print(f"[Prefixo]:\n{fim_prefix}")
        print(f"[Sufixo] :\n{fim_suffix}")
    else:
        # Modo Geração Normal
        tokens = sp.encode(prompt)
        print(f"\n[Prompt]: {prompt}")

    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    eos_id = sp.piece_to_id("<|eos|>")

    print("\n=== LLOAVECE GERANDO CÓDIGO... ===")
    out_tokens = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_id=eos_id
    )

    generated = sp.decode(out_tokens[0].tolist())
    print(generated)
    return generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenized/lloavece_code_bpe.model")
    parser.add_argument("--prompt", type=str, default="def calcular_desconto_pix(valor: float) -> float:")
    parser.add_argument("--fim_prefix", type=str, default="")
    parser.add_argument("--fim_suffix", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    generate_code(
        args.checkpoint,
        args.tokenizer,
        prompt=args.prompt,
        fim_prefix=args.fim_prefix,
        fim_suffix=args.fim_suffix,
        max_new_tokens=args.max_tokens
    )
