import os
import json
import argparse
import torch
import sentencepiece as spm
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel

def format_chat_prompt(sp: spm.SentencePieceProcessor, instruction: str, response: str) -> list[int]:
    """
    Formata instrução e resposta no padrão ChatML nativo:
    <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    """
    im_start = sp.piece_to_id("<|im_start|>")
    im_end = sp.piece_to_id("<|im_end|>")

    user_tag = sp.encode("user\n")
    asst_tag = sp.encode("assistant\n")

    inst_ids = sp.encode(instruction)
    resp_ids = sp.encode(response)

    tokens = (
        [im_start] + user_tag + inst_ids + [im_end] +
        sp.encode("\n") +
        [im_start] + asst_tag + resp_ids + [im_end]
    )
    return tokens

def train_sft(
    base_checkpoint: str,
    sft_data_path: str = "data/raw/code_sft.jsonl",
    tokenizer_path: str = "data/tokenized/lloavece_code_bpe.model",
    out_dir: str = "checkpoints_sft",
    drive_dir: str = "",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 4
):
    os.makedirs(out_dir, exist_ok=True)
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("=== ALINHAMENTO E SFT LLOAVECE-CODER (PORTUGUÊS TÉCNICO) ===")
    print("=" * 65)
    print(f"Dispositivo: {device.upper()}")

    # Carrega Tokenizer
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    # Carrega Checkpoint Base Pré-Treinado
    print(f"Carregando pesos pré-treinados de {base_checkpoint}...")
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = LloaveceModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    # Otimizador com LR suave para não esquecer os dados pré-treinados
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Carrega dados SFT
    records = []
    with open(sft_data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = format_chat_prompt(sp, data["prompt"], data["response"])
            if len(tokens) <= config.block_size:
                records.append(tokens)

    print(f"Total de conversas técnicas carregadas: {len(records)}")

    # Loop de Fine-Tuning
    for epoch in range(epochs):
        total_loss = 0.0
        for i, tokens in enumerate(records):
            # Pad ou corta para bloco
            x_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, loss = model(x_ids, y_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{epochs} | Amostra {i + 1}/{len(records)} | Loss: {loss.item():.4f}")

        # Salva o modelo alinhado
        sft_path = os.path.join(out_dir, f"lloavece_sft_epoch_{epoch + 1}.pt")
        save_dict = {"model": model.state_dict(), "config": config, "epoch": epoch + 1}
        torch.save(save_dict, sft_path)
        if drive_dir:
            torch.save(save_dict, os.path.join(drive_dir, f"lloavece_sft_epoch_{epoch + 1}.pt"))
        print(f"[*] Modelo SFT salvo em: {sft_path}")

    print("\n[OK] Alinhamento SFT concluído com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data", type=str, default="data/raw/code_sft.jsonl")
    parser.add_argument("--drive_dir", type=str, default="")
    args = parser.parse_args()

    train_sft(base_checkpoint=args.checkpoint, sft_data_path=args.data, drive_dir=args.drive_dir)
