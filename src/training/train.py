import os
import math
import time
import argparse
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn import functional as F
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel

def get_lr_scheduler(it: int, warmup_iters: int, lr_decay_iters: int, min_lr: float, max_lr: float) -> float:
    """Cosine learning rate schedule com linear warmup."""
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, device: str):
    filename = os.path.join(data_dir, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(filename, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if "cuda" in device:
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: LloaveceModel, data_dir: str, eval_iters: int, batch_size: int, block_size: int, device: str, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, split, batch_size, block_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description="Treinador de Produção Lloavece-Coder")
    parser.add_argument("--config", type=str, default="0.5B", choices=["test", "0.5B", "1.5B", "3B", "7B"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--drive_dir", type=str, default="", help="Caminho do Google Drive para backup automático")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size por passo")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Passos de acumulação de gradiente")
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.drive_dir:
        os.makedirs(args.drive_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"=== INICIANDO TREINAMENTO LLOAVECE-CODER ({args.config}) ===")
    print("=" * 65)
    print(f"Hardware detectado    : {device.upper()}")
    if device == "cuda":
        print(f"GPU                   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Disponível       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Configuração de Precisão
    dtype = torch.bfloat16 if (args.precision == "bf16" and device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if (args.precision == "fp16" and device == "cuda") else torch.float32
    )
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" else nullcontext()
    print(f"Precisão de Treino    : {dtype}")

    # Carrega Arquitetura
    config = LloaveceConfig.from_name(args.config)
    model = LloaveceModel(config).to(device)

    total_params = model.get_num_params()
    print(f"Total de Parâmetros   : {total_params / 1e6:.2f} Milhões")
    print(f"Tokens por Passo      : {args.batch_size * args.grad_accum_steps * config.block_size:,}")

    # Otimizador AdamW desabilitando weight decay em RMSNorm e embeddings
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # Retoma Checkpoint se existir
    ckpt_path = os.path.join(args.out_dir, "latest_ckpt.pt")
    start_iter = 0
    best_val_loss = float("inf")

    if os.path.exists(ckpt_path):
        print(f"\n[!] Checkpoint encontrado em {ckpt_path}. Retomando treino...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[OK] Retomado a partir do passo {start_iter}")

    # Loop Principal de Treinamento
    t0 = time.time()
    for iter_num in range(start_iter, args.max_iters):
        lr = get_lr_scheduler(iter_num, args.warmup_iters, args.max_iters, args.min_lr, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Validação periódica
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, args.data_dir, args.eval_iters, args.batch_size, config.block_size, device, ctx)
            print(f"\n[Passo {iter_num:6d}] Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

            # Salva o melhor modelo
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": iter_num,
                    "config": config,
                    "best_val_loss": best_val_loss,
                }
                torch.save(save_payload, os.path.join(args.out_dir, "best_model.pt"))
                if args.drive_dir:
                    torch.save(save_payload, os.path.join(args.drive_dir, "best_model.pt"))
                print(f"[*] Melhor modelo salvo com sucesso! (Val Loss: {best_val_loss:.4f})")

        # Acumulação de Gradiente
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.grad_accum_steps):
            X, Y = get_batch(args.data_dir, "train", args.batch_size, config.block_size, device)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.grad_accum_steps

            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Gradient Clipping (evita explosão de gradiente)
        if dtype == torch.float16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Log de Velocidade
        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tok_sec = (args.batch_size * args.grad_accum_steps * config.block_size) / dt if dt > 0 else 0
            print(f"Passo {iter_num:6d}/{args.max_iters} | Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {lr:.2e} | Vel: {tok_sec:,.0f} tok/s")

    print("\n[OK] Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    main()
