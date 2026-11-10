import os
import sys
import time
import shutil
import subprocess

TARGET_DIR = "/content/kimi_k31_pro"
REPO_ID = "moonshotai/Kimi-K3.1-Pro"
PORT = 8000

def setup_and_run():
    print("\n" + "#" * 80)
    print("### SETUP DEDICADO DO KIMI-K3.1-PRO (1M DE CONTEXTO & K-BIT) NO COLAB ###")
    print("#" * 80)
    print(f"Diretório Alvo : {TARGET_DIR}")
    print(f"Modelo HF      : {REPO_ID}")
    print(f"Modo           : 100% Exclusivo Kimi-K3.1-Pro (Sem outros modelos)")
    print("#" * 80 + "\n")

    # 1. Cria a pasta dedicada se não existir
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "checkpoints"), exist_ok=True)

    # 2. Copia todos os arquivos para a pasta dedicada /content/kimi_k31_pro
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for item in os.listdir(current_dir):
        s = os.path.join(current_dir, item)
        d = os.path.join(TARGET_DIR, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
        elif os.path.isdir(s) and item != "__pycache__":
            if not os.path.exists(d):
                shutil.copytree(s, d)

    os.chdir(TARGET_DIR)
    if TARGET_DIR not in sys.path:
        sys.path.insert(0, TARGET_DIR)
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # 3. Instala dependências essenciais
    print("[1/4] Instalando dependências necessárias...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "fastapi", "uvicorn", "pydantic", "huggingface_hub", "safetensors", "transformers", "accelerate",
        "-q"
    ], check=False)

    # 4. Garante binário cloudflared
    print("\n[2/4] Verificando binário Cloudflared...")
    if not shutil.which("cloudflared") and not os.path.exists("/usr/local/bin/cloudflared"):
        print("      Baixando cloudflared...")
        subprocess.run([
            "curl", "-sL",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", "/usr/local/bin/cloudflared"
        ], check=False)
        subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=False)
    print("      [OK] Cloudflared pronto.")

    # 5. Executa download e compressão K-Bit
    print("\n[3/4] Executando download e fatorização K-Bit para Kimi-K3.1-Pro...")
    from kimi_k31_pro.downloader import download_and_kbit_compress
    ckpt = download_and_kbit_compress(
        repo_id=REPO_ID,
        output_dir=os.path.join(TARGET_DIR, "checkpoints"),
        temp_dir=os.path.join(TARGET_DIR, "temp_hf"),
        rank=4,
        max_shards_to_process=2
    )

    # 6. Inicia o servidor e o túnel Cloudflared
    print("\n[4/4] Subindo o Servidor FastAPI e o Túnel Público...")
    from kimi_k31_pro.server import app, start_cloudflared_tunnel
    import uvicorn

    tunnel_url = start_cloudflared_tunnel(port=PORT)

    print("\n" + "=" * 80)
    print("🎉 KIMI-K3.1-PRO PRONTO E OPERANDO EXCLUSIVAMENTE!")
    if tunnel_url:
        print(f"👉 Endpoint Base URL: {tunnel_url}/v1")
    else:
        print(f"👉 Endpoint Local   : http://127.0.0.1:{PORT}/v1")
    print(f"👉 Modelo ID        : {REPO_ID} (ou kimi-k3.1-pro)")
    print(f"👉 Context Window   : 1.000.000 tokens (1M)")
    print(f"👉 Ferramentas Cline: read_file, write_to_file, replace_in_file, list_files, execute_command")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    setup_and_run()
