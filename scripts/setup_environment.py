#!/usr/bin/env python3
"""
Script de setup do ambiente para o Lloavece Kode.
Verifica dependências, configura paths e prepara o ambiente de treino.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional

import torch


def check_python_version() -> bool:
    """Verifica se a versão do Python é >= 3.11"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python {version.major}.{version.minor} detectado. Requer Python 3.11+")
        return False
    print(f"✅ Python {version.major}.{version.minor} detectado")
    return True


def check_cuda() -> Tuple[bool, Optional[str]]:
    """Verifica disponibilidade de CUDA e GPU"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA não disponível. Treinamento será em CPU (muito lento)")
        return False, None
    
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    vram_total = sum(torch.cuda.get_device_properties(i).total_memory / 1e9 
                     for i in range(gpu_count))
    
    print(f"✅ CUDA disponível")
    print(f"   GPUs: {gpu_count}")
    for i, name in enumerate(gpu_names):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   - GPU {i}: {name} ({vram:.1f} GB)")
    print(f"   VRAM Total: {vram_total:.1f} GB")
    
    if vram_total < 20:
        print("⚠️  VRAM < 20GB. Considere usar QLoRA ou modelo menor.")
    
    return True, torch.version.cuda


def check_disk_space(required_gb: float = 500) -> bool:
    """Verifica espaço em disco disponível"""
    project_root = Path(__file__).parent.parent
    total, used, free = shutil.disk_usage(project_root)
    free_gb = free / (1024**3)
    
    if free_gb < required_gb:
        print(f"⚠️  Espaço em disco insuficiente: {free_gb:.1f} GB disponíveis, {required_gb} GB necessários")
        return False
    
    print(f"✅ Espaço em disco: {free_gb:.1f} GB disponíveis")
    return True


def check_dependencies() -> bool:
    """Verifica se as dependências principais estão instaladas"""
    required = [
        'torch', 'transformers', 'accelerate', 'peft', 'trl',
        'datasets', 'pandas', 'numpy', 'fastapi', 'wandb'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Dependências faltando: {', '.join(missing)}")
        print(f"   Execute: pip install -r requirements.txt")
        return False
    
    print(f"✅ Todas as dependências principais instaladas")
    return True


def create_directories() -> None:
    """Cria estrutura de diretórios necessária"""
    project_root = Path(__file__).parent.parent
    
    dirs = [
        'data/raw',
        'data/processed',
        'data/sft',
        'data/dpo',
        'data/self_improvement',
        'models/base',
        'models/cpt',
        'models/sft',
        'models/dpo',
        'models/self_improvement',
        'models/gguf',
        'training/configs',
        'training/scripts',
        'training/utils',
        'logs',
        'checkpoints',
    ]
    
    for dir_path in dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Estrutura de diretórios criada")


def setup_wandb() -> None:
    """Configura Weights & Biases para monitoramento"""
    try:
        import wandb
        
        if not os.getenv('WANDB_API_KEY'):
            print("⚠️  WANDB_API_KEY não configurado. Execute: wandb login")
        else:
            print(f"✅ Weights & Biases configurado")
    except ImportError:
        print("⚠️  wandb não instalado. Pule esta etapa se não for usar.")


def setup_git_lfs() -> bool:
    """Configura Git LFS para arquivos grandes"""
    try:
        result = subprocess.run(
            ['git', 'lfs', 'version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Git LFS instalado: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  Git LFS não instalado. Execute:")
    print("   Ubuntu/Debian: sudo apt install git-lfs")
    print("   macOS: brew install git-lfs")
    print("   Windows: git-lfs-win.exe")
    return False


def create_env_file() -> None:
    """Cria arquivo .env com configurações padrão"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    if env_file.exists():
        print(f"⚠️  .env já existe. Pulando criação.")
        return
    
    env_content = """# Lloavece Kode - Configurações de Ambiente

# Modelo
MODEL_NAME=Qwen/Qwen3.8-27B
MODEL_PATH=models/base/qwen3.8-27b

# Paths
DATA_DIR=data
MODELS_DIR=models
LOGS_DIR=logs
CHECKPOINTS_DIR=checkpoints

# Treinamento
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=4096

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# W&B
WANDB_PROJECT=lloavece-kode
WANDB_API_KEY=your_api_key_here

# CUDA
CUDA_VISIBLE_DEVICES=0

# Inferência
INFERENCE_PORT=8000
MAX_TOKENS=2048
TEMPERATURE=0.7
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Arquivo .env criado em {env_file}")
    print("⚠️  Edite o .env e configure sua WANDB_API_KEY")


def main():
    """Executa todas as verificações e configurações"""
    print("=" * 60)
    print("🚀 Lloavece Kode - Setup do Ambiente")
    print("=" * 60)
    print()
    
    # Verificações
    checks = [
        ("Python Version", check_python_version),
        ("CUDA/GPU", lambda: check_cuda()[0]),
        ("Disk Space", lambda: check_disk_space(500)),
        ("Dependencies", check_dependencies),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erro ao verificar {name}: {e}")
            results.append((name, False))
        print()
    
    # Configurações
    print("-" * 60)
    print("⚙️  Configurando ambiente...")
    print("-" * 60)
    
    create_directories()
    setup_wandb()
    setup_git_lfs()
    create_env_file()
    
    print()
    print("=" * 60)
    print("📊 Resumo do Setup")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 Setup completo! Você está pronto para começar.")
        print()
        print("Próximos passos:")
        print("1. Edite o arquivo .env e configure suas variáveis")
        print("2. Execute: python scripts/download_model.py")
        print("3. Execute: python scripts/collect_data.py")
        print("4. Execute: python training/scripts/train_cpt.py")
    else:
        print("⚠️  Algumas verificações falharam. Corrija os problemas acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
