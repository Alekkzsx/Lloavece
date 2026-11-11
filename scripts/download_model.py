#!/usr/bin/env python3
"""
Script para download do modelo base Qwen3.8-27B.
Usa huggingface_hub para download eficiente com suporte a resume.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import argparse


def download_model(
    model_name: str = "Qwen/Qwen3.8-27B",
    output_dir: str = "models/base",
    use_auth_token: Optional[str] = None,
    resume_download: bool = True,
) -> None:
    """
    Baixa o modelo do Hugging Face Hub.
    
    Args:
        model_name: Nome do modelo no Hugging Face
        output_dir: Diretório de destino
        use_auth_token: Token de autenticação (opcional)
        resume_download: Se deve resumir download interrompido
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub não instalado. Execute:")
        print("   pip install huggingface_hub")
        sys.exit(1)
    
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_dir / model_name.split('/')[-1].lower()
    
    print("=" * 60)
    print("📥 Download do Modelo Base")
    print("=" * 60)
    print(f"Modelo: {model_name}")
    print(f"Destino: {output_path}")
    print(f"Tamanho estimado: ~55 GB")
    print()
    
    # Verifica espaço em disco
    import shutil
    total, used, free = shutil.disk_usage(output_path.parent)
    free_gb = free / (1024**3)
    
    if free_gb < 60:
        print(f"⚠️  Espaço insuficiente: {free_gb:.1f} GB disponíveis")
        print(f"   Necessário: ~60 GB")
        response = input("Continuar mesmo assim? (y/n): ")
        if response.lower() != 'y':
            print("❌ Download cancelado")
            sys.exit(0)
    
    print("🚀 Iniciando download...")
    print("   (Isso pode levar várias horas dependendo da conexão)")
    print()
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            resume_download=resume_download,
            token=use_auth_token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Ignora formatos alternativos
        )
        
        print()
        print("✅ Download concluído com sucesso!")
        print(f"   Modelo salvo em: {output_path}")
        print()
        
        # Lista arquivos baixados
        files = list(output_path.glob('*'))
        print(f"📁 Arquivos baixados: {len(files)}")
        for f in files[:10]:  # Mostra primeiros 10
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name}: {size_mb:.1f} MB")
        
        if len(files) > 10:
            print(f"   ... e mais {len(files) - 10} arquivos")
        
        # Atualiza .env
        update_env_file(str(output_path))
        
    except Exception as e:
        print()
        print(f"❌ Erro durante o download: {e}")
        print()
        print("Soluções possíveis:")
        print("1. Verifique sua conexão com a internet")
        print("2. Tente novamente (resume_download está habilitado)")
        print("3. Verifique se você tem permissão para acessar o modelo")
        print("4. Se o modelo é privado, configure HF_TOKEN")
        sys.exit(1)


def update_env_file(model_path: str) -> None:
    """Atualiza o arquivo .env com o path do modelo"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    if not env_file.exists():
        print("⚠️  Arquivo .env não encontrado. Pulando atualização.")
        return
    
    # Lê conteúdo atual
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Atualiza MODEL_PATH
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('MODEL_PATH='):
            lines[i] = f'MODEL_PATH={model_path}\n'
            updated = True
            break
    
    if not updated:
        lines.append(f'\nMODEL_PATH={model_path}\n')
    
    # Escreve de volta
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Arquivo .env atualizado com MODEL_PATH={model_path}")


def verify_model(model_path: str) -> bool:
    """Verifica se o modelo foi baixado corretamente"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print()
        print("🔍 Verificando integridade do modelo...")
        
        # Tenta carregar o tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Tokenizer carregado: {len(tokenizer)} tokens")
        
        # Tenta carregar a config (sem carregar pesos para economizar memória)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Config carregada: {config.num_hidden_layers} camadas, {config.hidden_size} hidden dim")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao verificar modelo: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download do modelo base Qwen3.8-27B"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3.8-27B',
        help='Nome do modelo no Hugging Face (default: Qwen/Qwen3.8-27B)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/base',
        help='Diretório de destino (default: models/base)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Token de autenticação do Hugging Face (opcional)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Não resumir download interrompido'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verificar modelo após download'
    )
    
    args = parser.parse_args()
    
    # Download
    download_model(
        model_name=args.model,
        output_dir=args.output_dir,
        use_auth_token=args.token,
        resume_download=not args.no_resume,
    )
    
    # Verificação opcional
    if args.verify:
        project_root = Path(__file__).parent.parent
        model_path = project_root / args.output_dir / args.model.split('/')[-1].lower()
        
        if verify_model(str(model_path)):
            print()
            print("🎉 Modelo pronto para uso!")
        else:
            print()
            print("⚠️  Verificação falhou. Tente baixar novamente.")
            sys.exit(1)


if __name__ == "__main__":
    main()
