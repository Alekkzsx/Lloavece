import os
import json
import argparse
from typing import Optional

OUTPUT_DIR = "data/raw"
OUTPUT_PRETRAIN = os.path.join(OUTPUT_DIR, "code_corpus.jsonl")
OUTPUT_SFT = os.path.join(OUTPUT_DIR, "code_sft.jsonl")

# Exemplos nativos de alta fidelidade do ecossistema e regras de negócio brasileiras
BRAZILIAN_ECOSYSTEM_EXAMPLES = [
    {
        "prompt": "Como criar uma função em Python para validar se uma string é um CPF brasileiro válido, calculando os dígitos verificadores?",
        "response": '''def validar_cpf(cpf: str) -> bool:
    """Valida um CPF brasileiro segundo o algoritmo oficial da Receita Federal."""
    # Remove caracteres não numéricos
    numeros = [int(char) for char in cpf if char.isdigit()]
    
    # CPF deve conter exatamente 11 dígitos e não pode ter todos os dígitos iguais
    if len(numeros) != 11 or len(set(numeros)) == 1:
        return False
        
    # Validação do primeiro dígito verificador
    soma_1 = sum(a * b for a, b in zip(numeros[:9], range(10, 1, -1)))
    digito_1 = (soma_1 * 10 % 11) % 10
    if digito_1 != numeros[9]:
        return False
        
    # Validação do segundo dígito verificador
    soma_2 = sum(a * b for a, b in zip(numeros[:10], range(11, 1, -1)))
    digito_2 = (soma_2 * 10 % 11) % 10
    return digito_2 == numeros[10]'''
    },
    {
        "prompt": "Crie um endpoint FastAPI assíncrono para gerar uma cobrança PIX imediata (QR Code Estático) seguindo o padrão EMV do Banco Central do Brasil.",
        "response": '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import crcmod

app = FastAPI(title="Lloavece PIX Engine")

class PixRequest(BaseModel):
    chave: str
    nome_beneficiario: str
    cidade: str
    valor: float
    identificador: str = "***"

def calcular_crc16(payload: str) -> str:
    crc16 = crcmod.mkCrcFun(0x11021, initCrc=0xFFFF, rev=False, xorOut=0x0000)
    resultado = hex(crc16(payload.encode("utf-8"))).upper()
    return resultado[2:].zfill(4)

@app.post("/pix/gerar-qrcode-estatico")
async def gerar_qrcode_pix(dados: PixRequest):
    """Gera o payload oficial EMV para PIX Estático (Bacen)."""
    nome = dados.nome_beneficiario[:25].upper()
    cidade = dados.cidade[:15].upper()
    valor_str = f"{dados.valor:.2f}"
    
    payload = (
        "000201"                                      # Formato do Payload
        "26360014br.gov.bcb.pix"                     # Merchant Account Information
        f"01{len(dados.chave):02d}{dados.chave}"
        "52040000"                                    # Merchant Category Code
        "5303986"                                     # Moeda BRL (986)
        f"54{len(valor_str):02d}{valor_str}"          # Valor
        "5802BR"                                      # País
        f"59{len(nome):02d}{nome}"                    # Nome do Beneficiário
        f"60{len(cidade):02d}{cidade}"                # Cidade
        f"62{len(dados.identificador) + 4:02d}05{len(dados.identificador):02d}{dados.identificador}"
        "6304"                                        # Início do CRC16
    )
    
    checksum = calcular_crc16(payload)
    payload_completo = payload + checksum
    
    return {
        "status": "sucesso",
        "pix_copia_e_cola": payload_completo
    }'''
    }
]

def download_code_data(max_samples: int = 50000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("=== COLETANDO DATASET DE CÓDIGO (PRE-TREINO E SFT) ===")
    print("=" * 60)

    # 1. Salva exemplos nativos brasileiros em SFT
    print(f"\n[1/3] Injetando regras de negócio e integrações brasileiras (PIX, CPF, FastAPI)...")
    with open(OUTPUT_SFT, "w", encoding="utf-8") as f:
        for ex in BRAZILIAN_ECOSYSTEM_EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[OK] {len(BRAZILIAN_ECOSYSTEM_EXAMPLES)} exemplos base brasileiros gravados.")

    # 2. Puxa dados reais de código via Hugging Face datasets
    try:
        from datasets import load_dataset
        print(f"\n[2/3] Baixando corpus de código de alta fidelidade...")
        dataset = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            streaming=True
        )

        count = 0
        with open(OUTPUT_PRETRAIN, "w", encoding="utf-8") as f:
            for item in dataset:
                content = item.get("content", "").strip()
                if content and len(content) > 50:
                    record = {"content": content, "lang": "python"}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= max_samples:
                        break
                    if count % 5000 == 0:
                        print(f"Progresso: {count}/{max_samples} códigos coletados...")

        print(f"[OK] {count} arquivos de código gravados em: {OUTPUT_PRETRAIN}")

    except ImportError:
        print("[AVISO] Pacote 'datasets' não encontrado. Rode: pip install datasets")
        print("Gravando dados locais de bootstrap em formato JSONL...")
        with open(OUTPUT_PRETRAIN, "w", encoding="utf-8") as f:
            for ex in BRAZILIAN_ECOSYSTEM_EXAMPLES:
                f.write(json.dumps({"content": ex["response"]}, ensure_ascii=False) + "\n")

    print("\n[3/3] Coleta concluída com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Quantidade de amostras de código")
    args = parser.parse_args()
    download_code_data(max_samples=args.samples)
