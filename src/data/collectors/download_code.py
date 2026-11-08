import os
import json
import argparse

OUTPUT_DIR = "data/raw"
OUTPUT_PRETRAIN = os.path.join(OUTPUT_DIR, "code_corpus.jsonl")
OUTPUT_SFT = os.path.join(OUTPUT_DIR, "code_sft.jsonl")

# Exemplos nativos de alta fidelidade do ecossistema e regras de negócio brasileiras
BRAZILIAN_ECOSYSTEM_EXAMPLES = [
    {
        "prompt": "Como criar uma função em Python para validar se uma string é um CPF brasileiro válido, calculando os dígitos verificadores?",
        "response": '''def validar_cpf(cpf: str) -> bool:
    """Valida um CPF brasileiro segundo o algoritmo oficial da Receita Federal."""
    numeros = [int(char) for char in cpf if char.isdigit()]
    
    if len(numeros) != 11 or len(set(numeros)) == 1:
        return False
        
    soma_1 = sum(a * b for a, b in zip(numeros[:9], range(10, 1, -1)))
    digito_1 = (soma_1 * 10 % 11) % 10
    if digito_1 != numeros[9]:
        return False
        
    soma_2 = sum(a * b for a, b in zip(numeros[:10], range(11, 1, -1)))
    digito_2 = (soma_2 * 10 % 11) % 10
    return digito_2 == numeros[10]'''
    },
    {
        "prompt": "Implemente uma função em Python para validar CNPJ brasileiro de acordo com o padrão da Receita Federal.",
        "response": '''def validar_cnpj(cnpj: str) -> bool:
    """Valida um CNPJ brasileiro com os dois dígitos verificadores."""
    numeros = [int(c) for c in cnpj if c.isdigit()]
    if len(numeros) != 14 or len(set(numeros)) == 1:
        return False
        
    pesos_1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    soma_1 = sum(a * b for a, b in zip(numeros[:12], pesos_1))
    resto_1 = soma_1 % 11
    d1 = 0 if resto_1 < 2 else 11 - resto_1
    if numeros[12] != d1:
        return False
        
    pesos_2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    soma_2 = sum(a * b for a, b in zip(numeros[:13], pesos_2))
    resto_2 = soma_2 % 11
    d2 = 0 if resto_2 < 2 else 11 - resto_2
    return numeros[13] == d2'''
    },
    {
        "prompt": "Crie um endpoint FastAPI assíncrono para gerar uma cobrança PIX imediata (QR Code Estático) seguindo o padrão EMV do Banco Central do Brasil.",
        "response": '''from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Lloavece PIX Engine")

class PixRequest(BaseModel):
    chave: str
    nome_beneficiario: str
    cidade: str
    valor: float
    identificador: str = "***"

def calcular_crc16(payload: str) -> str:
    crc = 0xFFFF
    for byte in payload.encode("utf-8"):
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return hex(crc)[2:].upper().zfill(4)

@app.post("/pix/gerar-qrcode-estatico")
async def gerar_qrcode_pix(dados: PixRequest):
    """Gera o payload oficial EMV para PIX Estático (Bacen)."""
    nome = dados.nome_beneficiario[:25].upper()
    cidade = dados.cidade[:15].upper()
    valor_str = f"{dados.valor:.2f}"
    
    payload = (
        "000201"
        "26360014br.gov.bcb.pix"
        f"01{len(dados.chave):02d}{dados.chave}"
        "52040000"
        "5303986"
        f"54{len(valor_str):02d}{valor_str}"
        "5802BR"
        f"59{len(nome):02d}{nome}"
        f"60{len(cidade):02d}{cidade}"
        f"62{len(dados.identificador) + 4:02d}05{len(dados.identificador):02d}{dados.identificador}"
        "6304"
    )
    return {"pix_copia_e_cola": payload + calcular_crc16(payload)}'''
    },
    {
        "prompt": "Escreva uma função async para consultar o CEP brasileiro na API ViaCEP com timeout e tratamento de erros.",
        "response": '''import httpx
from typing import Optional, Dict

async def consultar_cep(cep: str) -> Optional[Dict[str, str]]:
    """Consulta dados de endereço brasileiro a partir do CEP via HTTP assíncrono."""
    cep_limpo = "".join(filter(str.isdigit, cep))
    if len(cep_limpo) != 8:
        raise ValueError(f"CEP inválido: {cep}. Deve conter exatamente 8 dígitos.")
        
    url = f"https://viacep.com.br/ws/{cep_limpo}/json/"
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get("erro"):
                return None
            return {
                "logradouro": data.get("logradouro", ""),
                "bairro": data.get("bairro", ""),
                "cidade": data.get("localidade", ""),
                "uf": data.get("uf", ""),
                "cep": data.get("cep", "")
            }
        except httpx.HTTPError as e:
            return None'''
    }
]

def download_code_data(max_samples: int = 500):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("=== COLETANDO DATASET DE CÓDIGO (PRE-TREINO E SFT) ===")
    print("=" * 60)

    # 1. Grava os exemplos de alta fidelidade nacionais em SFT e Pretrain
    print(f"\n[1/3] Gravando regras de negócio nacionais (PIX, CPF, CNPJ, ViaCEP)...")
    with open(OUTPUT_SFT, "w", encoding="utf-8") as f_sft, open(OUTPUT_PRETRAIN, "w", encoding="utf-8") as f_pre:
        for ex in BRAZILIAN_ECOSYSTEM_EXAMPLES:
            f_sft.write(json.dumps(ex, ensure_ascii=False) + "\n")
            f_pre.write(json.dumps({"content": ex["response"]}, ensure_ascii=False) + "\n")
    print(f"[OK] {len(BRAZILIAN_ECOSYSTEM_EXAMPLES)} exemplos nacionais registrados.")

    # 2. Puxa dados reais de código aberto via dataset ungated
    print(f"\n[2/3] Baixando instruções e código Python de alta fidelidade...")
    try:
        from datasets import load_dataset
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
        count = 0
        with open(OUTPUT_PRETRAIN, "a", encoding="utf-8") as f_pre, open(OUTPUT_SFT, "a", encoding="utf-8") as f_sft:
            for item in ds:
                instruction = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                if inp:
                    instruction += f"\nEntrada: {inp}"
                code = item.get("output", "").strip()

                if code and len(code) > 20:
                    f_pre.write(json.dumps({"content": code}, ensure_ascii=False) + "\n")
                    f_sft.write(json.dumps({"prompt": instruction, "response": code}, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= max_samples:
                        break
                    if count % 100 == 0:
                        print(f"Progresso: {count}/{max_samples} códigos coletados...")

        print(f"[OK] {count} amostras adicionadas a {OUTPUT_PRETRAIN} e {OUTPUT_SFT}")

    except Exception as e:
        print(f"[AVISO] Não foi possível carregar dataset remoto ({e}). Usando base local de bootstrap.")

    print("\n[3/3] Coleta concluída com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500, help="Quantidade de amostras de código")
    args = parser.parse_args()
    download_code_data(max_samples=args.samples)
