import os
import json
import argparse
from src.data.collectors.ptbr_curriculum import export_ptbr_dataset

OUTPUT_DIR = "data/raw"
OUTPUT_PRETRAIN = os.path.join(OUTPUT_DIR, "code_corpus.jsonl")
OUTPUT_SFT = os.path.join(OUTPUT_DIR, "code_sft.jsonl")

def download_code_data(max_samples: int = 5000, ptbr_heavy: bool = True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("=== COLETANDO DATASET LLOAVECE-CODER (PT-BR NATIVO + CÓDIGO) ===")
    print("=" * 65)

    # 1. Injeta o currículo brasileiro aprofundado (PIX, Boleto, CLT, LGPD, PostgreSQL, FastAPI)
    print(f"\n[1/3] Gravando currículo nacional de alta fidelidade (PT-BR)...")
    export_ptbr_dataset(OUTPUT_SFT, OUTPUT_PRETRAIN)
    print(f"[OK] Módulos de regras de negócio nacionais injetados com sucesso!")

    # 2. Puxa dados de instruções em Português do Brasil (Alpaca-PTBR)
    if ptbr_heavy:
        print(f"\n[2/3] Baixando instruções avançadas em Português do Brasil (dominguesm/alpaca-data-pt-br)...")
        try:
            from datasets import load_dataset
            alpaca_pt = load_dataset("dominguesm/alpaca-data-pt-br", split="train", streaming=True)
            pt_count = 0
            max_pt = min(max_samples, 10000)
            with open(OUTPUT_SFT, "a", encoding="utf-8") as f_sft, open(OUTPUT_PRETRAIN, "a", encoding="utf-8") as f_pre:
                for item in alpaca_pt:
                    inst = item.get("instruction", "").strip()
                    inp = item.get("input", "").strip()
                    resp = item.get("output", "").strip()
                    if inp:
                        inst += f"\nContexto: {inp}"
                    if inst and resp:
                        f_sft.write(json.dumps({"prompt": inst, "response": resp}, ensure_ascii=False) + "\n")
                        # Também salva texto para vocabulário do tokenizer
                        f_pre.write(json.dumps({"content": f"{inst}\n{resp}"}, ensure_ascii=False) + "\n")
                        pt_count += 1
                        if pt_count >= max_pt:
                            break
                        if pt_count % 1000 == 0:
                            print(f"Progresso PT-BR: {pt_count}/{max_pt} instruções carregadas...")
            print(f"[OK] {pt_count} instruções em Português do Brasil registradas!")
        except Exception as e:
            print(f"[AVISO] Não foi possível carregar dataset Alpaca-PTBR ({e}).")

    # 3. Puxa dados de código de programação (Python Code Instructions)
    print(f"\n[3/3] Baixando código de programação de alta qualidade...")
    try:
        from datasets import load_dataset
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
        code_count = 0
        with open(OUTPUT_PRETRAIN, "a", encoding="utf-8") as f_pre, open(OUTPUT_SFT, "a", encoding="utf-8") as f_sft:
            for item in ds_code:
                instruction = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                if inp:
                    instruction += f"\nEntrada: {inp}"
                code = item.get("output", "").strip()

                if code and len(code) > 20:
                    f_pre.write(json.dumps({"content": code}, ensure_ascii=False) + "\n")
                    f_sft.write(json.dumps({"prompt": instruction, "response": code}, ensure_ascii=False) + "\n")
                    code_count += 1
                    if code_count >= max_samples:
                        break
                    if code_count % 1000 == 0:
                        print(f"Progresso Código: {code_count}/{max_samples} códigos coletados...")

        print(f"[OK] {code_count} códigos adicionados com sucesso!")
    except Exception as e:
        print(f"[AVISO] Erro na coleta de código: {e}")

    print("\n[OK] Coleta de Português e Código concluída com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000, help="Quantidade de amostras")
    parser.add_argument("--no_ptbr", action="store_true", help="Desabilita carga pesada de PT-BR")
    args = parser.parse_args()

    download_code_data(max_samples=args.samples, ptbr_heavy=not args.no_ptbr)
