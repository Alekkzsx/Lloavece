import os
import json
import argparse
from src.data.collectors.triad_curriculum import export_triad_dataset

OUTPUT_DIR = "data/raw"
OUTPUT_PRETRAIN = os.path.join(OUTPUT_DIR, "code_corpus.jsonl")
OUTPUT_SFT = os.path.join(OUTPUT_DIR, "code_sft.jsonl")

def download_code_data(max_samples: int = 5000, include_math: bool = True, include_ptbr: bool = True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("=== COLETANDO A TRÍADE LLOAVECE: MATEMÁTICA + INGLÊS + PT-BR ===")
    print("=" * 65)

    # 1. Injeta o Currículo Formal da Tríade (Matemática, Álgebra Linear, Grafos, Criptografia)
    print(f"\n[1/4] Injetando o Currículo Formal da Tríade (Matemática Pura, Big-O, Grafos)...")
    export_triad_dataset(OUTPUT_SFT, OUTPUT_PRETRAIN)
    print(f"[OK] Módulos formais de matemática e algoritmos registrados!")

    # 2. Puxa Raciocínio Matemático Passo a Passo (GSM8K)
    if include_math:
        print(f"\n[2/4] Baixando problemas de raciocínio matemático (GSM8K / Math Reasoning)...")
        try:
            from datasets import load_dataset
            gsm = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
            math_count = 0
            max_math = min(max_samples, 5000)
            with open(OUTPUT_SFT, "a", encoding="utf-8") as f_sft, open(OUTPUT_PRETRAIN, "a", encoding="utf-8") as f_pre:
                for item in gsm:
                    q = item.get("question", "").strip()
                    a = item.get("answer", "").strip()
                    if q and a:
                        prompt_pt = f"Resolva este problema matemático passo a passo:\n{q}"
                        f_sft.write(json.dumps({"prompt": prompt_pt, "response": a}, ensure_ascii=False) + "\n")
                        f_pre.write(json.dumps({"content": f"{q}\nSolution:\n{a}"}, ensure_ascii=False) + "\n")
                        math_count += 1
                        if math_count >= max_math:
                            break
                        if math_count % 1000 == 0:
                            print(f"Progresso Matemática: {math_count}/{max_math} problemas carregados...")
            print(f"[OK] {math_count} problemas matemáticos de raciocínio lógico injetados!")
        except Exception as e:
            print(f"[AVISO] Dataset GSM8K não carregado ({e}). Continuando com base primária.")

    # 3. Puxa Código de Programação e Ciência da Computação em Inglês
    print(f"\n[3/4] Baixando código de algoritmos e estruturas de dados em Inglês...")
    try:
        from datasets import load_dataset
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
        code_count = 0
        with open(OUTPUT_PRETRAIN, "a", encoding="utf-8") as f_pre, open(OUTPUT_SFT, "a", encoding="utf-8") as f_sft:
            for item in ds_code:
                instruction = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                if inp:
                    instruction += f"\nInput: {inp}"
                code = item.get("output", "").strip()

                if code and len(code) > 20:
                    f_pre.write(json.dumps({"content": code}, ensure_ascii=False) + "\n")
                    f_sft.write(json.dumps({"prompt": instruction, "response": code}, ensure_ascii=False) + "\n")
                    code_count += 1
                    if code_count >= max_samples:
                        break
                    if code_count % 1000 == 0:
                        print(f"Progresso Código: {code_count}/{max_samples} códigos coletados...")
        print(f"[OK] {code_count} implementações de código registradas!")
    except Exception as e:
        print(f"[AVISO] Erro na coleta de código: {e}")

    # 4. Puxa Instruções e Raciocínio Fluente em Português do Brasil
    if include_ptbr:
        print(f"\n[4/4] Baixando raciocínio explicativo e diálogo em Português do Brasil...")
        try:
            from datasets import load_dataset
            alpaca_pt = load_dataset("dominguesm/alpaca-data-pt-br", split="train", streaming=True)
            pt_count = 0
            max_pt = min(max_samples, 5000)
            with open(OUTPUT_SFT, "a", encoding="utf-8") as f_sft, open(OUTPUT_PRETRAIN, "a", encoding="utf-8") as f_pre:
                for item in alpaca_pt:
                    inst = item.get("instruction", "").strip()
                    inp = item.get("input", "").strip()
                    resp = item.get("output", "").strip()
                    if inp:
                        inst += f"\nContexto: {inp}"
                    if inst and resp:
                        f_sft.write(json.dumps({"prompt": inst, "response": resp}, ensure_ascii=False) + "\n")
                        f_pre.write(json.dumps({"content": f"{inst}\n{resp}"}, ensure_ascii=False) + "\n")
                        pt_count += 1
                        if pt_count >= max_pt:
                            break
                        if pt_count % 1000 == 0:
                            print(f"Progresso PT-BR: {pt_count}/{max_pt} instruções carregadas...")
            print(f"[OK] {pt_count} instruções em Português do Brasil registradas!")
        except Exception as e:
            print(f"[AVISO] Erro na coleta PT-BR: {e}")

    print("\n[OK] Tríade de Dados (Matemática + Inglês + PT-BR) pronta com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    download_code_data(max_samples=args.samples)
