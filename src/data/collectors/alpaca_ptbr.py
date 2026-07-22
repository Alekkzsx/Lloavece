import os
import json

# Vamos usar a biblioteca oficial do HuggingFace para evitar links quebrados!
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "alpaca_ptbr.jsonl")

def download_alpaca():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("🔧 Instalando a biblioteca 'datasets' da HuggingFace automaticamente...")
    os.system("pip install datasets -q")
    from datasets import load_dataset
        
    print(f"📥 Baixando dataset de instruções Alpaca-PTBR direto dos servidores da HuggingFace...")
    try:
        # Puxando o arquivo oficial mantido pela comunidade
        dataset = load_dataset("dominguesm/alpaca-data-pt-br", split="train")
    except Exception as e:
        print(f"❌ Erro ao baixar a base de dados: {e}")
        return

    print(f"📊 Total de exemplos coletados: {len(dataset)}")
    
    print("💾 Convertendo e salvando em formato JSONL...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset:
            prompt = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            if inp:
                prompt += f"\n\nContexto anexo:\n{inp}"
                
            response_text = item.get("output", "").strip()
            
            if prompt and response_text:
                record = {
                    "prompt": prompt,
                    "response": response_text
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
    tamanho_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"✅ Arquivo salvo em: {OUTPUT_FILE} ({tamanho_mb:.2f} MB)")

if __name__ == "__main__":
    download_alpaca()
