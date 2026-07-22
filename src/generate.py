import os
import torch
import sentencepiece as spm
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel

# Caminhos dos arquivos
CHECKPOINT_PATH = 'checkpoints/ckpt.pt'
TOKENIZER_PATH = 'data/tokenized/ia_br_bpe.model'

# Como estamos testando no seu PC, vamos manter na CPU
DEVICE = 'cpu'

def main():
    print("\n[1/3] Acordando o Tokenizer Brasileiro (BPE)...")
    try:
        sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    except Exception as e:
        print(f"Erro ao carregar Tokenizer: {e}")
        return

    print(f"[2/3] Carregando o Cérebro da Lloavece ({CHECKPOINT_PATH})...")
    if not os.path.exists(CHECKPOINT_PATH):
        print("Erro: Checkpoint não encontrado. Você precisa rodar o train.py primeiro.")
        return

    # Carrega as memórias salvas no HD
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Recria o modelo exatamente com a configuração ("micro") que estava salva
    config = checkpoint['config']
    model = LloaveceModel(config)
    
    # Injeta o conhecimento (pesos matemáticos) dentro da rede neural
    model.load_state_dict(checkpoint['model'])
    
    # Trava o modelo no modo de Leitura/Geração (desativa o aprendizado/treino)
    model.eval() 
    model.to(DEVICE)

    print(f"[3/3] ✅ Cérebro carregado! (Passo do Treino: {checkpoint.get('iter_num', 'Desconhecido')})")
    print("\n" + "="*50)
    print("🤖 TERMINAL DE INFERÊNCIA - LLOAVECE (MICRO)")
    print("="*50)
    print("Aviso: Ela treinou por apenas 200 passos. O vocabulário dela")
    print("ainda é bem confuso e ela pode gerar 'sopa de letrinhas'.")
    print("Digite 'sair' para encerrar.")
    print("="*50)

    while True:
        try:
            prompt = input("\nVocê: ")
            if prompt.strip().lower() in ['sair', 'exit', 'quit', 'q']:
                break
                
            if not prompt.strip():
                continue

            # Passo A: Transforma o texto em números (Tokens)
            input_ids = sp.encode(prompt)
            x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

            print("Lloavece: ", end="", flush=True)
            
            # Passo B: A Mágica. Pede pra rede prever os próximos 50 tokens
            with torch.no_grad():
                # max_new_tokens diz o tamanho da resposta dela
                y = model.generate(x, max_new_tokens=40)
                
            # Passo C: Transforma a matriz matemática devolta para o Português
            output_ids = y[0].tolist()
            
            # Corta a parte do prompt para exibir apenas a resposta nova
            generated_ids = output_ids[len(input_ids):]
            response_text = sp.decode(generated_ids)
            
            print(response_text)
            
        except KeyboardInterrupt:
            print("\nEncerrando inferência...")
            break

if __name__ == "__main__":
    main()
