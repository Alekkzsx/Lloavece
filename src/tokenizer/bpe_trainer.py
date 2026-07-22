import os
import json
import sentencepiece as spm

# Configurações do plano
VOCAB_SIZE = 16384
DATA_FILE = "data/raw/wikipedia/ptwiki_articles.jsonl"
OUTPUT_DIR = "data/tokenized"
MODEL_PREFIX = os.path.join(OUTPUT_DIR, "ia_br_bpe")

# Tokens Especiais obrigatórios ditados no seu PLAN.md
# Nota: bos, eos, pad e unk são tratados separadamente pela engine do SentencePiece
SPECIAL_TOKENS = [
    "<|sep|>",
    "<|sys|>",
    "<|user|>",
    "<|assistant|>"
]
# + 8 tokens extras para extensibilidade futura (conforme plano)
SPECIAL_TOKENS += [f"<|ext_{i}|>" for i in range(1, 9)]

def text_iterator():
    """ 
    Gerador que lê o JSONL linha por linha para alimentar o SentencePiece
    sem carregar o arquivo inteiro de 1.5GB na memória (eficiência total).
    """
    print(f"Lendo dados de {DATA_FILE}...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Para o treinamento do Tokenizer, não precisamos da Wikipedia inteira.
            # Os primeiros 300.000 artigos (umas centenas de MBs de texto) já 
            # contém praticamente TODAS as palavras, gírias e sílabas da língua portuguesa.
            if i >= 300000:
                break
            
            doc = json.loads(line)
            # Retorna apenas o texto do artigo limpo
            yield doc['text']

def train_tokenizer():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Iniciando treinamento do Tokenizer BPE (Vocab: {VOCAB_SIZE})...")
    print("Isso pode levar ALGUNS MINUTOS (ele vai rodar milhões de cálculos para descobrir os sufixos mais comuns do Brasil!)")

    # String de tokens especiais formatada para o SentencePiece
    user_defined_symbols = ",".join(SPECIAL_TOKENS)

    # O SentencePiece é brutalmente rápido porque é escrito em C++
    spm.SentencePieceTrainer.train(
        sentence_iterator=text_iterator(),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type='bpe',          # O algoritmo mágico BPE!
        character_coverage=0.9995, # Cobre quase todos os caracteres que existem
        user_defined_symbols=user_defined_symbols,
        pad_id=2,                  # Forçando as IDs exatas do seu plano
        unk_id=3,
        bos_id=0,
        eos_id=1,
        pad_piece="<|pad|>",
        unk_piece="<|unk|>",
        bos_piece="<|bos|>",
        eos_piece="<|eos|>",
        byte_fallback=True,        # Truque: se vir um emoji estranho, converte pra byte em vez de Unknown
        max_sentence_length=20000, 
        num_threads=os.cpu_count() # Usa todos os núcleos do processador!
    )

    print(f"✅ Tokenizer BPE treinado com sucesso!")
    print(f"Arquivos salvos em:")
    print(f" - {MODEL_PREFIX}.model (Binário de predição do C++)")
    print(f" - {MODEL_PREFIX}.vocab (A lista de vocabulário visível)")

def test_tokenizer():
    """ Pequeno teste para ver o quão bom ficou o nosso Tokenizer PT-BR """
    print("\n--- Teste Rápido do nosso Novo Tokenizer ---")
    sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")
    
    frase = "A inteligência artificial do Brasil fala português fluentemente! Configuração e maçã."
    
    # Encode transforma a string -> lista de IDs de tokens (números)
    ids = sp.encode(frase)
    # Mostra também como string para vermos os "pedacinhos" (chunks) que ele criou
    pieces = sp.encode(frase, out_type=str)
    
    print(f"\nFrase original: {frase}")
    print(f"Tokens gerados (IDs): {ids}")
    print(f"Tokens divididos: {pieces}")
    print(f"Quantidade de tokens gastos: {len(ids)}")
    print("\nVeja os pedaços acima! Ele provavelmente agrupou palavras inteiras ou sílabas grandes como um token só.")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"ERRO: Não encontrei {DATA_FILE}. Rode o coletor da Wikipedia primeiro.")
    else:
        train_tokenizer()
        test_tokenizer()
