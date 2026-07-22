import os
import bz2
import json
import urllib.request
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re

# Dump oficial e mais recente apenas de artigos da Wikipedia em português
WIKI_DUMP_URL = "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2"
RAW_DIR = "data/raw/wikipedia"
OUTPUT_FILE = os.path.join(RAW_DIR, "ptwiki_articles.jsonl")
DUMP_FILE = os.path.join(RAW_DIR, "ptwiki-latest-pages-articles.xml.bz2")

def download_dump():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        
    if os.path.exists(DUMP_FILE):
        print(f"Dump já existe em {DUMP_FILE}. Pulando a etapa de download.")
        return

    print(f"Baixando Wikipedia PT-BR de {WIKI_DUMP_URL}...")
    print("⚠️ ATENÇÃO: Este arquivo tem cerca de 2.5 GB! Dependendo da internet, pode levar de 15 a 45 minutos.")
    
    # Precisamos disfarçar o nosso bot com um User-Agent, senão a Wikipedia bloqueia (Erro 403)
    req = urllib.request.Request(WIKI_DUMP_URL, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 IA-Brasileira-Bot'})
    
    with urllib.request.urlopen(req) as response, open(DUMP_FILE, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Baixando ptwiki") as t:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                t.update(len(chunk))
    print("Download concluído!")

def clean_wikitext(text):
    """ Limpeza básica de marcações do Wikitext para sobrar apenas o texto de alta qualidade """
    if not text: return ""
    
    # Remove as tags de referências bibliográficas
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    # Remove as tabelas feias em formato de texto da Wikipedia
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove os links internos, mas MANTÉM a palavra âncora (Ex: [[São Paulo|SP]] -> SP)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove marcações de negrito e itálico
    text = re.sub(r"'''|''", "", text)
    # Remove os "templates" de chaves {{...}} que a wikipedia usa para infoboxes
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove os sublinhados de seções == Seção ==
    text = re.sub(r'==+', '', text)
    
    # Limpa espaços em branco vazios e linhas múltiplas
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def strip_ns(tag):
    # O XML da Wikipedia tem namespaces chatos, essa função os remove para facilitar a busca
    return tag.split('}')[-1] if '}' in tag else tag

def extract_and_save():
    if not os.path.exists(DUMP_FILE):
        print(f"Erro: Arquivo {DUMP_FILE} não encontrado.")
        return
        
    print(f"Extraindo XML, limpando sujeiras e salvando no formato ideal (ISSO VAI DEMORAR VÁRIOS MINUTOS)...")
    
    # ET.iterparse lê o arquivo enquanto ele descompacta (stream), evitando que a RAM do PC exploda
    context = ET.iterparse(bz2.BZ2File(DUMP_FILE), events=("start", "end"))
    
    article_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        title = ""
        text = ""
        
        for event, elem in context:
            tag = strip_ns(elem.tag)
            
            if event == 'end':
                if tag == 'page':
                    # Processa apenas artigos que não são mero redirecionamento
                    if text and not text.startswith("#REDIRECT"):
                        clean_text = clean_wikitext(text)
                        
                        # Salva apenas se o artigo tiver um tamanho útil (> 200 letras)
                        if len(clean_text) > 200: 
                            doc = {"title": title, "text": clean_text}
                            # Salvamos em .jsonl (JSON Lines) - o formato padrão da indústria para datasets
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                            article_count += 1
                            
                            if article_count % 10000 == 0:
                                print(f"  -> {article_count} artigos processados e salvos...")
                    
                    # FUNDAMENTAL: Limpa a árvore do XML da memória a cada artigo processado!
                    elem.clear()
                    title = ""
                    text = ""
                elif tag == 'title':
                    title = elem.text
                elif tag == 'text':
                    text = elem.text

    print(f"✅ SUCESSO ABSOLUTO! {article_count} artigos de altíssima qualidade da Wikipedia salvos em {OUTPUT_FILE}.")

if __name__ == "__main__":
    print("==============================================")
    print("🇧🇷 INICIANDO PIPELINE DE DADOS WIKIPEDIA PT-BR")
    print("==============================================\n")
    download_dump()
    extract_and_save()
