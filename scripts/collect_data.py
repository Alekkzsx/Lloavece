#!/usr/bin/env python3
"""
Script para coleta de dados de código para treinamento do Lloavece Kode.
Coleta dados de múltiplas fontes: GitHub, Stack Overflow, documentação, etc.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import argparse

try:
    import pandas as pd
except ImportError:
    print("❌ pandas não instalado. Execute: pip install pandas")
    exit(1)


class CodeDataCollector:
    """Coletor de dados de código de múltiplas fontes"""
    
    def __init__(self, output_dir: str = "data/raw", github_token: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        
        # Estatísticas
        self.stats = {
            'repositories': 0,
            'files': 0,
            'tokens': 0,
            'errors': 0,
        }
    
    def collect_github_repos(
        self,
        languages: List[str] = None,
        min_stars: int = 100,
        max_repos: int = 1000,
    ) -> List[Dict]:
        """
        Coleta repositórios do GitHub.
        
        Args:
            languages: Lista de linguagens (default: Python, JavaScript, TypeScript)
            min_stars: Número mínimo de estrelas
            max_repos: Número máximo de repositórios
        """
        if languages is None:
            languages = ['Python', 'JavaScript', 'TypeScript', 'Java', 'Go', 'Rust']
        
        print(f"🔍 Buscando repositórios no GitHub...")
        print(f"   Linguagens: {', '.join(languages)}")
        print(f"   Min stars: {min_stars}")
        print(f"   Max repos: {max_repos}")
        print()
        
        repos = []
        
        for language in languages:
            query = f"language:{language} stars:>={min_stars}"
            page = 1
            
            while len(repos) < max_repos:
                try:
                    # GitHub API search
                    headers = {}
                    if self.github_token:
                        headers['Authorization'] = f'token {self.github_token}'
                    
                    response = requests.get(
                        'https://api.github.com/search/repositories',
                        params={
                            'q': query,
                            'sort': 'stars',
                            'order': 'desc',
                            'per_page': 100,
                            'page': page,
                        },
                        headers=headers,
                        timeout=30,
                    )
                    
                    if response.status_code == 403:
                        print("⚠️  Rate limit atingido. Aguardando 60 segundos...")
                        time.sleep(60)
                        continue
                    
                    if response.status_code != 200:
                        print(f"⚠️  Erro na API: {response.status_code}")
                        break
                    
                    data = response.json()
                    
                    if not data['items']:
                        break
                    
                    for repo in data['items']:
                        repos.append({
                            'name': repo['full_name'],
                            'language': repo['language'],
                            'stars': repo['stargazers_count'],
                            'url': repo['html_url'],
                            'clone_url': repo['clone_url'],
                            'description': repo.get('description', ''),
                        })
                        
                        if len(repos) >= max_repos:
                            break
                    
                    page += 1
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"⚠️  Erro ao buscar {language}: {e}")
                    self.stats['errors'] += 1
                    break
        
        print(f"✅ Coletados {len(repos)} repositórios")
        self.stats['repositories'] = len(repos)
        
        # Salva lista de repos
        repos_file = self.output_dir / 'github_repos.json'
        with open(repos_file, 'w') as f:
            json.dump(repos, f, indent=2)
        
        print(f"💾 Lista salva em: {repos_file}")
        
        return repos
    
    def collect_stackoverflow(
        self,
        tags: List[str] = None,
        min_score: int = 10,
        max_questions: int = 10000,
    ) -> List[Dict]:
        """
        Coleta perguntas e respostas do Stack Overflow.
        
        Args:
            tags: Lista de tags (default: python, javascript, etc)
            min_score: Score mínimo da resposta
            max_questions: Número máximo de perguntas
        """
        if tags is None:
            tags = ['python', 'javascript', 'typescript', 'java', 'react', 'node.js']
        
        print(f"🔍 Buscando no Stack Overflow...")
        print(f"   Tags: {', '.join(tags)}")
        print(f"   Min score: {min_score}")
        print(f"   Max questions: {max_questions}")
        print()
        
        questions = []
        
        for tag in tags:
            page = 1
            
            while len(questions) < max_questions:
                try:
                    response = requests.get(
                        'https://api.stackexchange.com/2.3/questions',
                        params={
                            'order': 'desc',
                            'sort': 'votes',
                            'tagged': tag,
                            'site': 'stackoverflow',
                            'filter': 'withbody',
                            'pagesize': 100,
                            'page': page,
                        },
                        timeout=30,
                    )
                    
                    if response.status_code != 200:
                        print(f"⚠️  Erro na API: {response.status_code}")
                        break
                    
                    data = response.json()
                    
                    if not data['items']:
                        break
                    
                    for question in data['items']:
                        # Busca respostas
                        answers_response = requests.get(
                            f"https://api.stackexchange.com/2.3/questions/{question['question_id']}/answers",
                            params={
                                'order': 'desc',
                                'sort': 'votes',
                                'site': 'stackoverflow',
                                'filter': 'withbody',
                            },
                            timeout=30,
                        )
                        
                        if answers_response.status_code == 200:
                            answers_data = answers_response.json()
                            
                            for answer in answers_data['items']:
                                if answer['score'] >= min_score:
                                    questions.append({
                                        'question_id': question['question_id'],
                                        'question_title': question['title'],
                                        'question_body': question['body'],
                                        'question_tags': question.get('tags', []),
                                        'answer_id': answer['answer_id'],
                                        'answer_body': answer['body'],
                                        'answer_score': answer['score'],
                                        'is_accepted': answer.get('is_accepted', False),
                                    })
                                    
                                    if len(questions) >= max_questions:
                                        break
                        
                        # Rate limiting
                        time.sleep(0.1)
                    
                    page += 1
                    
                except Exception as e:
                    print(f"⚠️  Erro ao buscar {tag}: {e}")
                    self.stats['errors'] += 1
                    break
        
        print(f"✅ Coletadas {len(questions)} perguntas com respostas")
        self.stats['files'] = len(questions)
        
        # Salva dados
        so_file = self.output_dir / 'stackoverflow.jsonl'
        with open(so_file, 'w') as f:
            for question in questions:
                f.write(json.dumps(question) + '\n')
        
        print(f"💾 Dados salvos em: {so_file}")
        
        return questions
    
    def collect_brazilian_repos(self, max_repos: int = 500) -> List[Dict]:
        """
        Coleta repositórios brasileiros de qualidade.
        
        Args:
            max_repos: Número máximo de repositórios
        """
        print(f"🔍 Buscando repositórios brasileiros...")
        print(f"   Max repos: {max_repos}")
        print()
        
        # Lista de organizações/usuarios brasileiros conhecidos
        brazilian_repos = [
            # Adicione aqui repos brasileiros de qualidade
            # Exemplo: 'rocketseat/unform', 'die3go/reactjs-nlw', etc
        ]
        
        repos = []
        
        # Por enquanto, apenas busca repos com "brasil" ou "brazil" no nome
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        for query in ['brasil', 'brazil', 'brasileiro']:
            try:
                response = requests.get(
                    'https://api.github.com/search/repositories',
                    params={
                        'q': f'{query} language:python stars:>=50',
                        'sort': 'stars',
                        'order': 'desc',
                        'per_page': 100,
                    },
                    headers=headers,
                    timeout=30,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for repo in data['items']:
                        repos.append({
                            'name': repo['full_name'],
                            'language': repo['language'],
                            'stars': repo['stargazers_count'],
                            'url': repo['html_url'],
                            'description': repo.get('description', ''),
                            'source': 'brazilian',
                        })
                        
                        if len(repos) >= max_repos:
                            break
                
                time.sleep(2)
                
            except Exception as e:
                print(f"⚠️  Erro ao buscar repos brasileiros: {e}")
                self.stats['errors'] += 1
        
        print(f"✅ Coletados {len(repos)} repositórios brasileiros")
        
        # Salva lista
        repos_file = self.output_dir / 'brazilian_repos.json'
        with open(repos_file, 'w') as f:
            json.dump(repos, f, indent=2)
        
        print(f"💾 Lista salva em: {repos_file}")
        
        return repos
    
    def download_repo_files(
        self,
        repos_file: str = None,
        max_files_per_repo: int = 50,
        extensions: List[str] = None,
    ) -> int:
        """
        Baixa arquivos de código dos repositórios coletados.
        
        Args:
            repos_file: Arquivo JSON com lista de repositórios
            max_files_per_repo: Máximo de arquivos por repositório
            extensions: Extensões de arquivo para baixar
        """
        if extensions is None:
            extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
                '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
                '.sh', '.bash', '.md', '.txt'
            ]
        
        if repos_file is None:
            repos_file = self.output_dir / 'github_repos.json'
        
        repos_path = Path(repos_file)
        if not repos_path.exists():
            print(f"❌ Arquivo de repositórios não encontrado: {repos_path}")
            return 0
        
        with open(repos_path, 'r') as f:
            repos = json.load(f)
        
        print(f"📥 Baixando arquivos de {len(repos)} repositórios...")
        print(f"   Max arquivos/repo: {max_files_per_repo}")
        print(f"   Extensões: {', '.join(extensions)}")
        print()
        
        all_files = []
        files_dir = self.output_dir / 'files'
        files_dir.mkdir(exist_ok=True)
        
        for i, repo in enumerate(repos):
            if i % 10 == 0:
                print(f"   Processando {i+1}/{len(repos)}: {repo['name']}")
            
            try:
                repo_name = repo['name']
                owner, name = repo_name.split('/')
                
                # Lista arquivos do repositório via API
                headers = {}
                if self.github_token:
                    headers['Authorization'] = f'token {self.github_token}'
                
                # Busca árvore do repositório (último commit da branch principal)
                # Tenta main primeiro, depois master
                tree_url = None
                for branch in ['main', 'master']:
                    try:
                        ref_resp = requests.get(
                            f'https://api.github.com/repos/{owner}/{name}/git/refs/heads/{branch}',
                            headers=headers,
                            timeout=30,
                        )
                        if ref_resp.status_code == 200:
                            sha = ref_resp.json()['object']['sha']
                            tree_url = f'https://api.github.com/repos/{owner}/{name}/git/trees/{sha}?recursive=1'
                            break
                    except:
                        continue
                
                if not tree_url:
                    continue
                
                tree_resp = requests.get(tree_url, headers=headers, timeout=30)
                if tree_resp.status_code != 200:
                    continue
                
                tree_data = tree_resp.json()
                
                # Filtra arquivos por extensão
                files_to_download = [
                    item for item in tree_data.get('tree', [])
                    if item['type'] == 'blob' and any(item['path'].endswith(ext) for ext in extensions)
                ][:max_files_per_repo]
                
                # Baixa cada arquivo
                for file_info in files_to_download:
                    try:
                        file_resp = requests.get(
                            file_info['url'],
                            headers=headers,
                            timeout=30,
                        )
                        
                        if file_resp.status_code == 200:
                            file_data = file_resp.json()
                            content = file_data.get('content', '')
                            
                            # Decodifica base64 se necessário
                            if file_data.get('encoding') == 'base64':
                                import base64
                                content = base64.b64decode(content).decode('utf-8', errors='ignore')
                            
                            all_files.append({
                                'repo': repo_name,
                                'path': file_info['path'],
                                'language': repo.get('language', 'unknown'),
                                'content': content,
                                'size': file_info.get('size', 0),
                                'source': 'github',
                            })
                        
                        # Rate limiting
                        time.sleep(0.05)
                        
                    except Exception as e:
                        self.stats['errors'] += 1
                        continue
                
                # Rate limiting entre repos
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  Erro ao processar {repo.get('name', 'unknown')}: {e}")
                self.stats['errors'] += 1
                continue
        
        # Salva todos os arquivos
        files_output = self.output_dir / 'code_files.jsonl'
        with open(files_output, 'w', encoding='utf-8') as f:
            for file_data in all_files:
                f.write(json.dumps(file_data, ensure_ascii=False) + '\n')
        
        self.stats['files'] = len(all_files)
        self.stats['tokens'] = sum(len(f['content'].split()) for f in all_files)
        
        print(f"✅ Baixados {len(all_files)} arquivos")
        print(f"💾 Salvos em: {files_output}")
        
        return len(all_files)
    
    def generate_statistics(self) -> None:
        """Gera estatísticas da coleta"""
        print()
        print("=" * 60)
        print("📊 Estatísticas da Coleta")
        print("=" * 60)
        print(f"Repositórios: {self.stats['repositories']}")
        print(f"Arquivos: {self.stats['files']}")
        print(f"Tokens: {self.stats['tokens']}")
        print(f"Erros: {self.stats['errors']}")
        print()
        
        # Salva estatísticas
        stats_file = self.output_dir / 'collection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'stats': self.stats,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"💾 Estatísticas salvas em: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Coleta de dados de código para Lloavece Kode"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Diretório de saída (default: data/raw)'
    )
    parser.add_argument(
        '--github-token',
        type=str,
        default=None,
        help='Token do GitHub para maior rate limit'
    )
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        default=['github', 'stackoverflow', 'brazilian'],
        choices=['github', 'stackoverflow', 'brazilian'],
        help='Fontes de dados (default: todas)'
    )
    parser.add_argument(
        '--max-repos',
        type=int,
        default=1000,
        help='Número máximo de repositórios do GitHub'
    )
    parser.add_argument(
        '--max-questions',
        type=int,
        default=10000,
        help='Número máximo de perguntas do Stack Overflow'
    )
    
    args = parser.parse_args()
    
    collector = CodeDataCollector(
        output_dir=args.output_dir,
        github_token=args.github_token,
    )
    
    print("=" * 60)
    print("🚀 Lloavece Kode - Coleta de Dados")
    print("=" * 60)
    print()
    
    if 'github' in args.sources:
        repos = collector.collect_github_repos(max_repos=args.max_repos)
        # Baixa os arquivos de código dos repositórios
        collector.download_repo_files(max_files_per_repo=30)
        print()
    
    if 'stackoverflow' in args.sources:
        collector.collect_stackoverflow(max_questions=args.max_questions)
        print()
    
    if 'brazilian' in args.sources:
        repos = collector.collect_brazilian_repos()
        if repos:
            collector.download_repo_files(
                repos_file=collector.output_dir / 'brazilian_repos.json',
                max_files_per_repo=30,
            )
        print()
    
    collector.generate_statistics()
    
    print()
    print("🎉 Coleta concluída!")
    print()
    print("Próximos passos:")
    print("1. Execute: python scripts/curate_data.py")
    print("2. Execute: python training/scripts/train_cpt.py")


if __name__ == "__main__":
    main()
