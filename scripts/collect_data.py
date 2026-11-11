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
        collector.collect_github_repos(max_repos=args.max_repos)
        print()
    
    if 'stackoverflow' in args.sources:
        collector.collect_stackoverflow(max_questions=args.max_questions)
        print()
    
    if 'brazilian' in args.sources:
        collector.collect_brazilian_repos()
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
