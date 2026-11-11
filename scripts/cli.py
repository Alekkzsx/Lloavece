#!/usr/bin/env python3
"""
Lloavece Kode - Command Line Interface (CLI)

Interface de linha de comando para interagir com o modelo.

Funcionalidades:
- Chat interativo
- Code completion
- Code explanation
- Code review
- Code generation
- Batch processing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import logging
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress
from rich.prompt import Prompt, Confirm
from rich import print as rprint


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


class LloaveceKode:
    """Interface principal do Lloavece Kode"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Args:
            model_path: Caminho para o modelo
            device: Device para rodar ('cuda', 'cpu', 'auto')
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        console.print(Panel.fit(
            "[bold blue]🚀 Lloavece Kode[/bold blue]\n"
            "IA Local Brasileira para Código",
            border_style="blue"
        ))
        
        self._load_model()
        self.conversation_history = []
        self.system_prompt = """Você é o Lloavece Kode, uma IA especialista em programação e engenharia de software.

Suas características:
- Responde SEMPRE em português brasileiro (PT-BR)
- Foca exclusivamente em código e desenvolvimento de software
- Explica conceitos de forma clara e didática
- Sugere melhores práticas e padrões de projeto
- Aponta bugs, problemas de segurança e otimizações
- Gera código limpo, bem documentado e eficiente

Quando o usuário pedir código:
- Use comentários em português
- Siga as melhores práticas da linguagem
- Inclua tratamento de erros
- Considere edge cases

Quando o usuário pedir explicação:
- Seja didático e claro
- Use exemplos práticos
- Explique o "porquê" além do "como"
"""
    
    def _load_model(self):
        """Carrega modelo e tokenizer"""
        with console.status("[bold green]Carregando modelo...", spinner="dots"):
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            
            console.print(f"[green]✓[/green] Modelo carregado: {self.model_path}")
            console.print(f"[green]✓[/green] Device: {self.device}")
            
            if self.device == "cuda":
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                console.print(f"[green]✓[/green] GPU Memory: {gpu_mem:.1f} GB")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7,
                 top_p: float = 0.9, stream: bool = False) -> str:
        """
        Gera texto a partir de um prompt
        
        Args:
            prompt: Texto de entrada
            max_tokens: Máximo de tokens a gerar
            temperature: Temperatura (0.0 = determinístico, 1.0 = criativo)
            top_p: Nucleus sampling
            stream: Se deve streamar a geração
        
        Returns:
            Texto gerado
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer = None
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        if stream:
            return ""
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()
    
    def chat(self, message: str, max_tokens: int = 1024) -> str:
        """
        Envia mensagem no modo chat
        
        Args:
            message: Mensagem do usuário
            max_tokens: Máximo de tokens na resposta
        
        Returns:
            Resposta do modelo
        """
        # Adiciona mensagem ao histórico
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Constrói prompt com histórico
        prompt = f"{self.system_prompt}\n\n"
        
        for msg in self.conversation_history[-10:]:  # Últimas 10 mensagens
            if msg["role"] == "user":
                prompt += f"### Usuário:\n{msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"### Assistente:\n{msg['content']}\n\n"
        
        prompt += "### Assistente:\n"
        
        # Gera resposta
        response = self.generate(prompt, max_tokens=max_tokens)
        
        # Adiciona resposta ao histórico
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def complete_code(self, code: str, language: str = "python") -> str:
        """
        Completa código
        
        Args:
            code: Código parcial
            language: Linguagem de programação
        
        Returns:
            Código completado
        """
        prompt = f"Complete o seguinte código {language}:\n\n```{language}\n{code}\n```\n\nCódigo completo:\n```{language}\n{code}"
        
        response = self.generate(prompt, max_tokens=512, temperature=0.3)
        
        # Extrai código
        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                return parts[0].strip()
        
        return response
    
    def explain_code(self, code: str, language: str = "python") -> str:
        """
        Explica código
        
        Args:
            code: Código a explicar
            language: Linguagem de programação
        
        Returns:
            Explicação do código
        """
        prompt = f"""Explique detalhadamente o seguinte código {language}:

```{language}
{code}
```

Por favor:
1. Descreva o que o código faz
2. Explique linha por linha (ou bloco por bloco)
3. Aponte conceitos importantes usados
4. Mencione possíveis melhorias
"""
        
        return self.generate(prompt, max_tokens=1024, temperature=0.7)
    
    def review_code(self, code: str, language: str = "python") -> str:
        """
        Faz review de código
        
        Args:
            code: Código a revisar
            language: Linguagem de programação
        
        Returns:
            Review do código
        """
        prompt = f"""Faça uma revisão completa do seguinte código {language}:

```{language}
{code}
```

Por favor avalie:
1. **Correção**: O código funciona corretamente?
2. **Performance**: Há otimizações possíveis?
3. **Segurança**: Há vulnerabilidades?
4. **Legibilidade**: O código é claro e bem estruturado?
5. **Boas práticas**: Segue padrões da comunidade?
6. **Bugs**: Identifique bugs ou edge cases problemáticos

Para cada problema encontrado, sugira uma correção.
"""
        
        return self.generate(prompt, max_tokens=1536, temperature=0.7)
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """
        Gera código a partir de descrição
        
        Args:
            description: Descrição do que o código deve fazer
            language: Linguagem de programação
        
        Returns:
            Código gerado
        """
        prompt = f"""Gere código {language} que faça o seguinte:

{description}

Requisitos:
- Código limpo e bem estruturado
- Comentários explicativos em português
- Tratamento de erros
- Exemplos de uso (se aplicável)
- Siga as melhores práticas da linguagem

Código:
"""
        
        response = self.generate(prompt, max_tokens=1536, temperature=0.7)
        
        # Extrai código
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                if language in part or part.strip().startswith(language):
                    code = '\n'.join(part.split('\n')[1:])
                    return code.strip()
        
        return response
    
    def interactive_chat(self):
        """Inicia chat interativo"""
        console.print(Panel.fit(
            "[bold cyan]💬 Modo Chat Interativo[/bold cyan]\n"
            "Digite 'sair' ou 'exit' para encerrar\n"
            "Digite 'limpar' para resetar o histórico\n"
            "Digite 'ajuda' para ver comandos",
            border_style="cyan"
        ))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]Você[/bold cyan]")
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    console.print("[yellow]👋 Até logo![/yellow]")
                    break
                
                if user_input.lower() == 'limpar':
                    self.conversation_history = []
                    console.print("[green]✓[/green] Histórico limpo")
                    continue
                
                if user_input.lower() == 'ajuda':
                    self._show_help()
                    continue
                
                if not user_input.strip():
                    continue
                
                # Gera resposta
                console.print()
                console.print("[bold magenta]Lloavece Kode:[/bold magenta]")
                response = self.chat(user_input)
                console.print(Markdown(response))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Até logo![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Erro: {e}[/red]")
                logger.error(f"Error in chat: {e}", exc_info=True)
    
    def _show_help(self):
        """Mostra ajuda"""
        table = Table(title="Comandos Disponíveis", show_header=True, header_style="bold cyan")
        table.add_column("Comando", style="cyan")
        table.add_column("Descrição", style="white")
        
        commands = [
            ("sair/exit", "Encerra o programa"),
            ("limpar", "Limpa o histórico de conversa"),
            ("ajuda", "Mostra esta ajuda"),
            ("complete", "Modo de completar código"),
            ("explain", "Modo de explicar código"),
            ("review", "Modo de revisão de código"),
            ("generate", "Modo de gerar código"),
        ]
        
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        
        console.print(table)
    
    def batch_process(self, input_file: str, output_file: str, mode: str = "chat"):
        """
        Processa múltiplas entradas em batch
        
        Args:
            input_file: Arquivo de entrada (JSONL)
            output_file: Arquivo de saída (JSONL)
            mode: Modo de processamento (chat, complete, explain, review, generate)
        """
        console.print(f"[bold blue]🔄 Processando em batch...[/bold blue]")
        console.print(f"   Input: {input_file}")
        console.print(f"   Output: {output_file}")
        console.print(f"   Mode: {mode}")
        
        # Carrega entradas
        entries = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))
        
        console.print(f"   Entradas: {len(entries)}")
        
        # Processa
        results = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processando...", total=len(entries))
            
            for entry in entries:
                if mode == "chat":
                    response = self.chat(entry['message'])
                elif mode == "complete":
                    response = self.complete_code(entry['code'], entry.get('language', 'python'))
                elif mode == "explain":
                    response = self.explain_code(entry['code'], entry.get('language', 'python'))
                elif mode == "review":
                    response = self.review_code(entry['code'], entry.get('language', 'python'))
                elif mode == "generate":
                    response = self.generate_code(entry['description'], entry.get('language', 'python'))
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                results.append({
                    **entry,
                    'response': response
                })
                
                progress.update(task, advance=1)
        
        # Salva resultados
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        console.print(f"[green]✓[/green] Processamento concluído -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Lloavece Kode - CLI")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--mode", type=str, default="chat",
                       choices=["chat", "complete", "explain", "review", "generate", "batch"],
                       help="Mode of operation")
    
    # Batch mode options
    parser.add_argument("--input-file", type=str, default=None,
                       help="Input file for batch processing (JSONL)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file for batch processing (JSONL)")
    
    # Single prompt mode
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt to process")
    parser.add_argument("--language", type=str, default="python",
                       help="Programming language")
    
    args = parser.parse_args()
    
    # Initialize Lloavece
    lloavece = LloaveceKode(args.model_path, args.device)
    
    # Run in appropriate mode
    if args.mode == "chat":
        if args.prompt:
            # Single prompt
            response = lloavece.chat(args.prompt)
            console.print(Markdown(response))
        else:
            # Interactive chat
            lloavece.interactive_chat()
    
    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            console.print("[red]❌ Batch mode requires --input-file and --output-file[/red]")
            sys.exit(1)
        
        lloavece.batch_process(args.input_file, args.output_file, args.mode)
    
    else:
        # Single prompt mode
        if not args.prompt:
            console.print(f"[red]❌ Mode '{args.mode}' requires --prompt[/red]")
            sys.exit(1)
        
        if args.mode == "complete":
            response = lloavece.complete_code(args.prompt, args.language)
            console.print(Syntax(response, args.language, theme="monokai"))
        elif args.mode == "explain":
            response = lloavece.explain_code(args.prompt, args.language)
            console.print(Markdown(response))
        elif args.mode == "review":
            response = lloavece.review_code(args.prompt, args.language)
            console.print(Markdown(response))
        elif args.mode == "generate":
            response = lloavece.generate_code(args.prompt, args.language)
            console.print(Syntax(response, args.language, theme="monokai"))


if __name__ == "__main__":
    main()
