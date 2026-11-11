#!/usr/bin/env python3
"""
Lloavece Kode - Self-Improvement Loop

Loop iterativo de auto-aprimoramento onde o modelo:
1. Gera respostas para problemas de código
2. Avalia a qualidade (testes + juiz LLM)
3. Cria pares (resposta_ruim, resposta_boa)
4. Treina DPO nos pares
5. Repete com modelo melhorado

Pipeline:
1. Gera N respostas por problema
2. Executa testes automáticos
3. Avalia qualidade com múltiplas métricas
4. Cria pares de preferência
5. Fine-tune DPO
6. Avalia melhoria
7. Repete até saturação
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SelfImprovementConfig:
    """Configuração do loop de self-improvement"""
    model_path: str
    output_dir: str
    max_iterations: int = 6
    min_improvement: float = 1.0  # % mínimo de melhoria para continuar
    n_samples: int = 5  # Respostas por problema
    temperature: float = 0.8
    max_new_tokens: int = 512
    min_score_diff: float = 0.2  # Diferença mínima para criar par
    train_epochs: int = 1
    train_batch_size: int = 2
    learning_rate: float = 5e-6
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    seed: int = 42


class CodeEvaluator:
    """Avaliador de código gerado"""
    
    def __init__(self):
        self.test_results = {}
    
    def evaluate_code(self, code: str, tests: List[str], language: str = "python") -> Dict:
        """
        Avalia código com múltiplas métricas
        
        Returns:
            Dict com scores:
                - test_score: % de testes que passaram
                - syntax_score: código é sintaticamente válido
                - quality_score: qualidade do código (pylint, etc)
                - total_score: score combinado
        """
        scores = {
            "test_score": 0.0,
            "syntax_score": 0.0,
            "quality_score": 0.0,
            "total_score": 0.0
        }
        
        if language == "python":
            scores = self._evaluate_python(code, tests)
        elif language in ["javascript", "typescript"]:
            scores = self._evaluate_javascript(code, tests)
        else:
            # Linguagem não suportada, retorna score baixo
            logger.warning(f"Language {language} not supported, using minimal evaluation")
            scores["syntax_score"] = 1.0 if self._check_syntax(code, language) else 0.0
            scores["total_score"] = scores["syntax_score"] * 0.5
        
        return scores
    
    def _evaluate_python(self, code: str, tests: List[str]) -> Dict:
        """Avalia código Python"""
        scores = {
            "test_score": 0.0,
            "syntax_score": 0.0,
            "quality_score": 0.0,
            "total_score": 0.0
        }
        
        # 1. Check syntax
        try:
            compile(code, "<string>", "exec")
            scores["syntax_score"] = 1.0
        except SyntaxError:
            scores["syntax_score"] = 0.0
            scores["total_score"] = 0.0
            return scores
        
        # 2. Run tests
        if tests:
            passed = 0
            total = len(tests)
            
            # Cria arquivo temporário
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.write('\n\n')
                for test in tests:
                    f.write(test)
                    f.write('\n')
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                # Conta testes que passaram
                # Assume que testes imprimem "PASS" ou "FAIL"
                output = result.stdout + result.stderr
                passed = output.count("PASS")
                total = max(output.count("PASS") + output.count("FAIL"), total)
                
                if total > 0:
                    scores["test_score"] = passed / total
                
            except subprocess.TimeoutExpired:
                scores["test_score"] = 0.0
            except Exception as e:
                logger.error(f"Error running tests: {e}")
                scores["test_score"] = 0.0
            finally:
                os.unlink(temp_file)
        
        # 3. Quality score (simplified)
        # Aqui você poderia usar pylint, flake8, etc
        scores["quality_score"] = self._simple_quality_score(code)
        
        # 4. Total score (weighted)
        scores["total_score"] = (
            scores["test_score"] * 0.6 +
            scores["syntax_score"] * 0.2 +
            scores["quality_score"] * 0.2
        )
        
        return scores
    
    def _evaluate_javascript(self, code: str, tests: List[str]) -> Dict:
        """Avalia código JavaScript"""
        scores = {
            "test_score": 0.0,
            "syntax_score": 0.0,
            "quality_score": 0.0,
            "total_score": 0.0
        }
        
        # 1. Check syntax
        try:
            # Usa node para validar
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['node', '--check', temp_file],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                scores["syntax_score"] = 1.0
            else:
                scores["syntax_score"] = 0.0
                
        except Exception as e:
            logger.error(f"Error checking JS syntax: {e}")
            scores["syntax_score"] = 0.0
        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)
        
        # 2. Tests (simplified)
        if tests:
            # Aqui você implementaria execução de testes JS
            scores["test_score"] = 0.5  # Placeholder
        
        # 3. Quality
        scores["quality_score"] = self._simple_quality_score(code)
        
        # 4. Total
        scores["total_score"] = (
            scores["test_score"] * 0.6 +
            scores["syntax_score"] * 0.2 +
            scores["quality_score"] * 0.2
        )
        
        return scores
    
    def _check_syntax(self, code: str, language: str) -> bool:
        """Check básico de sintaxe"""
        try:
            if language == "python":
                compile(code, "<string>", "exec")
                return True
            elif language in ["javascript", "typescript"]:
                # Check básico
                return code.count('{') == code.count('}')
            else:
                return True
        except:
            return False
    
    def _simple_quality_score(self, code: str) -> float:
        """Score simples de qualidade"""
        score = 1.0
        
        # Penaliza linhas muito longas
        long_lines = sum(1 for line in code.split('\n') if len(line) > 100)
        if long_lines > 0:
            score -= min(0.3, long_lines * 0.05)
        
        # Penaliza falta de comentários (para funções)
        if 'def ' in code or 'function ' in code:
            if '"""' not in code and "'''" not in code and '#' not in code:
                score -= 0.1
        
        # Penaliza variáveis de uma letra (exceto i, j, k)
        single_letter_vars = len([w for w in code.split() if len(w) == 1 and w.isalpha() and w not in 'ijk'])
        if single_letter_vars > 5:
            score -= 0.1
        
        return max(0.0, score)


class SelfImprovementLoop:
    """Loop de auto-aprimoramento"""
    
    def __init__(self, config: SelfImprovementConfig):
        self.config = config
        self.evaluator = CodeEvaluator()
        self.current_model_path = config.model_path
        self.iteration = 0
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project="lloavece-kode",
            name=f"self-improvement-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(config)
        )
        
        logger.info(f"🚀 Self-Improvement Loop initialized")
        logger.info(f"   Max iterations: {config.max_iterations}")
        logger.info(f"   Min improvement: {config.min_improvement}%")
        logger.info(f"   Samples per problem: {config.n_samples}")
    
    def load_model(self):
        """Carrega modelo atual"""
        logger.info(f"📦 Loading model from: {self.current_model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.current_model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.current_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        return model, tokenizer
    
    def generate_responses(self, model, tokenizer, prompts: List[Dict]) -> List[Dict]:
        """Gera múltiplas respostas para cada prompt"""
        logger.info(f"🔄 Generating {self.config.n_samples} responses per prompt...")
        
        results = []
        
        model.eval()
        
        for item in tqdm(prompts, desc="Generating"):
            prompt = item['prompt']
            tests = item.get('tests', [])
            language = item.get('language', 'python')
            
            responses = []
            
            for _ in range(self.config.n_samples):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                code = generated[len(prompt):].strip()
                
                # Extrai bloco de código
                if "```" in code:
                    code = code.split("```")[1]
                    if code.startswith("python") or code.startswith("javascript"):
                        code = '\n'.join(code.split('\n')[1:])
                
                responses.append(code)
            
            results.append({
                "prompt": prompt,
                "responses": responses,
                "tests": tests,
                "language": language
            })
        
        return results
    
    def evaluate_responses(self, results: List[Dict]) -> List[Dict]:
        """Avalia todas as respostas"""
        logger.info("📊 Evaluating responses...")
        
        evaluated = []
        
        for item in tqdm(results, desc="Evaluating"):
            prompt = item['prompt']
            responses = item['responses']
            tests = item['tests']
            language = item['language']
            
            # Avalia cada resposta
            scored_responses = []
            for code in responses:
                scores = self.evaluator.evaluate_code(code, tests, language)
                scored_responses.append({
                    "code": code,
                    "scores": scores
                })
            
            # Ordena por total_score
            scored_responses.sort(key=lambda x: x['scores']['total_score'], reverse=True)
            
            evaluated.append({
                "prompt": prompt,
                "responses": scored_responses,
                "tests": tests,
                "language": language
            })
        
        return evaluated
    
    def create_preference_pairs(self, evaluated: List[Dict]) -> List[Dict]:
        """Cria pares de preferência (melhor vs pior)"""
        logger.info("🔗 Creating preference pairs...")
        
        pairs = []
        
        for item in evaluated:
            prompt = item['prompt']
            responses = item['responses']
            
            if len(responses) < 2:
                continue
            
            best = responses[0]
            worst = responses[-1]
            
            # Só cria par se diferença for significativa
            score_diff = best['scores']['total_score'] - worst['scores']['total_score']
            
            if score_diff >= self.config.min_score_diff:
                pairs.append({
                    "prompt": prompt,
                    "chosen": best['code'],
                    "rejected": worst['code'],
                    "score_diff": score_diff
                })
        
        logger.info(f"✅ Created {len(pairs)} preference pairs")
        return pairs
    
    def train_dpo(self, pairs: List[Dict]) -> str:
        """Treina DPO nos pares"""
        logger.info("🎓 Training DPO...")
        
        # Salva pares
        pairs_file = self.output_dir / f"pairs_iter_{self.iteration}.jsonl"
        with open(pairs_file, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        # Prepara comando de treino
        output_dir = self.output_dir / f"checkpoint_iter_{self.iteration}"
        
        cmd = [
            "python", "training/scripts/train_dpo.py",
            "--model_name_or_path", self.current_model_path,
            "--train_data", str(pairs_file),
            "--output_dir", str(output_dir),
            "--num_train_epochs", str(self.config.train_epochs),
            "--per_device_train_batch_size", str(self.config.train_batch_size),
            "--learning_rate", str(self.config.learning_rate),
            "--use_lora" if self.config.use_lora else "--no-use_lora",
            "--lora_r", str(self.config.lora_r),
            "--lora_alpha", str(self.config.lora_alpha),
            "--report_to", "wandb",
            "--run_name", f"self-improvement-iter-{self.iteration}"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"DPO training failed: {result.stderr}")
            raise RuntimeError(f"DPO training failed: {result.stderr}")
        
        logger.info(f"✅ DPO training completed -> {output_dir}")
        
        return str(output_dir / "merged")
    
    def evaluate_improvement(self) -> float:
        """Avalia melhoria do modelo"""
        logger.info("📈 Evaluating improvement...")
        
        # Roda avaliação rápida
        eval_cmd = [
            "python", "scripts/evaluate_model.py",
            "--model-path", self.current_model_path,
            "--output", str(self.output_dir / f"eval_iter_{self.iteration}.json"),
            "--sample-limit", "50"  # Avaliação rápida
        ]
        
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Evaluation failed: {result.stderr}")
            return 0.0
        
        # Carrega resultados
        eval_file = self.output_dir / f"eval_iter_{self.iteration}.json"
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)
        
        # Calcula score médio
        scores = []
        if "benchmarks" in eval_results:
            for benchmark, metrics in eval_results["benchmarks"].items():
                if "pass@1" in metrics:
                    scores.append(metrics["pass@1"])
        
        if scores:
            avg_score = np.mean(scores)
            logger.info(f"✅ Average score: {avg_score:.2f}%")
            return avg_score
        else:
            return 0.0
    
    def run(self, prompts_file: str):
        """
        Executa o loop completo
        
        Args:
            prompts_file: Arquivo com prompts (JSONL)
        """
        logger.info("="*60)
        logger.info("🚀 STARTING SELF-IMPROVEMENT LOOP")
        logger.info("="*60)
        
        # Carrega prompts
        prompts = []
        with open(prompts_file, 'r') as f:
            for line in f:
                prompts.append(json.loads(line))
        
        logger.info(f"Loaded {len(prompts)} prompts")
        
        # Avalia modelo inicial
        initial_score = self.evaluate_improvement()
        logger.info(f"Initial score: {initial_score:.2f}%")
        
        previous_score = initial_score
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            logger.info("\n" + "="*60)
            logger.info(f"🔄 ITERATION {iteration + 1}/{self.config.max_iterations}")
            logger.info("="*60)
            
            # 1. Carrega modelo
            model, tokenizer = self.load_model()
            
            # 2. Gera respostas
            results = self.generate_responses(model, tokenizer, prompts)
            
            # 3. Avalia respostas
            evaluated = self.evaluate_responses(results)
            
            # 4. Cria pares
            pairs = self.create_preference_pairs(evaluated)
            
            if len(pairs) < 100:
                logger.warning(f"⚠️ Too few pairs ({len(pairs)}), stopping")
                break
            
            # 5. Treina DPO
            new_model_path = self.train_dpo(pairs)
            
            # 6. Atualiza modelo atual
            self.current_model_path = new_model_path
            
            # 7. Avalia melhoria
            current_score = self.evaluate_improvement()
            improvement = current_score - previous_score
            
            logger.info(f"📊 Score: {previous_score:.2f}% -> {current_score:.2f}% (Δ: {improvement:+.2f}%)")
            
            # Log para wandb
            wandb.log({
                "iteration": iteration,
                "score": current_score,
                "improvement": improvement,
                "pairs_created": len(pairs)
            })
            
            # Verifica se vale continuar
            if improvement < self.config.min_improvement:
                logger.info(f"⚠️ Improvement below threshold ({self.config.min_improvement}%), stopping")
                break
            
            previous_score = current_score
            
            # Limpa memória
            del model, tokenizer
            torch.cuda.empty_cache()
        
        logger.info("\n" + "="*60)
        logger.info("✅ SELF-IMPROVEMENT LOOP COMPLETED")
        logger.info("="*60)
        logger.info(f"Initial score: {initial_score:.2f}%")
        logger.info(f"Final score: {previous_score:.2f}%")
        logger.info(f"Total improvement: {previous_score - initial_score:+.2f}%")
        logger.info(f"Final model: {self.current_model_path}")
        
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Lloavece Kode - Self-Improvement Loop")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--prompts-file", type=str, required=True,
                       help="JSONL file with prompts")
    parser.add_argument("--output-dir", type=str, default="models/self_improvement",
                       help="Output directory")
    parser.add_argument("--max-iterations", type=int, default=6,
                       help="Maximum number of iterations")
    parser.add_argument("--min-improvement", type=float, default=1.0,
                       help="Minimum improvement (%) to continue")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of responses per prompt")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (JSON)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SelfImprovementConfig(**config_dict)
    else:
        config = SelfImprovementConfig(
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            min_improvement=args.min_improvement,
            n_samples=args.n_samples
        )
    
    # Run loop
    loop = SelfImprovementLoop(config)
    loop.run(args.prompts_file)


if __name__ == "__main__":
    main()
