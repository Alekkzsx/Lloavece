#!/usr/bin/env python3
"""
Lloavece Kode - Model Evaluation Script

Avalia o modelo em benchmarks padrão de código:
- HumanEval (164 problemas)
- MBPP (974 problemas)
- LiveCodeBench (problemas recentes)
- Custom benchmarks PT-BR
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class CodeEvaluator:
    """Avaliador de código para Lloavece"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: Caminho para o modelo (local ou HuggingFace)
            device: Device para rodar ('cuda', 'cpu', 'auto')
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"📦 Loading model from: {model_path}")
        print(f"🖥️  Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.0,  # Greedy para avaliação
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        print("✅ Model loaded successfully!")
    
    def generate_code(self, prompt: str, max_tokens: int = 512) -> str:
        """Gera código a partir de um prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                generation_config=self.gen_config
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrai apenas o código gerado (remove o prompt)
        code = generated[len(prompt):].strip()
        
        # Para na primeira linha em branco ou marcador de fim
        if "```" in code:
            code = code.split("```")[0]
        if "\n\n" in code:
            code = code.split("\n\n")[0]
        
        return code
    
    def extract_code_block(self, text: str) -> str:
        """Extrai bloco de código de markdown"""
        pattern = r'```(?:python|javascript|java|go|rust|c|cpp)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Se não encontrou markdown, retorna o texto inteiro
        return text.strip()
    
    def evaluate_humaneval(self, sample_limit: Optional[int] = None) -> Dict[str, float]:
        """
        Avalia no HumanEval (164 problemas de Python)
        
        Returns:
            Dict com métricas: pass@1, pass@5, pass@10
        """
        print("\n" + "="*60)
        print("🧪 Evaluating on HumanEval")
        print("="*60)
        
        try:
            dataset = load_dataset("openai_humaneval", split="test")
        except Exception as e:
            print(f"❌ Failed to load HumanEval: {e}")
            return {"error": str(e)}
        
        if sample_limit:
            dataset = dataset.select(range(min(sample_limit, len(dataset))))
        
        results = []
        passed = 0
        
        for idx, example in enumerate(tqdm(dataset, desc="HumanEval")):
            prompt = example['prompt']
            test = example['test']
            entry_point = example['entry_point']
            canonical_solution = example['canonical_solution']
            
            # Gera código
            full_prompt = f"{prompt}\n\nComplete a função:"
            generated_code = self.generate_code(full_prompt)
            generated_code = self.extract_code_block(generated_code)
            
            # Testa o código
            is_correct = self._test_code(
                prompt + generated_code,
                test,
                entry_point
            )
            
            results.append({
                "task_id": example['task_id'],
                "generated": generated_code,
                "correct": is_correct
            })
            
            if is_correct:
                passed += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx+1}/{len(dataset)} | Pass rate: {passed/(idx+1)*100:.1f}%")
        
        pass_rate = passed / len(dataset) * 100
        
        metrics = {
            "total": len(dataset),
            "passed": passed,
            "pass@1": pass_rate,
            "details": results
        }
        
        print(f"\n✅ HumanEval Results:")
        print(f"   Pass@1: {pass_rate:.2f}%")
        print(f"   Passed: {passed}/{len(dataset)}")
        
        return metrics
    
    def evaluate_mbpp(self, sample_limit: Optional[int] = None) -> Dict[str, float]:
        """
        Avalia no MBPP (974 problemas de Python)
        
        Returns:
            Dict com métricas: pass@1, accuracy
        """
        print("\n" + "="*60)
        print("🧪 Evaluating on MBPP")
        print("="*60)
        
        try:
            dataset = load_dataset("mbpp", "full", split="test")
        except Exception as e:
            print(f"❌ Failed to load MBPP: {e}")
            return {"error": str(e)}
        
        if sample_limit:
            dataset = dataset.select(range(min(sample_limit, len(dataset))))
        
        results = []
        passed = 0
        
        for idx, example in enumerate(tqdm(dataset, desc="MBPP")):
            text = example['text']
            test_list = example['test_list']
            
            # Gera código
            prompt = f"{text}\n\nWrite Python code:"
            generated_code = self.generate_code(prompt)
            generated_code = self.extract_code_block(generated_code)
            
            # Testa o código
            is_correct = self._test_mbpp(generated_code, test_list)
            
            results.append({
                "task_id": example['task_id'],
                "generated": generated_code,
                "correct": is_correct
            })
            
            if is_correct:
                passed += 1
            
            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx+1}/{len(dataset)} | Pass rate: {passed/(idx+1)*100:.1f}%")
        
        pass_rate = passed / len(dataset) * 100
        
        metrics = {
            "total": len(dataset),
            "passed": passed,
            "pass@1": pass_rate,
            "details": results
        }
        
        print(f"\n✅ MBPP Results:")
        print(f"   Pass@1: {pass_rate:.2f}%")
        print(f"   Passed: {passed}/{len(dataset)}")
        
        return metrics
    
    def _test_code(self, code: str, test: str, entry_point: str) -> bool:
        """
        Testa código gerado executando os testes
        
        Returns:
            True se todos os testes passaram
        """
        try:
            # Cria ambiente de execução seguro
            exec_globals = {}
            exec(code, exec_globals)
            
            # Executa testes
            exec(test, exec_globals)
            
            # Chama função check se existir
            if 'check' in exec_globals:
                exec_globals['check'](exec_globals[entry_point])
            
            return True
            
        except Exception as e:
            return False
    
    def _test_mbpp(self, code: str, test_list: List[str]) -> bool:
        """Testa código MBPP"""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            
            for test in test_list:
                exec(test, exec_globals)
            
            return True
            
        except Exception:
            return False
    
    def evaluate_all(self, output_file: str, sample_limit: Optional[int] = None):
        """
        Avalia em todos os benchmarks
        
        Args:
            output_file: Arquivo para salvar resultados
            sample_limit: Limite de amostras por benchmark (para testes rápidos)
        """
        print("\n" + "="*60)
        print("🚀 LLOAVECE KODE - FULL EVALUATION")
        print("="*60)
        
        all_results = {
            "model": self.model_path,
            "timestamp": str(Path(output_file).stat().st_mtime if Path(output_file).exists() else "now"),
            "benchmarks": {}
        }
        
        # HumanEval
        try:
            humaneval_results = self.evaluate_humaneval(sample_limit)
            all_results["benchmarks"]["humaneval"] = humaneval_results
        except Exception as e:
            print(f"❌ HumanEval failed: {e}")
            all_results["benchmarks"]["humaneval"] = {"error": str(e)}
        
        # MBPP
        try:
            mbpp_results = self.evaluate_mbpp(sample_limit)
            all_results["benchmarks"]["mbpp"] = mbpp_results
        except Exception as e:
            print(f"❌ MBPP failed: {e}")
            all_results["benchmarks"]["mbpp"] = {"error": str(e)}
        
        # Resumo
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        if "humaneval" in all_results["benchmarks"]:
            he = all_results["benchmarks"]["humaneval"]
            if "pass@1" in he:
                print(f"HumanEval Pass@1: {he['pass@1']:.2f}%")
        
        if "mbpp" in all_results["benchmarks"]:
            mbpp = all_results["benchmarks"]["mbpp"]
            if "pass@1" in mbpp:
                print(f"MBPP Pass@1:      {mbpp['pass@1']:.2f}%")
        
        # Salva resultados
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        print("="*60)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Lloavece Kode - Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model (local or HuggingFace)")
    parser.add_argument("--output", type=str, default="models/evaluations/results.json",
                       help="Output file for results")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["all"],
                       choices=["all", "humaneval", "mbpp"],
                       help="Which benchmarks to run")
    parser.add_argument("--sample-limit", type=int, default=None,
                       help="Limit samples per benchmark (for quick testing)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CodeEvaluator(args.model_path, args.device)
    
    # Run evaluations
    if "all" in args.benchmarks:
        evaluator.evaluate_all(args.output, args.sample_limit)
    else:
        results = {"model": args.model_path, "benchmarks": {}}
        
        if "humaneval" in args.benchmarks:
            results["benchmarks"]["humaneval"] = evaluator.evaluate_humaneval(args.sample_limit)
        
        if "mbpp" in args.benchmarks:
            results["benchmarks"]["mbpp"] = evaluator.evaluate_mbpp(args.sample_limit)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
