#!/usr/bin/env python3
"""
Lloavece Kode - Model Quantization Script

Quantiza o modelo para formatos eficientes:
- GGUF (para llama.cpp)
- AWQ (para inferência rápida)
- GPTQ (alternativa)

Pipeline:
1. Carrega modelo base (pós-treinamento)
2. Calibra com dataset de calibração
3. Aplica quantização
4. Salva modelo quantizado
5. Testa qualidade
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuração de quantização"""
    model_path: str
    output_dir: str
    method: str = "gguf"  # "gguf", "awq", "gptq"
    
    # GGUF options
    gguf_quant_type: str = "Q4_K_M"  # Q2_K, Q3_K_S, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0
    
    # AWQ options
    awq_bits: int = 4
    awq_group_size: int = 128
    
    # GPTQ options
    gptq_bits: int = 4
    gptq_group_size: int = 128
    
    # Calibration
    calibration_samples: int = 512
    calibration_max_length: int = 2048
    calibration_dataset: str = "wikitext"


class ModelQuantizer:
    """Quantizador de modelos"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Quantizer initialized")
        logger.info(f"   Method: {config.method}")
        logger.info(f"   Model: {config.model_path}")
        logger.info(f"   Output: {config.output_dir}")
    
    def load_calibration_data(self) -> List[str]:
        """Carrega dados de calibração"""
        logger.info(f"📊 Loading calibration data ({self.config.calibration_samples} samples)...")
        
        texts = []
        
        if self.config.calibration_dataset == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            for i, item in enumerate(dataset):
                if i >= self.config.calibration_samples:
                    break
                text = item['text'].strip()
                if len(text) > 50:  # Pula textos muito curtos
                    texts.append(text)
        
        elif self.config.calibration_dataset == "code":
            # Usa dados de código para calibração
            dataset = load_dataset("code_search_net", "python", split="train")
            for i, item in enumerate(dataset):
                if i >= self.config.calibration_samples:
                    break
                code = item.get('whole_func_string', '')
                if code:
                    texts.append(code)
        
        else:
            # Dataset custom
            dataset = load_dataset(self.config.calibration_dataset, split="train")
            for i, item in enumerate(dataset):
                if i >= self.config.calibration_samples:
                    break
                text = str(item.get('text', ''))
                if len(text) > 50:
                    texts.append(text)
        
        logger.info(f"✅ Loaded {len(texts)} calibration samples")
        return texts
    
    def quantize_gguf(self) -> str:
        """Quantiza para GGUF usando llama.cpp"""
        logger.info("🔨 Quantizing to GGUF...")
        
        # 1. Converte modelo para formato llama.cpp
        logger.info("   Step 1: Converting to llama.cpp format...")
        
        convert_script = self._find_llama_cpp_convert_script()
        if not convert_script:
            raise RuntimeError("llama.cpp convert script not found. Please install llama.cpp first.")
        
        # Cria diretório temporário
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Converte para formato gguf (fp16)
            fp16_output = tmpdir / "model-f16.gguf"
            
            cmd = [
                "python", convert_script,
                "--outfile", str(fp16_output),
                "--outtype", "f16",
                self.config.model_path
            ]
            
            logger.info(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Conversion failed: {result.stderr}")
                raise RuntimeError(f"Conversion failed: {result.stderr}")
            
            # 2. Quantiza
            logger.info(f"   Step 2: Quantizing to {self.config.gguf_quant_type}...")
            
            quantize_script = self._find_llama_cpp_quantize_script()
            if not quantize_script:
                raise RuntimeError("llama.cpp quantize script not found")
            
            output_file = self.output_dir / f"lloavece-kode-{self.config.gguf_quant_type.lower()}.gguf"
            
            cmd = [
                quantize_script,
                str(fp16_output),
                str(output_file),
                self.config.gguf_quant_type
            ]
            
            logger.info(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Quantization failed: {result.stderr}")
                raise RuntimeError(f"Quantization failed: {result.stderr}")
            
            logger.info(f"✅ GGUF quantization completed -> {output_file}")
            
            # 3. Verifica tamanho
            file_size = output_file.stat().st_size / (1024**3)  # GB
            logger.info(f"   File size: {file_size:.2f} GB")
            
            return str(output_file)
    
    def quantize_awq(self) -> str:
        """Quantiza usando AWQ"""
        logger.info("🔨 Quantizing with AWQ...")
        
        try:
            from awq import AutoAWQForCausalLM
            from awq.utils import fused_module
        except ImportError:
            raise RuntimeError("AWQ not installed. Run: pip install autoawq")
        
        # 1. Carrega dados de calibração
        calibration_texts = self.load_calibration_data()
        
        # 2. Carrega tokenizer
        logger.info("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        # 3. Carrega modelo AWQ
        logger.info("   Loading model for AWQ...")
        model = AutoAWQForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 4. Quantiza
        logger.info(f"   Quantizing to {self.config.awq_bits}-bit...")
        
        quant_config = {
            "zero_point": True,
            "q_group_size": self.config.awq_group_size,
            "w_bit": self.config.awq_bits,
            "version": "GEMM"
        }
        
        # Prepara dados de calibração
        calib_data = []
        for text in calibration_texts[:self.config.calibration_samples]:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.calibration_max_length,
                truncation=True
            )
            calib_data.append(tokens)
        
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
        
        # 5. Salva modelo quantizado
        output_path = self.output_dir / f"lloavece-kode-awq-{self.config.awq_bits}bit"
        logger.info(f"   Saving quantized model to: {output_path}")
        
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"✅ AWQ quantization completed -> {output_path}")
        
        return str(output_path)
    
    def quantize_gptq(self) -> str:
        """Quantiza usando GPTQ"""
        logger.info("🔨 Quantizing with GPTQ...")
        
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise RuntimeError("AutoGPTQ not installed. Run: pip install auto-gptq")
        
        # 1. Carrega dados de calibração
        calibration_texts = self.load_calibration_data()
        
        # 2. Carrega tokenizer
        logger.info("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 3. Configura quantização
        quantize_config = BaseQuantizeConfig(
            bits=self.config.gptq_bits,
            group_size=self.config.gptq_group_size,
            desc_act=False,
            sym=True
        )
        
        # 4. Carrega modelo
        logger.info("   Loading model for GPTQ...")
        model = AutoGPTQForCausalLM.from_pretrained(
            self.config.model_path,
            quantize_config=quantize_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 5. Prepara dados de calibração
        logger.info("   Preparing calibration data...")
        calib_data = []
        for text in calibration_texts[:self.config.calibration_samples]:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.calibration_max_length,
                truncation=True,
                padding=True
            )
            calib_data.append({
                "input_ids": tokens.input_ids,
                "attention_mask": tokens.attention_mask
            })
        
        # 6. Quantiza
        logger.info(f"   Quantizing to {self.config.gptq_bits}-bit...")
        model.quantize(calib_data)
        
        # 7. Salva modelo quantizado
        output_path = self.output_dir / f"lloavece-kode-gptq-{self.config.gptq_bits}bit"
        logger.info(f"   Saving quantized model to: {output_path}")
        
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"✅ GPTQ quantization completed -> {output_path}")
        
        return str(output_path)
    
    def _find_llama_cpp_convert_script(self) -> Optional[str]:
        """Encontra script de conversão do llama.cpp"""
        possible_paths = [
            "llama.cpp/convert.py",
            "../llama.cpp/convert.py",
            "../../llama.cpp/convert.py",
            os.path.expanduser("~/llama.cpp/convert.py"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Tenta encontrar no PATH
        try:
            result = subprocess.run(["which", "llama-convert"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def _find_llama_cpp_quantize_script(self) -> Optional[str]:
        """Encontra script de quantização do llama.cpp"""
        possible_paths = [
            "llama.cpp/quantize",
            "../llama.cpp/quantize",
            "../../llama.cpp/quantize",
            os.path.expanduser("~/llama.cpp/quantize"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Tenta encontrar no PATH
        try:
            result = subprocess.run(["which", "llama-quantize"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def test_quantized_model(self, model_path: str):
        """Testa modelo quantizado"""
        logger.info("🧪 Testing quantized model...")
        
        if model_path.endswith(".gguf"):
            # Testa com llama.cpp
            try:
                cmd = ["llama-cli", "-m", model_path, "-p", "def fibonacci(n):", "-n", "50"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info(f"✅ Model test passed")
                    logger.info(f"   Output sample: {result.stdout[:200]}")
                else:
                    logger.warning(f"⚠️ Model test failed: {result.stderr}")
            except FileNotFoundError:
                logger.warning("⚠️ llama-cli not found, skipping test")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Model test timeout")
        else:
            # Testa com transformers
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                prompt = "def fibonacci(n):"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                outputs = model.generate(**inputs, max_new_tokens=50)
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                logger.info(f"✅ Model test passed")
                logger.info(f"   Output sample: {generated[:200]}")
                
            except Exception as e:
                logger.warning(f"⚠️ Model test failed: {e}")
    
    def run(self):
        """Executa quantização"""
        logger.info("="*60)
        logger.info("🔨 MODEL QUANTIZATION")
        logger.info("="*60)
        
        try:
            if self.config.method == "gguf":
                output_path = self.quantize_gguf()
            elif self.config.method == "awq":
                output_path = self.quantize_awq()
            elif self.config.method == "gptq":
                output_path = self.quantize_gptq()
            else:
                raise ValueError(f"Unknown quantization method: {self.config.method}")
            
            # Testa modelo
            self.test_quantized_model(output_path)
            
            logger.info("\n" + "="*60)
            logger.info("✅ QUANTIZATION COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Output: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Lloavece Kode - Model Quantization")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model to quantize")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for quantized model")
    parser.add_argument("--method", type=str, default="gguf",
                       choices=["gguf", "awq", "gptq"],
                       help="Quantization method")
    
    # GGUF options
    parser.add_argument("--gguf-quant-type", type=str, default="Q4_K_M",
                       choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", 
                               "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
                       help="GGUF quantization type")
    
    # AWQ options
    parser.add_argument("--awq-bits", type=int, default=4, choices=[4, 8],
                       help="AWQ bits")
    parser.add_argument("--awq-group-size", type=int, default=128,
                       help="AWQ group size")
    
    # GPTQ options
    parser.add_argument("--gptq-bits", type=int, default=4, choices=[2, 3, 4, 8],
                       help="GPTQ bits")
    parser.add_argument("--gptq-group-size", type=int, default=128,
                       help="GPTQ group size")
    
    # Calibration
    parser.add_argument("--calibration-samples", type=int, default=512,
                       help="Number of calibration samples")
    parser.add_argument("--calibration-dataset", type=str, default="wikitext",
                       help="Calibration dataset")
    
    args = parser.parse_args()
    
    # Create config
    config = QuantizationConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        method=args.method,
        gguf_quant_type=args.gguf_quant_type,
        awq_bits=args.awq_bits,
        awq_group_size=args.awq_group_size,
        gptq_bits=args.gptq_bits,
        gptq_group_size=args.gptq_group_size,
        calibration_samples=args.calibration_samples,
        calibration_dataset=args.calibration_dataset
    )
    
    # Run quantization
    quantizer = ModelQuantizer(config)
    quantizer.run()


if __name__ == "__main__":
    main()
