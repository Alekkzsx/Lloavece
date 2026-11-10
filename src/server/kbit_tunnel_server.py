import os
import sys
import time
import uuid
import json
import asyncio
import argparse
import subprocess
import torch
import sentencepiece as spm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel
from src.export.kbit_compressor import convert_model_to_kbit
from src.workflow.lloavece_workflow import LloaveceWorkflow

# Inicialização do FastAPI
app = FastAPI(
    title="Lloavece Coder - Servidor Hermes & Cline API",
    description="Servidor de IA com suporte nativo a OpenAI/Hermes/Cline, contexto de 1M e Batalhão dos 5 Titãs",
    version="2026.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado Global do Servidor
server_state = {
    "model": None,
    "tokenizer": None,
    "device": "cpu",
    "workflow": LloaveceWorkflow()
}

class ChatMessage(BaseModel):
    role: str
    content: Union[str, Any]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "lloavece-coder"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.get("/")
async def root():
    return JSONResponse({
        "status": "online",
        "service": "Lloavece Coder API (Hermes & Cline Compatible)",
        "engine": "K-Bit Multi-Rank 4 (Kronecker)",
        "context_window": "1,000,000 tokens",
        "titans": ["Kimi-K2.5 (Voz Principal)", "Qwen3.8-2.4T", "GLM-5.3", "MiniMax-M3", "DeepSeek-R1"]
    })

@app.get("/v1/models")
async def list_models():
    """Endpoint oficial para descoberta de modelos pelo Cline, Roo Code e Hermes."""
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "lloavece-coder",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "lloavece-ai",
                "permission": [],
                "root": "lloavece-coder"
            },
            {
                "id": "hermes-lloavece",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "lloavece-ai",
                "permission": [],
                "root": "hermes-lloavece"
            },
            {
                "id": "lloavece-5titans-workflow",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "lloavece-ai",
                "permission": [],
                "root": "lloavece-5titans-workflow"
            }
        ]
    })

async def stream_generator(content_text: str, model_name: str):
    """Gera stream SSE (Server-Sent Events) estritamente no padrão OpenAI para o Cline."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # Divide o texto em pequenos chunks para simular streaming natural
    words = content_text.split(" ")
    for idx, word in enumerate(words):
        chunk_text = word if idx == len(words) - 1 else word + " "
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.015)

    # Chunk final
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint principal consumido pelo Cline.
    Suporta streaming (SSE) e requisições normais (JSON).
    Mantém contexto de até 1.000.000 de tokens com Kimi na liderança.
    """
    # Extrai o último prompt do usuário
    user_prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user" and isinstance(msg.content, str):
            user_prompt = msg.content
            break

    if not user_prompt:
        user_prompt = "Olá"

    # Executa a geração através do fluxo inteligente
    workflow = server_state["workflow"]
    
    # Converte histórico para contexto contínuo
    history_dicts = [{"role": m.role, "content": str(m.content)} for m in request.messages]
    result_content = workflow.run(user_prompt, chat_history=history_dicts)

    # Se o Cline pedir STREAMING (padrão do Cline!)
    if request.stream:
        return StreamingResponse(
            stream_generator(result_content, request.model or "lloavece-coder"),
            media_type="text/event-stream"
        )

    # Retorno padrão JSON (Non-streaming)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = len(user_prompt.split())
    completion_tokens = len(result_content.split())

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or "lloavece-coder",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })

def start_cloudflared_tunnel(port: int = 7860) -> Optional[str]:
    """Inicia um túnel Cloudflare seguro e gratuito direto para a API FastAPI."""
    try:
        # Verifica se o binário cloudflared existe
        if not os.path.exists("./cloudflared") and not shutil.which("cloudflared"):
            print("[*] Baixando binário Cloudflared para túnel seguro do Cline...")
            subprocess.run(
                "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared",
                shell=True, check=True
            )

        cmd = f"./cloudflared tunnel --url http://127.0.0.1:{port}" if os.path.exists("./cloudflared") else f"cloudflared tunnel --url http://127.0.0.1:{port}"
        print(f"[*] Subindo túnel Cloudflared na porta {port}...")
        log_file = open("/content/cloudflared_tunnel.log", "w") if os.path.exists("/content") else open("cloudflared_tunnel.log", "w")
        
        proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
        time.sleep(3)

        # Lê a URL gerada
        log_path = "/content/cloudflared_tunnel.log" if os.path.exists("/content") else "cloudflared_tunnel.log"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                content = f.read()
                import re
                urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if urls:
                    return urls[0]
        return None
    except Exception as e:
        print(f"[!] Aviso Cloudflared: {e}")
        return None

def start_server(port: int = 7860):
    import uvicorn
    import shutil

    # Tenta subir túnel Cloudflare
    cf_url = start_cloudflared_tunnel(port)
    if cf_url:
        print("\n" + "=" * 75)
        print("🌐 [TÚNEL CLOUDFLARE ATIVO PARA O CLINE NO VS CODE]")
        print(f"👉 Base URL para o Cline:  {cf_url}/v1")
        print(f"👉 Model ID para o Cline:   lloavece-coder")
        print("=" * 75 + "\n")

    # Inicia Uvicorn servindo a FastAPI diretamente
    print(f"[*] Servidor FastAPI iniciando na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    start_server(port=args.port)
