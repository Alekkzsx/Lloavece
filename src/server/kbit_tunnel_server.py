import os
import re
import sys
import time
import uuid
import json
import shutil
import asyncio
import argparse
import subprocess

try:
    import torch
except ImportError:
    torch = None
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any, Union

from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel
from src.export.kbit_compressor import convert_model_to_kbit
from src.workflow.lloavece_workflow import LloaveceWorkflow

MAX_CONTEXT_WINDOW = 1_000_000  # 1M Tokens de Contexto (1.000.000)

# Inicialização do FastAPI com suporte completo a Cline, Roo Code, Hermes e OpenAI
app = FastAPI(
    title="Lloavece Coder 1M - Fluxo Livre dos 5 Modelos (Cline & Hermes API)",
    description=(
        "Servidor de IA com 1.000.000 de tokens de contexto (1M), onde os 5 modelos "
        "(Kimi-K2.5, Qwen3.8-2.4T, GLM-5.3, MiniMax-M3 e DeepSeek-R1) estão 100% livres, "
        "conversam entre si com Aval Mútuo (5/5) e o Kimi-K2.5 fala diretamente com o usuário."
    ),
    version="2026.2"
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
    "device": "cuda" if (torch and torch.cuda.is_available()) else "cpu",
    "context_window": MAX_CONTEXT_WINDOW,
    "workflow": LloaveceWorkflow()
}

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[Any], Dict[str, Any]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = "lloavece-coder"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 16384
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stop: Optional[Union[str, List[str]]] = None

class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = "lloavece-coder"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

def _build_model_card(model_id: str, description: str) -> Dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "lloavece-ai",
        "permission": [],
        "root": model_id,
        "context_length": MAX_CONTEXT_WINDOW,
        "max_model_len": MAX_CONTEXT_WINDOW,
        "context_window": MAX_CONTEXT_WINDOW,
        "max_context_length": MAX_CONTEXT_WINDOW,
        "max_completion_tokens": 65536,
        "supports_tools": True,
        "supports_prompt_cache": True,
        "description": description
    }

@app.get("/")
@app.get("/health")
async def root():
    return JSONResponse({
        "status": "online",
        "service": "Lloavece Coder 1M API (Cline, Roo Code & Hermes Compatible)",
        "engine": "K-Bit Multi-Rank 4 (Kronecker) + Fluxo Livre com Aval Mútuo",
        "context_window": MAX_CONTEXT_WINDOW,
        "context_window_formatted": "1,000,000 tokens (1M)",
        "user_voice": "moonshotai/Kimi-K2.5 (Fala com o usuário e participa livremente do fluxo)",
        "free_models": [
            "moonshotai/Kimi-K2.5 (Livre + Voz do Usuário)",
            "Qwen/Qwen3.8-2.4T-A95B (Livre)",
            "zai-org/GLM-5.3 (Livre)",
            "MiniMaxAI/MiniMax-M3 (Livre)",
            "deepseek-ai/DeepSeek-R1 (Livre)"
        ],
        "consensus_rule": "Conversa livre entre os 5 modelos com Aval Mútuo (5/5) obrigatório antes de liberar o resultado"
    })

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Endpoint oficial para descoberta de modelos (1M de contexto) pelo Cline, Roo Code e Hermes."""
    return JSONResponse({
        "object": "list",
        "data": [
            _build_model_card(
                "lloavece-coder",
                "Lloavece Coder 1M - 5 Modelos Livres com Aval Mútuo e Kimi-K2.5 na Voz Principal"
            ),
            _build_model_card(
                "kimi-k2.5",
                "Kimi-K2.5 (1M Contexto) - Interlocutor Oficial & Participante do Fluxo Livre Lloavece"
            ),
            _build_model_card(
                "hermes-lloavece",
                "Hermes Lloavece 1M - Agente Autônomo Compatível com Hermes & OpenAI Function Calling"
            ),
            _build_model_card(
                "lloavece-5titans-workflow",
                "Fluxo Livre dos 5 Modelos com Aval Mútuo Unânime (1M Tokens)"
            )
        ]
    })

async def stream_generator(
    user_prompt: str,
    history_dicts: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    model_name: str
):
    """
    Gera stream SSE (Server-Sent Events) estritamente no padrão OpenAI para o Cline e Hermes.
    Envia o primeiro pacote imediatamente (<10ms) para evitar timeout do Cloudflared/Cline
    enquanto os 5 modelos conversam e aprovam o resultado na GPU.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # 1. Envia imediatamente o chunk de abertura com role='assistant' para manter o túnel ativo
    opening_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(opening_chunk, ensure_ascii=False)}\n\n"

    # 2. Executa o fluxo livre dos 5 modelos em thread assíncrona sem bloquear o event loop
    workflow: LloaveceWorkflow = server_state["workflow"]
    content_text = await asyncio.to_thread(
        workflow.run,
        user_prompt,
        history_dicts,
        tools
    )

    # 3. Transmite o conteúdo aprovado em blocos fluidos e rápidos
    chunk_size = 48
    for i in range(0, len(content_text), chunk_size):
        piece = content_text[i:i + chunk_size]
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.004)

    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in history_dicts)
    completion_tokens = len(content_text.split())

    # 4. Chunk final com finish_reason='stop' e métricas de uso (para barra de 1M de contexto do Cline)
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
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "context_window": MAX_CONTEXT_WINDOW
        }
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint principal consumido pelo Cline, Roo Code e Hermes.
    Suporta mensagens em string ou lista de blocos multimodais/texto (padrão do Cline),
    streaming SSE rápido e contexto de até 1.000.000 de tokens (1M).
    """
    workflow: LloaveceWorkflow = server_state["workflow"]

    # Converte o histórico extraindo texto limpo de blocos [{'type': 'text', 'text': '...'}] do Cline
    history_dicts: List[Dict[str, Any]] = []
    user_prompt = ""

    for msg in request.messages:
        extracted_text = workflow.extract_text_from_content(msg.content)
        history_dicts.append({
            "role": msg.role,
            "content": extracted_text
        })
        if msg.role == "user" and extracted_text.strip():
            user_prompt = extracted_text

    if not user_prompt:
        user_prompt = "Olá"

    model_name = request.model or "lloavece-coder"

    # Se o cliente (Cline / Hermes) pedir STREAMING (SSE)
    if request.stream:
        return StreamingResponse(
            stream_generator(user_prompt, history_dicts, request.tools, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Modo JSON padrão (Non-streaming)
    result_content = await asyncio.to_thread(
        workflow.run,
        user_prompt,
        history_dicts,
        request.tools
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = sum(len(m["content"].split()) for m in history_dicts)
    completion_tokens = len(result_content.split())

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
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
            "total_tokens": prompt_tokens + completion_tokens,
            "context_window": MAX_CONTEXT_WINDOW
        }
    })

@app.post("/v1/completions")
@app.post("/completions")
async def legacy_completions(request: CompletionRequest):
    """Suporte adicional para clientes que usam /v1/completions (FIM / Autocomplete / Hermes)."""
    workflow: LloaveceWorkflow = server_state["workflow"]
    prompt_str = request.prompt[0] if isinstance(request.prompt, list) else str(request.prompt)
    result_content = await asyncio.to_thread(workflow.run, prompt_str)

    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model or "lloavece-coder",
        "choices": [
            {
                "text": result_content,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ]
    })

def start_cloudflared_tunnel(port: int = 7860) -> Optional[str]:
    """Inicia ou reutiliza um túnel Cloudflare seguro e gratuito direto para a API FastAPI."""
    try:
        log_path = "/content/cloudflared_tunnel.log" if os.path.exists("/content") else "cloudflared_tunnel.log"
        # Verifica se já existe um túnel ativo rodando com URL válida
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                existing_urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
                if existing_urls:
                    return existing_urls[-1]

        if not os.path.exists("./cloudflared") and not shutil.which("cloudflared"):
            print("[*] Baixando binário Cloudflared para túnel seguro do Cline/Hermes...")
            subprocess.run(
                "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared",
                shell=True, check=True
            )

        cmd = f"./cloudflared tunnel --url http://127.0.0.1:{port}" if os.path.exists("./cloudflared") else f"cloudflared tunnel --url http://127.0.0.1:{port}"
        print(f"[*] Subindo túnel Cloudflared na porta {port}...")
        log_file = open(log_path, "w")
        subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)

        for _ in range(12):
            time.sleep(1)
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
                    if urls:
                        return urls[-1]
        return None
    except Exception as e:
        print(f"[!] Aviso Cloudflared: {e}")
        return None

def start_server(port: int = 7860, preload_gpu: bool = True):
    import uvicorn

    if preload_gpu and torch and torch.cuda.is_available():
        server_state["workflow"]._ensure_neural_engine()

    cf_url = start_cloudflared_tunnel(port)
    if cf_url:
        print("\n" + "=" * 78)
        print("🌐 [TÚNEL CLOUDFLARE ATIVO PARA O CLINE & HERMES - 1M DE CONTEXTO]")
        print(f"👉 Base URL (Cline / Hermes):  {cf_url}/v1")
        print(f"👉 Model ID (Cline / Hermes):  lloavece-coder")
        print(f"👉 Context Window:             1,000,000 tokens (1M)")
        print(f"👉 Voz do Usuário:             Kimi-K2.5 (com os 5 modelos livres e aval mútuo)")
        print("=" * 78 + "\n")

    print(f"[*] Servidor FastAPI Lloavece 1M iniciando na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no_preload_gpu", action="store_true")
    args = parser.parse_args()

    start_server(port=args.port, preload_gpu=not args.no_preload_gpu)
