import os
import re
import sys
import time
import uuid
import json
import asyncio
import argparse
import subprocess
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from kimi_k31_pro.orchestrator import KimiK31Orchestrator

MAX_CONTEXT = 1_000_000
MODEL_ID = "moonshotai/Kimi-K3.1-Pro"

app = FastAPI(
    title="Kimi-K3.1-Pro (1M de Contexto) - Servidor Dedicado K-Bit",
    description="Servidor de IA com 1.000.000 de tokens de contexto (1M) dedicado exclusivamente ao Kimi-K3.1-Pro com compressão K-Bit Multi-Rank.",
    version="2026.3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = KimiK31Orchestrator(context_window=MAX_CONTEXT)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[Any], Dict[str, Any]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = MODEL_ID
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
    model: Optional[str] = MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


def _build_model_card(model_id: str) -> Dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "moonshotai",
        "permission": [],
        "root": model_id,
        "context_length": MAX_CONTEXT,
        "max_model_len": MAX_CONTEXT,
        "context_window": MAX_CONTEXT,
        "max_context_length": MAX_CONTEXT,
        "max_completion_tokens": 65536,
        "supports_tools": True,
        "supports_prompt_cache": True,
        "description": "Kimi-K3.1-Pro 1M - Modelo Flagship da Moonshot AI com Compressão K-Bit Multi-Rank"
    }


@app.get("/")
@app.get("/health")
async def health():
    return JSONResponse({
        "status": "online",
        "model": MODEL_ID,
        "context_window": MAX_CONTEXT,
        "context_window_formatted": "1,000,000 tokens (1M)",
        "engine": "K-Bit Multi-Rank Kronecker (Exclusivo Kimi-K3.1-Pro)",
        "compatible_clients": ["Cline (VS Code)", "Roo Code", "Hermes", "OpenAI CLI"]
    })


@app.get("/v1/models")
async def list_models():
    models = [
        _build_model_card(MODEL_ID),
        _build_model_card("kimi-k3.1-pro"),
        _build_model_card("default")
    ]
    return JSONResponse({"object": "list", "data": models})


@app.get("/v1/models/{model_name:path}")
async def retrieve_model(model_name: str):
    return JSONResponse(_build_model_card(model_name))


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    req_id = f"chatcmpl-kimi-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    dict_messages = [m.model_dump() for m in req.messages]
    last_user_prompt = ""
    for m in reversed(dict_messages):
        if m.get("role") == "user":
            last_user_prompt = orchestrator.extract_text(m.get("content", ""))
            break

    # Processa pelo orquestrador do Kimi-K3.1-Pro
    loop = asyncio.get_event_loop()
    reply_content = await loop.run_in_executor(
        None, orchestrator.process, last_user_prompt, dict_messages
    )

    if req.stream:
        async def stream_generator():
            # Chunk inicial zero-latency para evitar timeouts de tunnel
            ping_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model or MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(ping_chunk)}\n\n"

            # Stream do conteúdo
            chunk_size = 64
            for i in range(0, len(reply_content), chunk_size):
                sub = reply_content[i:i + chunk_size]
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model or MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": sub},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)

            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model or MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    return JSONResponse({
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": reply_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": max(1, len(last_user_prompt) // 4),
            "completion_tokens": max(1, len(reply_content) // 4),
            "total_tokens": (len(last_user_prompt) + len(reply_content)) // 4,
            "context_window": MAX_CONTEXT
        }
    })


def start_cloudflared_tunnel(port: int = 8000) -> Optional[str]:
    """Inicia o túnel Cloudflared e extrai a URL pública HTTPS."""
    cloudflared_bin = "cloudflared"
    if not shutil.which(cloudflared_bin):
        if os.path.exists("/usr/local/bin/cloudflared"):
            cloudflared_bin = "/usr/local/bin/cloudflared"
        else:
            print("[*] Instalando binário cloudflared...")
            subprocess.run([
                "curl", "-sL",
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "-o", "/usr/local/bin/cloudflared"
            ], check=False)
            subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=False)
            cloudflared_bin = "/usr/local/bin/cloudflared"

    cmd = [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    print(f"[*] Iniciando túnel Cloudflared para Kimi-K3.1-Pro na porta {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    tunnel_url = None
    start_t = time.time()
    while time.time() - start_t < 25:
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.5)
            continue
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group(0)
            break

    if tunnel_url:
        print("\n" + "=" * 78)
        print("🚀 [TÚNEL ATIVO] Kimi-K3.1-Pro Online!")
        print(f"👉 URL do Túnel: {tunnel_url}/v1")
        print(f"👉 Modelo ID   : {MODEL_ID}")
        print(f"👉 Contexto    : {MAX_CONTEXT:,} tokens (1M)")
        print("=" * 78 + "\n")
    else:
        print("[!] Aviso: Não foi possível obter a URL do túnel em 25s.")

    return tunnel_url


if __name__ == "__main__":
    import uvicorn
    import shutil

    parser = argparse.ArgumentParser(description="Servidor Kimi-K3.1-Pro 1M K-Bit")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tunnel", action="store_true", default=True)
    args = parser.parse_args()

    if args.tunnel:
        start_cloudflared_tunnel(port=args.port)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
