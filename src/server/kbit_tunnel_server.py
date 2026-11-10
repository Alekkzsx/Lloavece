import os
import time
import uuid
import argparse
import torch
import sentencepiece as spm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel
from src.export.kbit_compressor import convert_model_to_kbit
from src.workflow.lloavece_workflow import LloaveceWorkflow

# Inicializa FastAPI para compatibilidade nativa com Hermes, Cline e OpenAI
app = FastAPI(
    title="Lloavece Coder - Hermes & OpenAI Compatible Server",
    description="Servidor de Inferência e Fluxo dos 5 Titãs com endpoints compatíveis com Hermes e Cline",
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
    "workflow": None
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "lloavece-coder"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

@app.get("/v1/models")
async def list_models():
    """Endpoint compatível com OpenAI/Hermes para descoberta de modelos."""
    return {
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
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint padrão OpenAI/Hermes utilizado pelo Cline, Roo Code e agentes autônomos.
    Permite inferência direta ou acionamento do fluxo do Batalhão dos 5 Titãs.
    """
    # Extrai o último prompt do usuário
    user_prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_prompt = msg.content
            break
    
    if not user_prompt:
        user_prompt = request.messages[-1].content if request.messages else "Olá"

    # Se o modelo solicitado for o workflow ou contiver a tag de workflow
    if "workflow" in request.model.lower() or "[workflow]" in user_prompt.lower():
        result_content = server_state["workflow"].run(user_prompt.replace("[workflow]", "").strip())
    else:
        # Inferência direta sobre a rede neural K-Bit
        model = server_state["model"]
        sp = server_state["tokenizer"]
        device = server_state["device"]

        # Formata no estilo ChatML / Hermes: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        hermes_formatted = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokens = sp.encode(hermes_formatted)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        eos_id = sp.piece_to_id("<|eos|>")

        with torch.no_grad():
            out_tokens = model.generate(
                input_tensor,
                max_new_tokens=int(request.max_tokens or 256),
                temperature=float(request.temperature or 0.2),
                top_p=0.95,
                eos_id=eos_id
            )

        result_content = sp.decode(out_tokens[0].tolist())

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = len(user_prompt.split())
    completion_tokens = len(result_content.split())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
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
    }

def load_kbit_lloavece(checkpoint_path: str, tokenizer_path: str, rank: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Inicializando Servidor K-Bit em {device.upper()}...")
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = LloaveceModel(config).to(device)
        model.load_state_dict(ckpt["model"])
        print("[OK] Checkpoint carregado com sucesso!")
    else:
        print(f"[!] Checkpoint não encontrado em {checkpoint_path}. Criando configuração 'test'...")
        config = LloaveceConfig.from_name("test")
        model = LloaveceModel(config).to(device)

    print(f"\n[⚡ K-BIT ATIVO NO TÚNEL] Convertendo camadas para KBitLinear (Rank={rank})...")
    model = convert_model_to_kbit(model, rank=rank)
    model.eval()

    server_state["model"] = model
    server_state["tokenizer"] = sp
    server_state["device"] = device
    server_state["workflow"] = LloaveceWorkflow()

    return model, sp, device

def start_tunnel_server(
    checkpoint_path: str = "checkpoints/best_model.pt",
    tokenizer_path: str = "data/tokenized/lloavece_code_bpe.model",
    port: int = 7860,
    share: bool = True
):
    model, sp, device = load_kbit_lloavece(checkpoint_path, tokenizer_path)

    def generate_fn(prompt: str, max_tokens: int = 128, temperature: float = 0.2):
        if not prompt.strip():
            return "Por favor, digite uma instrução ou código para a Lloavece Coder."
        
        tokens = sp.encode(prompt)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        eos_id = sp.piece_to_id("<|eos|>")

        with torch.no_grad():
            out_tokens = model.generate(
                input_tensor,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=0.95,
                eos_id=eos_id
            )

        return sp.decode(out_tokens[0].tolist())

    def workflow_fn(task_prompt: str):
        if not task_prompt.strip():
            return "Digite uma tarefa para o batalhão dos 5 titãs processar."
        return server_state["workflow"].run(task_prompt)

    try:
        import gradio as gr

        with gr.Blocks(title="🇧🇷 Lloavece Coder - Batalhão dos 5 Titãs & Hermes API") as demo:
            gr.Markdown(
                "# 🇧🇷 Lloavece Coder - Batalhão dos 5 Titãs & Hermes/OpenAI API\n"
                "- **API Compatível com Hermes & Cline:** Conecte o Cline no VS Code em `https://[url-do-tunel]/v1`\n"
                "- **Motor K-Bit Multi-Rank:** Inferência 100% sobre matrizes de Kronecker.\n"
                "- **Fluxo Cooperativo dos 5 Titãs:** Kimi-K2.5 $\\to$ Qwen3.8 $\\to$ GLM-5.3 $\\to$ MiniMax-M3 $\\to$ DeepSeek-R1."
            )

            with gr.Tab("⚡ Inferência Rápida K-Bit"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(lines=4, label="Prompt / Instrução", value="def quicksort_algoritmo(arr: list) -> list:")
                        max_tokens = gr.Slider(minimum=32, maximum=512, value=128, step=16, label="Máximo de Tokens")
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature")
                        submit_btn = gr.Button("⚡ Gerar Código K-Bit", variant="primary")
                    with gr.Column():
                        output_code = gr.Code(label="Resposta do Modelo K-Bit", language="python")
                submit_btn.click(generate_fn, inputs=[prompt_input, max_tokens, temperature], outputs=output_code)

            with gr.Tab("🤝 Batalhão dos 5 Titãs (Refinamento Cooperativo)"):
                with gr.Row():
                    with gr.Column():
                        task_input = gr.Textbox(
                            lines=4,
                            label="Demanda Técnica Complexa",
                            value="Desenvolver um motor de execução assíncrono com validação criptográfica e contenção de concorrência"
                        )
                        run_btn = gr.Button("🚀 Disparar Ping-Pong dos 5 Titãs", variant="primary")
                    with gr.Column():
                        workflow_output = gr.Markdown(label="Solução Consolidada e Blindada")
                run_btn.click(workflow_fn, inputs=[task_input], outputs=workflow_output)

        print("\n" + "=" * 75)
        print("=== INICIANDO TÚNEL PÚBLICO HTTPS (GRADIO + HERMES/CLINE API) ===")
        print("=== Endpoints OpenAI Ativos em: /v1/chat/completions e /v1/models ===")
        print("=" * 75)

        # Monta a aplicação Gradio diretamente no FastAPI
        app_mounted = gr.mount_gradio_app(app, demo, path="/")
        
        # Inicia com uvicorn e tunnel gradio
        demo.launch(server_name="0.0.0.0", server_port=port, share=share)

    except ImportError:
        print("[!] Gradio não encontrado. Executando uvicorn puro...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/lloavece_checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenized/lloavece_code_bpe.model")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no_share", action="store_true")
    args = parser.parse_args()

    start_tunnel_server(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        port=args.port,
        share=not args.no_share
    )
