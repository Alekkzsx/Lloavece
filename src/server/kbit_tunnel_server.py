import os
import argparse
import torch
import sentencepiece as spm
from src.model.config import LloaveceConfig
from src.model.modeling import LloaveceModel
from src.export.kbit_compressor import convert_model_to_kbit

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
        print(f"[!] Checkpoint não encontrado em {checkpoint_path}. Criando configuração 'test' para inicialização rápida...")
        config = LloaveceConfig.from_name("test")
        model = LloaveceModel(config).to(device)

    # CONVERSÃO CRUCIAL: O servidor roda 100% sobre K-Bit!
    print(f"\n[⚡ K-BIT ATIVO NO TÚNEL] Convertendo todas as projeções para KBitLinear (Rank={rank})...")
    model = convert_model_to_kbit(model, rank=rank)
    model.eval()

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

        generated_text = sp.decode(out_tokens[0].tolist())
        return generated_text

    try:
        import gradio as gr
        from src.workflow.lloavece_workflow import LloaveceWorkflow
        workflow_engine = LloaveceWorkflow()

        print("\n" + "=" * 65)
        print("=== INICIANDO TÚNEL PÚBLICO HTTPS PARA A LLOAVECE K-BIT ===")
        print("=== Motor Ativo: Kronecker Multi-Rank 4 + Resíduo Ternário ===")
        print("=" * 65)

        def run_workflow_fn(task_prompt: str):
            if not task_prompt.strip():
                return "Digite uma tarefa para o fluxo técnico executar."
            return workflow_engine.run(task_prompt)

        with gr.Blocks(title="🇧🇷 Lloavece Coder - K-Bit Multi-Agent Workflow") as demo:
            gr.Markdown(
                "# 🇧🇷 Lloavece Coder - K-Bit Ultra-Lightweight Suite (2026)\n"
                "**Plataforma de Inferência e Fluxo Técnico:** Motor K-Bit Multi-Rank com decomposição de Kronecker.\n"
                "Sem políticas de persona aplicadas nesta versão de testes técnicos."
            )

            with gr.Tab("⚡ Inferência Rápida K-Bit"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(lines=4, label="Instrução / Código", value="def dijkstra_grafo(grafo: dict, origem: str):")
                        max_tokens = gr.Slider(minimum=32, maximum=512, value=128, step=16, label="Máximo de Tokens")
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature")
                        submit_btn = gr.Button("⚡ Gerar Código via K-Bit", variant="primary")
                    with gr.Column():
                        output_code = gr.Code(label="Resposta do Modelo K-Bit", language="python")
                submit_btn.click(generate_fn, inputs=[prompt_input, max_tokens, temperature], outputs=output_code)

            with gr.Tab("⚙️ Fluxo de Trabalho Técnico da Frota (Kimi + Especialistas + DeepSeek-R1)"):
                with gr.Row():
                    with gr.Column():
                        task_input = gr.Textbox(
                            lines=4,
                            label="Especificação da Demanda Técnica",
                            value="Criar um microserviço em Rust com banco de dados PostgreSQL, Docker e auditoria de segurança"
                        )
                        run_btn = gr.Button("🚀 Disparar Fluxo Técnico Completo", variant="primary")
                    with gr.Column():
                        workflow_output = gr.Markdown(label="Relatório & Código Integrado da Frota")
                run_btn.click(run_workflow_fn, inputs=[task_input], outputs=workflow_output)

        demo.launch(server_name="0.0.0.0", server_port=port, share=share)

    except ImportError:
        print("\n[!] Gradio não está instalado. Executando servidor HTTP nativo com túnel...")
        import http.server
        import socketserver

        class KBitHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "online", "engine": "kbit_multi_rank", "version": "2026.1"}')

        with socketserver.TCPServer(("", port), KBitHandler) as httpd:
            print(f"[OK] Servidor HTTP K-Bit ativo na porta {port}!")
            httpd.serve_forever()

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
