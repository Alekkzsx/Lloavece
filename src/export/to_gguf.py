import os
import argparse

def generate_ollama_modelfile(
    model_name: str = "lloavece",
    gguf_path: str = "lloavece-coder.gguf",
    system_prompt: str = (
        "Você é a Lloavece-Coder, uma inteligência artificial especialista em desenvolvimento de software, "
        "revisão de código e arquitetura de sistemas com alma brasileira. Você escreve código limpo, eficiente, "
        "testado e documentado, e explica problemas técnicos em português claro e objetivo."
    )
):
    modelfile_content = f"""FROM ./{gguf_path}

# Template de ChatML da Lloavece
TEMPLATE \"\"\"<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

# Parâmetros de Inferência de Código
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|eos|>"
"""
    modelfile_path = "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"[*] Modelfile gerado com sucesso em: {modelfile_path}")
    print(f"\nPara rodar no Ollama:")
    print(f"  ollama create {model_name} -f Modelfile")
    print(f"  ollama run {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, default="lloavece-coder.gguf")
    parser.add_argument("--name", type=str, default="lloavece")
    args = parser.parse_args()

    generate_ollama_modelfile(model_name=args.name, gguf_path=args.gguf)
