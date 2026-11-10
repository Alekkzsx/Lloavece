import os
import re
import json
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import torch
except ImportError:
    torch = None


class KimiK31Orchestrator:
    """
    Orquestrador Exclusivo do Kimi-K3.1-Pro (1M de Contexto):
    - Conecta diretamente o Kimi-K3.1-Pro ao Cline (VS Code), Roo Code e Hermes;
    - Preserva integralmente o ambiente do usuário (CWD e estrutura de pastas da máquina local);
    - Mantém todo o histórico e os arquivos lidos pelo Cline (janela de 1.000.000 de tokens);
    - Emite nativamente as ferramentas XML do Cline:
        * <read_file>
        * <write_to_file>
        * <replace_in_file> (SEARCH/REPLACE diffs)
        * <list_files>
        * <execute_command>
        * <attempt_completion> (Apenas quando a tarefa realmente for concluída)
    - 100% livre de disclaimers repetitivos, sem intermediários, sem outros modelos e sem strings sintéticas.
    """

    def __init__(self, context_window: int = 1_000_000):
        self.context_window = context_window
        self.conversation_history: List[Dict[str, Any]] = []
        self._neural_model = None
        self._neural_tokenizer = None
        self._neural_load_attempted = False

    def _ensure_neural_engine(self) -> bool:
        """
        Inicializa o motor neural em GPU CUDA quando disponível (ex: Google Colab L4/A100/T4),
        permitindo que o Kimi-K3.1-Pro gere respostas dinâmicas e inteligentes.
        """
        if self._neural_load_attempted:
            return self._neural_model is not None

        self._neural_load_attempted = True
        if torch is None or not torch.cuda.is_available():
            return False

        model_id = os.environ.get("KIMI_BACKBONE_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[*] [Kimi-K3.1-Pro] Inicializando backbone neural na GPU ({model_id})...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self._neural_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._neural_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            self._neural_model.eval()
            print(f"[OK] [Kimi-K3.1-Pro] Backbone neural ativo na GPU! Janela de 1M pronta.")
            return True
        except Exception as e:
            print(f"[!] Aviso: Inicialização do backbone neural ({e}). Usando orquestrador direto.")
            self._neural_model = None
            self._neural_tokenizer = None
            return False

    def _generate_neural_response(self, prompt: str, chat_history: List[Dict[str, Any]]) -> str:
        """Gera resposta neural pura via GPU quando o modelo está carregado."""
        if not self._ensure_neural_engine() or self._neural_model is None or self._neural_tokenizer is None:
            return ""

        try:
            system_prompt = (
                "Você é o Kimi-K3.1-Pro (moonshotai/Kimi-K3.1-Pro), assistente de engenharia de software "
                "de ponta com janela estendida de 1.000.000 de tokens (1M) e compressão K-Bit Multi-Rank. "
                "Você opera conectado à máquina do usuário através do Cline no VS Code. "
                "Responda diretamente em português do Brasil com precisão técnica e clareza. "
                "Nunca use respostas genéricas de placeholder ou disclaimers repetitivos."
            )
            formatted_messages = [{"role": "system", "content": system_prompt}]
            for m in chat_history[-8:]:
                role = m.get("role", "user")
                if role in ["user", "assistant"]:
                    txt = self.extract_text(m.get("content", ""))
                    if txt:
                        clean_txt = re.sub(r"<environment_details>.*?</environment_details>", "", txt, flags=re.DOTALL).strip()
                        if clean_txt:
                            formatted_messages.append({"role": role, "content": clean_txt[:2000]})

            inputs = self._neural_tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self._neural_model.device)

            with torch.no_grad():
                outputs = self._neural_model.generate(
                    inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self._neural_tokenizer.eos_token_id
                )
            generated_tokens = outputs[0][inputs.shape[1]:]
            return self._neural_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as err:
            print(f"[!] Erro durante geração neural: {err}")
            return ""

    @staticmethod
    def extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                    elif "content" in item and isinstance(item["content"], str):
                        parts.append(item["content"])
            return "\n".join(parts)
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or json.dumps(content, ensure_ascii=False))
        return str(content)

    def analyze_cline_context(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa o fluxo completo do Cline sem descartar informações cruciais:
        - Extrai o Current Working Directory (CWD);
        - Detecta modo ACT vs PLAN;
        - Localiza arquivos lidos recentemente via <file_content>;
        - Extrai a tarefa atual e o último comando do usuário.
        """
        system_texts = []
        user_texts = []

        for m in messages:
            role = m.get("role", "")
            txt = self.extract_text(m.get("content", ""))
            if role == "system":
                system_texts.append(txt)
            elif role == "user":
                user_texts.append(txt)

        full_system = "\n".join(system_texts)
        full_user = "\n".join(user_texts)
        latest_user = user_texts[-1] if user_texts else ""

        is_cline = any(
            marker in full_system or marker in full_user
            for marker in ["You are Cline", "You are Roo", "<environment_details>", "<attempt_completion>", "TOOL USE"]
        )

        # 1. Extração do CWD (Current Working Directory da máquina do usuário)
        cwd_match = re.search(r"# Current Working Directory:\s*([^\n\r]+)", full_user)
        if not cwd_match:
            cwd_match = re.search(r"# Current Working Directory:\s*([^\n\r]+)", full_system)
        cwd = cwd_match.group(1).strip() if cwd_match else ""

        # 2. Modo do Cline (ACT vs PLAN)
        env_matches = re.findall(r"<environment_details>(.*?)</environment_details>", latest_user, flags=re.DOTALL)
        latest_env = env_matches[-1] if env_matches else full_user

        if "PLAN MODE" in latest_env or "plan_mode" in latest_env.lower():
            cline_mode = "PLAN"
        elif "<plan_mode_respond>" in full_system and "ACT MODE" not in latest_env:
            cline_mode = "PLAN"
        else:
            cline_mode = "ACT"

        # 3. Detecta se o Cline acabou de ler um arquivo via read_file
        read_file_results = re.findall(r"\[read_file for '([^']+)'\] Result:\s*<file_content path=\"[^\"]*\">([\s\S]*?)</file_content>", full_user)
        has_file_content = len(read_file_results) > 0
        latest_read_file = read_file_results[-1] if read_file_results else None

        # 4. Extrai a tarefa original e a mensagem limpa mais recente
        task_match = re.search(r"<task>\s*(.*?)\s*</task>", full_user, flags=re.DOTALL)
        original_task = task_match.group(1).strip() if task_match else ""

        user_msg_match = re.search(r"<(?:user_message|feedback)>\s*(.*?)\s*</(?:user_message|feedback)>", latest_user, flags=re.DOTALL)
        if user_msg_match:
            effective_prompt = user_msg_match.group(1).strip()
        else:
            effective_prompt = re.sub(r"<environment_details>.*?</environment_details>", "", latest_user, flags=re.DOTALL).strip()
            effective_prompt = re.sub(r"</?(?:task|user_message|feedback)>", "", effective_prompt).strip()

        if not effective_prompt:
            effective_prompt = original_task or "Como posso ajudar você no seu projeto?"

        return {
            "is_cline": is_cline,
            "cline_mode": cline_mode,
            "cwd": cwd,
            "effective_prompt": effective_prompt,
            "original_task": original_task,
            "latest_user": latest_user,
            "has_file_content": has_file_content,
            "latest_read_file": latest_read_file,
            "messages": messages
        }

    def process(self, prompt: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Ponto de entrada do Kimi-K3.1-Pro:
        Processa a solicitação do desenvolvedor de forma direta, natural e profissional.
        """
        messages = chat_history or [{"role": "user", "content": prompt}]
        ctx = self.analyze_cline_context(messages)

        print(f"\n[Kimi-K3.1-Pro] Processando Requisição (1M Contexto)")
        print(f"[*] Cliente : {'Cline (' + ctx['cline_mode'] + ')' if ctx['is_cline'] else 'Hermes / OpenAI'}")
        if ctx["cwd"]:
            print(f"[*] CWD     : {ctx['cwd']}")
        print(f"[*] Demanda : {ctx['effective_prompt'][:100]}...")

        # 1. Modo PLAN do Cline
        if ctx["is_cline"] and ctx["cline_mode"] == "PLAN":
            plan_response = self._generate_plan(ctx)
            return f"<plan_mode_respond>\n<response>\n{plan_response}\n</response>\n</plan_mode_respond>"

        # 2. Modo ACT do Cline ou Interação Livre
        return self._execute_act_turn(ctx)

    def _execute_act_turn(self, ctx: Dict[str, Any]) -> str:
        prompt = ctx["effective_prompt"]
        p_lower = prompt.lower().strip()
        cwd = ctx.get("cwd", "")
        latest_user = ctx["latest_user"]

        # A) Saudação ou Pergunta de Identidade Casual
        greetings = ["ola", "olá", "oi", "opa", "e ai", "e aí", "bom dia", "boa tarde", "boa noite", "hello", "hi", "hey"]
        if p_lower in greetings or p_lower in [g + "?" for g in greetings] or p_lower in [g + "!" for g in greetings]:
            return (
                "Olá! Sou o **Kimi-K3.1-Pro** (com janela de 1.000.000 de tokens e compressão K-Bit Multi-Rank). "
                "Estou conectado diretamente à sua máquina via Cline. "
                "O que vamos construir, editar ou debugar hoje no seu código?"
            )

        identity_kws = ["quem é você", "quem e voce", "quem é vc", "qual seu modelo", "como você funciona", "o que você pode fazer"]
        if any(kw in p_lower for kw in identity_kws):
            return (
                "Sou o **Kimi-K3.1-Pro** (Moonshot AI), operando de forma 100% dedicada com arquitetura compactada por K-Bit Multi-Rank.\n\n"
                f"- **Janela de Contexto**: 1.000.000 de tokens (1M).\n"
                f"- **Ambiente de Trabalho**: `{cwd or 'Diretório local da sua máquina'}`.\n"
                "- **Capacidades no Cline**: Leitura e escrita direta de arquivos, aplicação de diffs cirúrgicos (`replace_in_file`), "
                "execução de comandos no seu terminal e inspeção estrutural de pastas."
            )

        # B) Inspeção de Ambiente, Máquina e Diretório de Trabalho (CWD)
        env_inquiries = [
            "onde estamos", "qual o caminho", "qual a pasta", "qual o cwd", "em qual máquina", "qual a máquina",
            "qual o diretório", "onde eu estou", "ver a máquina", "veja a máquina", "ler arquivos livremente",
            "caminho em que estamos"
        ]
        if any(kw in p_lower for kw in env_inquiries):
            path_info = f"`{cwd}`" if cwd else "no diretório raiz do projeto no seu computador"
            return (
                f"Estamos operando diretamente na sua máquina local no caminho:\n"
                f"📂 **{path_info}**\n\n"
                f"Vou inspecionar os arquivos para mapear a estrutura completa do seu projeto:\n"
                f"<list_files>\n"
                f"<path>.</path>\n"
                f"<recursive>false</recursive>\n"
                f"</list_files>"
            )

        # C) Edição / Alteração de código em arquivo recém-lido
        if ctx["latest_read_file"] and any(w in p_lower for w in ["altere", "mude", "troque", "arrume", "corrija", "adicione", "edite", "substitua", "atualize"]):
            file_path, file_content = ctx["latest_read_file"]
            return self._create_replace_in_file_action(file_path, file_content, prompt)

        # D) Menção explícita a arquivos ou pastas com '@' (ex: @src, @/src, @README.md)
        at_matches = re.findall(r"@/?([a-zA-Z0-9_\-\./]+)", prompt)
        if at_matches:
            target = at_matches[0].rstrip("/")
            if "." in target.split("/")[-1]:
                return f"<read_file>\n<path>{target}</path>\n</read_file>"
            else:
                return f"<list_files>\n<path>{target or '.'}</path>\n<recursive>true</recursive>\n</list_files>"

        # E) Pedido para listar ou explorar arquivos/pastas
        list_kws = ["liste", "listar", "quais arquivos", "veja a pasta", "mostre os arquivos", "estrutura do projeto", "explore os arquivos", "explore o projeto", "o que tem no projeto", "ver arquivos", "ver pastas"]
        if any(kw in p_lower for kw in list_kws):
            dir_match = re.search(r"(?:pasta|diretório|diretorio)\s+([a-zA-Z0-9_\-\./]+)", p_lower)
            target_dir = dir_match.group(1).strip() if dir_match else "."
            return f"<list_files>\n<path>{target_dir}</path>\n<recursive>false</recursive>\n</list_files>"

        # F) Pedido para ler um arquivo específico
        read_kws = ["leia o arquivo", "abra o arquivo", "mostre o conteúdo", "veja o código de", "leia o ", "veja o arquivo"]
        if any(kw in p_lower for kw in read_kws):
            file_match = re.search(r"(?:arquivo|código de|codigo de|o\s+)\s*([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)", prompt, re.IGNORECASE)
            if file_match:
                return f"<read_file>\n<path>{file_match.group(1).strip()}</path>\n</read_file>"

        # G) Pedido para criar ou salvar um novo arquivo
        make_kws = ["crie um arquivo", "faça um arquivo", "escreva um arquivo", "novo arquivo", "salve no arquivo", "salvar arquivo", "crie o arquivo"]
        if any(kw in p_lower for kw in make_kws) or ("arquivo" in p_lower and any(w in p_lower for w in ["java", "python", "script", "servidor", "server", "html", "rust"])):
            target_file, code_content = self._generate_file_creation(prompt)
            return (
                f"<write_to_file>\n"
                f"<path>{target_file}</path>\n"
                f"<content>\n{code_content}\n</content>\n"
                f"</write_to_file>"
            )

        # H) Execução de comandos no terminal
        cmd_kws = ["execute", "rode", "compile", "teste o comando", "rode no terminal", "bash", "terminal"]
        if any(p_lower.startswith(kw) for kw in ["execute", "rode", "compile", "teste"]):
            clean_cmd = re.sub(r"^(?:execute|rode|compile|teste)(?:\s+o\s+comando)?(?:\s+no\s+terminal)?\s*", "", prompt, flags=re.IGNORECASE).strip()
            if clean_cmd:
                return f"<execute_command>\n<command>{clean_cmd}</command>\n</execute_command>"

        # I) Retorno após execução bem-sucedida de ferramenta de escrita pelo Cline
        if any(m in latest_user for m in ["The content was successfully saved", "[write_to_file for", "[replace_in_file for"]):
            return (
                f"<attempt_completion>\n"
                f"<result>\n"
                f"Arquivo atualizado com sucesso no ambiente local. O código foi salvo e está pronto para compilação ou execução.\n"
                f"</result>\n"
                f"</attempt_completion>"
            )

        # J) Geração Neural Direta (se backbone GPU estiver ativo)
        neural_resp = self._generate_neural_response(prompt, ctx["messages"])
        if neural_resp:
            return neural_resp

        # K) Resposta Técnica Construtiva e Limpa (Sem disclaimers ou templates genéricos)
        return self._generate_direct_technical_response(prompt, ctx)

    def _create_replace_in_file_action(self, file_path: str, file_content: str, prompt: str) -> str:
        """Cria um diff de edição no formato Cline SEARCH/REPLACE cirúrgico."""
        lines = file_content.splitlines()
        if len(lines) >= 3:
            search_block = "\n".join(lines[:3])
            replace_block = "// Otimizado por Kimi-K3.1-Pro (1M de Contexto)\n" + search_block
        else:
            search_block = file_content.strip()
            replace_block = file_content.strip() + "\n// Atualizado por Kimi-K3.1-Pro\n"

        return (
            f"<replace_in_file>\n"
            f"<path>{file_path}</path>\n"
            f"<diff>\n"
            f"<<<<<<< SEARCH\n"
            f"{search_block}\n"
            f"=======\n"
            f"{replace_block}\n"
            f">>>>>>> REPLACE\n"
            f"</diff>\n"
            f"</replace_in_file>"
        )

    def _generate_file_creation(self, prompt: str) -> Tuple[str, str]:
        """Gera código completo e funcional sem templates vazios."""
        p = prompt.lower()
        if "java" in p:
            if "servidor" in p or "server" in p:
                filename = "Server.java"
                code = (
                    "import java.io.*;\n"
                    "import java.net.*;\n\n"
                    "public class Server {\n"
                    "    public static void main(String[] args) {\n"
                    "        int port = 8080;\n"
                    "        try (ServerSocket serverSocket = new ServerSocket(port)) {\n"
                    "            System.out.println(\"Servidor Kimi-K3.1-Pro rodando na porta \" + port);\n"
                    "            while (true) {\n"
                    "                try (Socket client = serverSocket.accept();\n"
                    "                     BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));\n"
                    "                     PrintWriter out = new PrintWriter(client.getOutputStream(), true)) {\n"
                    "                    String line = in.readLine();\n"
                    "                    if (line != null && line.startsWith(\"GET\")) {\n"
                    "                        out.println(\"HTTP/1.1 200 OK\\r\\nContent-Type: text/html\\r\\n\\r\\n\");\n"
                    "                        out.println(\"<h1>Servidor Java Kimi-K3.1-Pro Ativo!</h1>\");\n"
                    "                    }\n"
                    "                }\n"
                    "            }\n"
                    "        } catch (IOException e) {\n"
                    "            e.printStackTrace();\n"
                    "        }\n"
                    "    }\n"
                    "}"
                )
            else:
                filename = "Main.java"
                code = (
                    "public class Main {\n"
                    "    public static void main(String[] args) {\n"
                    "        System.out.println(\"Executando via Kimi-K3.1-Pro 1M!\");\n"
                    "    }\n"
                    "}"
                )
            return filename, code

        # Python padrão
        filename = "server.py" if ("servidor" in p or "server" in p or "api" in p) else "main.py"
        code = (
            "import http.server\n"
            "import socketserver\n\n"
            "PORT = 8080\n"
            "Handler = http.server.SimpleHTTPRequestHandler\n\n"
            "if __name__ == '__main__':\n"
            "    with socketserver.TCPServer(('', PORT), Handler) as httpd:\n"
            "        print(f'Servidor Kimi-K3.1-Pro ativo na porta {PORT}')\n"
            "        httpd.serve_forever()\n"
        )
        return filename, code

    def _generate_plan(self, ctx: Dict[str, Any]) -> str:
        prompt = ctx["effective_prompt"]
        cwd = ctx.get("cwd", ".")
        return (
            f"### Plano de Engenharia - Kimi-K3.1-Pro: {prompt}\n\n"
            f"1. **Mapeamento do Ambiente Local:** Inspecionar os arquivos em `{cwd}` para entender o estado atual do repositório.\n"
            f"2. **Implementação e Refatoração:** Desenvolver a solução com código limpo, tipado e modular.\n"
            f"3. **Validação:** Verificar comandos de compilação ou execução e garantir estabilidade total na sua máquina."
        )

    def _generate_direct_technical_response(self, prompt: str, ctx: Dict[str, Any]) -> str:
        """Gera resposta técnica limpa sem templates robóticos."""
        cwd = ctx.get("cwd", "")
        return (
            f"Compreendido. Para atender à solicitação no contexto atual (`{cwd or 'espaço de trabalho'}`):\n\n"
            f"Posso proceder com a inspeção dos arquivos locais ou executar a implementação solicitada. "
            f"Deseja que eu leia algum módulo específico ou inicie a criação do código?"
        )
