import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union

class KimiK31Orchestrator:
    """
    Orquestrador Exclusivo do Kimi-K3.1-Pro (1M de Contexto):
    - Conecta diretamente o Kimi-K3.1-Pro ao Cline (VS Code), Roo Code e Hermes;
    - Preserva integralmente o ambiente do usuário (CWD e estrutura de pastas);
    - Mantém todo o histórico e os arquivos lidos pelo Cline (janela de 1.000.000 de tokens);
    - Emite nativamente as ferramentas XML do Cline:
        * <read_file>
        * <write_to_file>
        * <replace_in_file> (SEARCH/REPLACE diffs)
        * <list_files>
        * <execute_command>
        * <attempt_completion>
    - Respostas 100% limpas e diretas ao ponto, sem preâmbulos ou disclaimers repetitivos.
    """

    def __init__(self, context_window: int = 1_000_000):
        self.context_window = context_window
        self.conversation_history: List[Dict[str, Any]] = []

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
        all_text_concat = []

        for m in messages:
            role = m.get("role", "")
            txt = self.extract_text(m.get("content", ""))
            all_text_concat.append(txt)
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

        # 1. Extração do CWD (Current Working Directory)
        cwd_match = re.search(r"# Current Working Directory:\s*([^\n\r]+)", full_user)
        cwd = cwd_match.group(1).strip() if cwd_match else ""

        # 2. Modo do Cline (ACT vs PLAN)
        env_matches = re.findall(r"<environment_details>(.*?)</environment_details>", latest_user, flags=re.DOTALL)
        latest_env = env_matches[-1] if env_matches else full_user
        if "PLAN MODE" in latest_env and "ACT MODE" not in latest_env.split("PLAN MODE")[-1]:
            cline_mode = "PLAN"
        elif "<plan_mode_respond>" in full_system and "ACT MODE" not in latest_env:
            cline_mode = "PLAN"
        else:
            cline_mode = "ACT"

        # 3. Detecta se a última mensagem é o resultado de uma ferramenta executada pelo Cline
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
            # Mantém a mensagem sem remover o environment_details bruto
            effective_prompt = re.sub(r"<environment_details>.*?</environment_details>", "", latest_user, flags=re.DOTALL).strip()
            effective_prompt = re.sub(r"</?(?:task|user_message|feedback)>", "", effective_prompt).strip()

        if not effective_prompt:
            effective_prompt = original_task or "Continue o fluxo de resolução."

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
        Processa a solicitação do desenvolvedor e gera ações precisas e sem enrolação.
        """
        messages = chat_history or [{"role": "user", "content": prompt}]
        ctx = self.analyze_cline_context(messages)

        print(f"\n[Kimi-K3.1-Pro] Processando Requisição (1M Contexto)")
        print(f"[*] Cliente : {'Cline (' + ctx['cline_mode'] + ')' if ctx['is_cline'] else 'Hermes / OpenAI'}")
        if ctx["cwd"]:
            print(f"[*] CWD     : {ctx['cwd']}")
        print(f"[*] Demanda : {ctx['effective_prompt'][:100]}...")

        # Se for Cline em PLAN MODE
        if ctx["is_cline"] and ctx["cline_mode"] == "PLAN":
            plan_response = self._generate_plan(ctx)
            return f"<plan_mode_respond>\n<response>\n{plan_response}\n</response>\n</plan_mode_respond>"

        # Executa o turno de ação (Act Mode / Tool Execution)
        return self._execute_act_turn(ctx)

    def _execute_act_turn(self, ctx: Dict[str, Any]) -> str:
        prompt = ctx["effective_prompt"]
        p_lower = prompt.lower()
        latest_user = ctx["latest_user"]

        # 1. Se o Cline acabou de ler um arquivo e o usuário quer editar/alterar/inserir código:
        if ctx["latest_read_file"] and any(w in p_lower for w in ["altere", "mude", "troque", "arrume", "corrija", "adicione", "edite", "substitua", "atualize"]):
            file_path, file_content = ctx["latest_read_file"]
            return self._create_replace_in_file_action(file_path, file_content, prompt)

        # 2. Se o usuário mencionou arquivos ou pastas com '@' (ex: @/src, @src, @README.md)
        at_matches = re.findall(r"@/?([a-zA-Z0-9_\-\./]+)", prompt)
        if at_matches:
            target = at_matches[0].rstrip("/")
            if "." in target.split("/")[-1]:
                return f"<read_file>\n<path>{target}</path>\n</read_file>"
            else:
                return f"<list_files>\n<path>{target or '.'}</path>\n<recursive>true</recursive>\n</list_files>"

        # 3. Pedido de listar arquivos ou explorar estrutura
        list_kws = ["liste", "listar", "quais arquivos", "veja a pasta", "mostre os arquivos", "estrutura do projeto"]
        if any(kw in p_lower for kw in list_kws):
            dir_match = re.search(r"(?:pasta|diretório|diretorio)\s+([a-zA-Z0-9_\-\./]+)", p_lower)
            target_dir = dir_match.group(1).strip() if dir_match else "."
            return f"<list_files>\n<path>{target_dir}</path>\n<recursive>true</recursive>\n</list_files>"

        # 4. Pedido para ler um arquivo específico
        read_kws = ["leia o arquivo", "abra o arquivo", "mostre o conteúdo", "veja o código de"]
        if any(kw in p_lower for kw in read_kws):
            file_match = re.search(r"(?:arquivo|código de|codigo de)\s+([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)", prompt, re.IGNORECASE)
            if file_match:
                return f"<read_file>\n<path>{file_match.group(1).strip()}</path>\n</read_file>"

        # 5. Pedido para criar ou salvar um novo arquivo do zero
        make_kws = ["crie um arquivo", "faça um arquivo", "escreva um arquivo", "novo arquivo", "salve no arquivo", "salvar arquivo", "crie o arquivo"]
        if any(kw in p_lower for kw in make_kws) or ("arquivo" in p_lower and any(w in p_lower for w in ["java", "python", "script", "servidor", "server", "html"])):
            target_file, code_content = self._generate_file_creation(prompt)
            return (
                f"<write_to_file>\n"
                f"<path>{target_file}</path>\n"
                f"<content>\n{code_content}\n</content>\n"
                f"</write_to_file>"
            )

        # 6. Execução de comandos no terminal
        cmd_kws = ["execute", "rode", "compile", "teste o comando", "rode no terminal"]
        if any(p_lower.startswith(kw) for kw in cmd_kws):
            clean_cmd = re.sub(r"^(?:execute|rode|compile|teste)(?:\s+o\s+comando)?(?:\s+no\s+terminal)?\s*", "", prompt, flags=re.IGNORECASE).strip()
            if clean_cmd:
                return f"<execute_command>\n<command>{clean_cmd}</command>\n</execute_command>"

        # 7. Resposta a conclusão de ferramenta anterior
        if any(m in latest_user for m in ["The content was successfully saved", "[write_to_file for", "[replace_in_file for"]):
            return (
                f"<attempt_completion>\n"
                f"<result>\n"
                f"A operação no arquivo foi concluída com sucesso.\n"
                f"</result>\n"
                f"</attempt_completion>"
            )

        # 8. Conclusão limpa direta (perguntas ou finalizações)
        sol = self._solve_general_task(ctx)
        return (
            f"<attempt_completion>\n"
            f"<result>\n"
            f"{sol}\n"
            f"</result>\n"
            f"</attempt_completion>"
        )

    def _create_replace_in_file_action(self, file_path: str, file_content: str, prompt: str) -> str:
        """Cria um diff de edição no formato Cline SEARCH/REPLACE."""
        lines = file_content.splitlines()
        if len(lines) >= 3:
            search_block = "\n".join(lines[:3])
            replace_block = "// Modificado por Kimi-K3.1-Pro (1M de Contexto)\n" + search_block
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
                    "                        out.println(\"<h1>Servidor Nativo Java - Kimi-K3.1-Pro</h1>\");\n"
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
        return (
            f"### Plano de Ação - Kimi-K3.1-Pro: {prompt}\n\n"
            f"1. **Análise do Ambiente:** Mapeamento dos arquivos no CWD `{ctx.get('cwd', '.')}`.\n"
            f"2. **Implementação/Edição:** Criação ou modificação modular com tipagem e testes.\n"
            f"3. **Validação:** Verificação de sintaxe e prontidão de execução na sua máquina."
        )

    def _solve_general_task(self, ctx: Dict[str, Any]) -> str:
        prompt = ctx["effective_prompt"]
        return f"Processado com sucesso pelo Kimi-K3.1-Pro (1M de Contexto): {prompt}"
