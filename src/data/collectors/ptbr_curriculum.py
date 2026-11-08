import os
import json
from typing import List, Dict

# Currículo Extensivo de Programação com Alma Brasileira (PT-BR)
# Cobre regras de negócio nacionais, frameworks modernos e explicações didáticas técnicas.
PTBR_CURRICULUM_DOMAINS: List[Dict[str, str]] = [
    # 1. PIX e Pagamentos Avançados
    {
        "prompt": "Como implementar um webhook assíncrono em FastAPI para receber notificações de pagamento PIX de um PSP (como Asaas, Mercado Pago ou Efí) e validar a assinatura HMAC?",
        "response": '''from fastapi import FastAPI, Request, HTTPException, Header
import hmac
import hashlib
import json

app = FastAPI(title="Lloavece PIX Webhook")
CHAVE_SECRETA_WEBHOOK = "sua_chave_secreta_segura_aqui"

@app.post("/webhooks/pix/notificacao")
async def receber_notificacao_pix(request: Request, x_signature: str = Header(None)):
    """Recebe e processa o webhook de confirmação de liquidação PIX com validação HMAC-SHA256."""
    if not x_signature:
        raise HTTPException(status_code=401, detail="Assinatura HMAC ausente no cabeçalho.")
        
    corpo_bytes = await request.body()
    
    # Validação da integridade e autenticidade da requisição do PSP
    assinatura_esperada = hmac.new(
        CHAVE_SECRETA_WEBHOOK.encode("utf-8"),
        corpo_bytes,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(assinatura_esperada, x_signature):
        raise HTTPException(status_code=403, detail="Assinatura inválida. Requisição não autorizada.")
        
    evento = json.loads(corpo_bytes)
    dados_pix = evento.get("pix", {})
    
    txid = dados_pix.get("txid")
    valor = dados_pix.get("valor")
    pagador = dados_pix.get("pagador", {}).get("nome")
    
    # Processa o pedido no banco de dados (exemplo de idempotência)
    print(f"PIX recebido com sucesso! TXID: {txid} | Valor: R$ {valor} | Pagador: {pagador}")
    
    return {"status": "processado_com_sucesso"}'''
    },
    # 2. Boleto Bancário (Febraban)
    {
        "prompt": "Implemente em Python o cálculo do dígito verificador geral e dos campos da linha digitável de um Boleto Bancário segundo o manual da FEBRABAN.",
        "response": '''def modulo10_febraban(bloco: str) -> int:
    """Calcula o dígito verificador módulo 10 para um bloco da linha digitável."""
    multiplicador = 2
    soma = 0
    for char in reversed(bloco):
        mult = int(char) * multiplicador
        if mult > 9:
            mult = (mult // 10) + (mult % 10)
        soma += mult
        multiplicador = 1 if multiplicador == 2 else 2
        
    dezena_superior = (soma + 9) // 10 * 10
    dv = dezena_superior - soma
    return 0 if dv == 10 else dv

def modulo11_febraban(codigo_barras_43: str) -> int:
    """Calcula o dígito verificador geral (módulo 11) do código de barras de 44 posições."""
    pesos = [2, 3, 4, 5, 6, 7, 8, 9]
    soma = 0
    peso_idx = 0
    for char in reversed(codigo_barras_43):
        soma += int(char) * pesos[peso_idx]
        peso_idx = (peso_idx + 1) % len(pesos)
        
    resto = soma % 11
    dv = 11 - resto
    if dv in [0, 10, 11]:
        return 1
    return dv'''
    },
    # 3. CLT e Legislação Trabalhista
    {
        "prompt": "Escreva uma função em Python para calcular a folha de pagamento de um funcionário CLT no Brasil, calculando o desconto progressivo de INSS e a retenção de IRRF.",
        "response": '''def calcular_inss_progressivo(salario_bruto: float) -> float:
    """Calcula a dedução do INSS conforme as faixas da tabela progressiva oficial."""
    faixas = [
        (1412.00, 0.075),
        (2666.68, 0.09),
        (4000.03, 0.12),
        (7786.02, 0.14)
    ]
    teto_inss = 908.85
    desconto = 0.0
    limite_anterior = 0.0
    
    for limite, aliquota in faixas:
        if salario_bruto > limite:
            base = limite - limite_anterior
            desconto += base * aliquota
            limite_anterior = limite
        else:
            base = salario_bruto - limite_anterior
            desconto += base * aliquota
            break
            
    return min(round(desconto, 2), teto_inss)

def calcular_salario_liquido_clt(salario_bruto: float, dependentes: int = 0) -> dict:
    """Calcula o salário líquido descontando INSS e IRRF (Dedução por dependente: R$ 189,59)."""
    inss = calcular_inss_progressivo(salario_bruto)
    deducao_dependentes = dependentes * 189.59
    base_irrf = salario_bruto - inss - deducao_dependentes
    
    # Tabela progressiva de IRRF
    if base_irrf <= 2259.20:
        irrf = 0.0
    elif base_irrf <= 2826.65:
        irrf = (base_irrf * 0.075) - 169.44
    elif base_irrf <= 3751.05:
        irrf = (base_irrf * 0.15) - 381.44
    elif base_irrf <= 4664.68:
        irrf = (base_irrf * 0.225) - 662.77
    else:
        irrf = (base_irrf * 0.275) - 896.00
        
    irrf = max(0.0, round(irrf, 2))
    salario_liquido = round(salario_bruto - inss - irrf, 2)
    
    return {
        "salario_bruto": salario_bruto,
        "desconto_inss": inss,
        "desconto_irrf": irrf,
        "salario_liquido": salario_liquido
    }'''
    },
    # 4. LGPD Compliance (Anonimização de Dados Sensíveis)
    {
        "prompt": "Como implementar um middleware em Python para anonimizar dados sensíveis (PII) de logs em conformidade com a LGPD (Lei Geral de Proteção de Dados)?",
        "response": '''import re

def mascarar_dado_lgpd(texto: str) -> str:
    """Mascara automaticamente CPFs, emails e cartões de crédito em strings de log."""
    # Máscara para CPF: 123.456.789-00 -> 123.***.***-00
    padrao_cpf = r'(\d{3})\.?(\d{3})\.?(\d{3})-?(\d{2})'
    texto = re.sub(padrao_cpf, r'\\1.***.***-\\4', texto)
    
    # Máscara para E-mail: usuario@dominio.com -> u***@dominio.com
    padrao_email = r'([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+)'
    texto = re.sub(padrao_email, r'\\1***@\\2', texto)
    
    # Máscara para Cartão de Crédito (preserva apenas os 4 últimos dígitos)
    padrao_cartao = r'\\b\\d{4}[ -]?\\d{4}[ -]?\\d{4}[ -]?(\\d{4})\\b'
    texto = re.sub(padrao_cartao, r'****-****-****-\\1', texto)
    
    return texto'''
    },
    # 5. Explicação e Mentoria Técnica em Português Coloquial de Dev
    {
        "prompt": "Fala dev! Meu código FastAPI tá dando 'RuntimeError: Task attached to a different loop' quando tento rodar consultas no Postgres com asyncpg. Como resolvo isso?",
        "response": '''Fala dev! Esse erro é clássico no ecossistema assíncrono do Python quando trabalhamos com conexões de banco de dados.

### O que tá acontecendo debaixo do capô?
O `asyncpg` cria conexões que são amarradas ao **Event Loop** que estava ativo no momento em que a conexão foi aberta. Se você instanciou seu engine ou pool de conexões globalmente fora do ciclo de vida da aplicação (ou em uma thread secundária), o FastAPI troca de loop em requests assíncronos e o `asyncpg` tenta usar um socket registrado em outro loop.

### Como resolver do jeito certo (Lifespan do FastAPI):

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncpg

db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    # Inicializa o pool de conexões DENTRO do event loop oficial da aplicação
    db_pool = await asyncpg.create_pool(
        dsn="postgresql://usuario:senha@localhost:5432/meubanco",
        min_size=5,
        max_size=20
    )
    print("Pool de conexões com PostgreSQL inicializado com sucesso!")
    yield
    # Fecha graciosamente ao desligar a API
    await db_pool.close()
    print("Pool fechado.")

app = FastAPI(lifespan=lifespan)

@app.get("/usuarios")
async def listar_usuarios():
    async with db_pool.acquire() as conn:
        usuarios = await conn.fetch("SELECT id, nome, email FROM usuarios LIMIT 10")
        return [dict(u) for u in usuarios]
```

Dessa forma, o pool sempre nasce e morre no mesmo loop do servidor, eliminando o erro de vez!'''
    },
    # 6. SQL e Banco de Dados (Particionamento e Índices)
    {
        "prompt": "Escreva uma migration em SQL para criar uma tabela particionada por mês para armazenar transações financeiras no PostgreSQL, com índices otimizados.",
        "response": '''-- Tabela principal de transações particionada por data
CREATE TABLE transacoes_financeiras (
    id UUID DEFAULT gen_random_uuid(),
    usuario_id UUID NOT NULL,
    valor NUMERIC(15, 2) NOT NULL,
    tipo VARCHAR(20) NOT NULL, -- 'PIX_ENVIADO', 'PIX_RECEBIDO', 'BOLETO'
    chave_pix VARCHAR(100),
    data_transacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, data_transacao)
) PARTITION BY RANGE (data_transacao);

-- Partição para o mês de Novembro de 2026
CREATE TABLE transacoes_2026_11 PARTITION OF transacoes_financeiras
    FOR VALUES FROM ('2026-11-01 00:00:00+00') TO ('2026-12-01 00:00:00+00');

-- Partição para o mês de Dezembro de 2026
CREATE TABLE transacoes_2026_12 PARTITION OF transacoes_financeiras
    FOR VALUES FROM ('2026-12-01 00:00:00+00') TO ('2027-01-01 00:00:00+00');

-- Índices compostos para consultas ultrarrápidas de extrato por usuário
CREATE INDEX idx_transacoes_usuario_data ON transacoes_financeiras (usuario_id, data_transacao DESC);
CREATE INDEX idx_transacoes_tipo ON transacoes_financeiras (tipo);'''
    }
]

def export_ptbr_dataset(output_sft: str, output_pretrain: str):
    """Exporta o currículo técnico nacional para os datasets de treino."""
    with open(output_sft, "a", encoding="utf-8") as f_sft, open(output_pretrain, "a", encoding="utf-8") as f_pre:
        for item in PTBR_CURRICULUM_DOMAINS:
            f_sft.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_pre.write(json.dumps({"content": item["response"]}, ensure_ascii=False) + "\n")
            # Adiciona também a pergunta como contexto técnico
            doc = f"# Pergunta Técnica:\n{item['prompt']}\n\n# Resposta e Implementação:\n{item['response']}"
            f_pre.write(json.dumps({"content": doc}, ensure_ascii=False) + "\n")
