# 🤝 Guia de Contribuição — Lloavece Kode

Obrigado por considerar contribuir com a **Lloavece Kode**! Este documento explica como você pode ajudar a construir a primeira IA local brasileira especializada em código.

---

## 🎯 Áreas Prioritárias

### 1. Coleta de Dados 📊

**Objetivo**: Construir datasets de código em PT-BR de alta qualidade.

**Como contribuir**:
- Identificar repositórios brasileiros de qualidade no GitHub
- Coletar documentação técnica em português
- Encontrar tutoriais, blog posts e cursos brasileiros
- Submeter datasets limpos e bem documentados

**Critérios de qualidade**:
- Código bem escrito, com boas práticas
- Comentários e documentação em PT-BR
- Licenças compatíveis (Apache 2.0, MIT, BSD)
- Sem conteúdo duplicado ou de baixa qualidade

**Ferramentas**:
```bash
# Script de coleta (a ser implementado)
python scripts/collect_data.py --source github --language pt-br
```

---

### 2. Dataset SFT (Supervised Fine-Tuning) 💬

**Objetivo**: Criar instruções de código em PT-BR para treinar o modelo.

**Formato esperado**:
```json
{
  "instruction": "Crie uma função Python que valide um CPF",
  "input": "",
  "output": "def validar_cpf(cpf: str) -> bool:\n    # Código aqui..."
}
```

**Tipos de instruções**:
- **Code Generation** (40%): "Crie...", "Implemente...", "Escreva..."
- **Code Explanation** (20%): "Explique...", "O que faz...", "Como funciona..."
- **Code Review** (20%): "Revise...", "Tem algum bug?", "Como melhorar..."
- **Debugging** (15%): "Por que dá erro?", "Corrija...", "Não funciona..."
- **Contexto Brasileiro** (5%): "Como integrar PIX?", "Como validar CPF?"

**Como contribuir**:
1. Criar instruções seguindo o formato acima
2. Submeter via Pull Request na pasta `data/sft/`
3. Aguardar revisão da equipe

---

### 3. Avaliação e Benchmarks 📈

**Objetivo**: Medir a qualidade do modelo de forma objetiva.

**Métricas**:
- **HumanEval**: Code generation (alvo: >80% para 32B)
- **MBPP**: Code completion (alvo: >75% para 32B)
- **CodeBLEU**: Qualidade do código (alvo: >0.7 para 32B)
- **Avaliação humana**: Fluência em PT-BR, precisão técnica, contexto brasileiro

**Como contribuir**:
- Rodar benchmarks e reportar resultados
- Criar testes específicos para contexto brasileiro
- Avaliar qualitativamente as saídas do modelo
- Propor novas métricas de avaliação

---

### 4. VS Code Extension 🧩

**Objetivo**: Integrar o modelo ao VS Code para code completion.

**Features prioritárias**:
- Autocompletar inteligente
- Explicação de código (hover)
- Code review inline
- Geração de código via comando
- Suporte a múltiplos modelos (32B, 14B, 4B)

**Como contribuir**:
- Implementar features em TypeScript
- Melhorar UI/UX
- Corrigir bugs
- Adicionar suporte a outras IDEs (JetBrains, Neovim)

**Setup**:
```bash
cd extension
npm install
npm run build
```

---

### 5. Documentação 📚

**Objetivo**: Documentação clara e completa para usuários e contribuidores.

**Documentos prioritários**:
- **installation.md**: Como instalar e configurar
- **usage.md**: Como usar o modelo (CLI, API, VS Code)
- **api.md**: Documentação da API REST
- **training.md**: Como treinar o modelo
- **contributing.md**: Este documento

**Como contribuir**:
- Escrever tutoriais
- Criar exemplos práticos
- Traduzir documentação
- Melhorar a documentação existente

---

## 🔄 Processo de Contribuição

### 1. Fork o Repositório

```bash
# No GitHub, clique em "Fork"
# Clone seu fork
git clone https://github.com/SEU-USUARIO/Lloavece.git
cd Lloavece
```

### 2. Crie uma Branch

```bash
# Para features
git checkout -b feature/minha-feature

# Para correções
git checkout -b fix/correcao-bug

# Para documentação
git checkout -b docs/melhoria-docs
```

### 3. Faça suas Alterações

- Siga os padrões de código do projeto
- Adicione testes se aplicável
- Atualize a documentação se necessário
- Mantenha commits pequenos e focados

### 4. Commit suas Mudanças

```bash
# Mensagens de commit claras e descritivas
git commit -m "feat: adiciona suporte a validação de CPF"
git commit -m "fix: corrige bug no tokenizer"
git commit -m "docs: atualiza guia de instalação"
```

**Convenção de commits**:
- `feat`: Nova feature
- `fix`: Correção de bug
- `docs`: Mudanças na documentação
- `refactor`: Refatoração de código
- `test`: Adição ou modificação de testes
- `chore`: Tarefas de manutenção

### 5. Push para seu Fork

```bash
git push origin feature/minha-feature
```

### 6. Abra um Pull Request

1. Vá para o repositório original no GitHub
2. Clique em "New Pull Request"
3. Selecione sua branch
4. Descreva suas mudanças claramente
5. Aguarde revisão

---

## ✅ Checklist para Pull Requests

- [ ] Código segue os padrões do projeto
- [ ] Testes foram adicionados/atualizados (se aplicável)
- [ ] Documentação foi atualizada (se necessário)
- [ ] Commits estão organizados e descritivos
- [ ] Pull Request tem descrição clara
- [ ] Não há conflitos com a branch principal
- [ ] CI/CD passa (quando implementado)

---

## 🐛 Reportar Bugs

Encontrou um bug? Abra uma issue!

**Informações importantes**:
- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs. atual
- Versão do modelo (32B, 14B, 4B)
- Sistema operacional e hardware
- Logs de erro (se houver)

**Template**:
```markdown
## Descrição do Bug
[Descreva o problema claramente]

## Passos para Reproduzir
1. Passo 1
2. Passo 2
3. Passo 3

## Comportamento Esperado
[O que deveria acontecer]

## Comportamento Atual
[O que está acontecendo]

## Ambiente
- Modelo: [32B/14B/4B]
- Sistema Operacional: [Windows/Linux/Mac]
- RAM: [X GB]
- GPU: [Modelo ou "CPU only"]

## Logs
[Inclua logs de erro se houver]
```

---

## 💡 Sugerir Features

Tem uma ideia para melhorar a Lloavece Kode? Abra uma issue!

**Informações importantes**:
- Descrição da feature
- Por que é útil
- Como deve funcionar
- Exemplos de uso (se aplicável)

**Template**:
```markdown
## Descrição da Feature
[Descreva a feature claramente]

## Por que é útil?
[Explique o problema que resolve ou o benefício]

## Como deve funcionar?
[Descreva o comportamento esperado]

## Exemplos de uso
[Inclua exemplos práticos se aplicável]
```

---

## 📜 Código de Conduta

### Nossos Valores

- **Respeito**: Trate todos com respeito e empatia
- **Colaboração**: Trabalhe em equipe, ajude outros contribuidores
- **Qualidade**: Busque excelência técnica e atenção aos detalhes
- **Transparência**: Seja aberto sobre decisões e processos
- **Inclusão**: Acolha contribuidores de todos os backgrounds

### Comportamento Esperado

- Use linguagem acolhedora e inclusiva
- Respeite diferentes pontos de vista
- Aceite críticas construtivas graciosamente
- Foque no que é melhor para a comunidade
- Mostre empatia com outros membros

### Comportamento Inaceitável

- Linguagem ou imagens sexualizadas
- Ataques pessoais ou insultos
- Assédio público ou privado
- Publicar informações privadas de outros
- Qualquer forma de discriminação

---

## 🔧 Setup de Desenvolvimento

### Pré-requisitos

```bash
# Python 3.11+
python --version

# Git
git --version

# NVIDIA CUDA (se usar GPU)
nvidia-smi
```

### Instalação

```bash
# Clone o repositório
git clone https://github.com/alekkzsx/Lloavece.git
cd Lloavece

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Instale em modo de desenvolvimento
pip install -e .
```

### Estrutura do Projeto

```
Lloavece/
├── data/              # Datasets
├── training/          # Scripts de treino
├── models/            # Modelos treinados
├── inference/         # Código de inferência
├── extension/         # VS Code extension
├── scripts/           # Scripts auxiliares
└── docs/              # Documentação
```

---

## 📞 Contato

- **GitHub Issues**: Para bugs, features e discussões técnicas
- **Discord**: (a definir) - Para chat em tempo real
- **Email**: (a definir) - Para assuntos privados

---

## 🙏 Agradecimentos

Agradecemos a todos os contribuidores que ajudam a construir a Lloavece Kode!

**Contribuidores**:
- (Será atualizado conforme contribuições)

---

**Desenvolvido com orgulho no Brasil** 🇧🇷
