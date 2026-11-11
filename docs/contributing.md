# 🤝 Contribuindo com o Lloavece Kode

Obrigado pelo interesse em contribuir! Este documento fornece diretrizes para contribuir com o projeto.

## 📋 Índice

- [Código de Conduta](#código-de-conduta)
- [Como Contribuir](#como-contribuir)
- [Áreas Prioritárias](#áreas-prioritárias)
- [Setup de Desenvolvimento](#setup-de-desenvolvimento)
- [Processo de Contribuição](#processo-de-contribuição)
- [Estilo de Código](#estilo-de-código)
- [Reportando Bugs](#reportando-bugs)
- [Sugerindo Features](#sugerindo-features)

---

## 📜 Código de Conduta

Este projeto adota o [Contributor Covenant](https://www.contributor-covenant.org/). Ao participar, você concorda em manter um ambiente respeitoso e inclusivo.

---

## 🚀 Como Contribuir

Existem várias formas de contribuir:

1. **Reportar bugs** - Encontrou um problema? Abra uma issue!
2. **Sugerir features** - Tem uma ideia? Compartilhe conosco!
3. **Melhorar documentação** - Documentação clara é essencial
4. **Contribuir com código** - Implemente features ou corrija bugs
5. **Coletar dados** - Ajude a construir datasets de qualidade
6. **Testar o modelo** - Use e forneça feedback

---

## 🎯 Áreas Prioritárias

### Alta Prioridade

1. **Coleta de Dados**
   - Repositórios brasileiros de qualidade
   - Documentação técnica em PT-BR
   - Exemplos de código real-world

2. **Curadoria de Dados**
   - Scripts de limpeza e filtragem
   - Detecção de duplicatas
   - Validação de qualidade

3. **Treinamento**
   - Otimização de hiperparâmetros
   - Scripts de CPT, SFT, DPO
   - Pipeline de self-improvement

4. **Avaliação**
   - Implementação de benchmarks
   - Métricas de qualidade
   - Análise de resultados

### Média Prioridade

5. **Inferência**
   - Otimização de performance
   - Suporte a mais frameworks
   - API REST/WebSocket

6. **Deploy**
   - VS Code Extension
   - CLI melhorado
   - Docker images

### Baixa Prioridade

7. **Documentação**
   - Tutoriais
   - Exemplos de uso
   - Guias de troubleshooting

---

## 💻 Setup de Desenvolvimento

### Pré-requisitos

- Python 3.11+
- Git
- CUDA 12.x (opcional, para GPU)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Lloavece.git
cd Lloavece

# Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale dependências de desenvolvimento
pip install -r requirements.txt

# Configure o ambiente
python scripts/setup_environment.py

# Instale pre-commit hooks (opcional)
pre-commit install
```

### Estrutura do Projeto

```
Lloavece/
├── scripts/          # Scripts auxiliares
├── training/         # Scripts de treino
│   ├── configs/     # Configurações YAML
│   ├── scripts/     # Scripts Python
│   └── utils/       # Utilitários
├── inference/        # Código de inferência
├── data/            # Datasets
├── models/          # Modelos treinados
├── tests/           # Testes
└── docs/            # Documentação
```

---

## 🔄 Processo de Contribuição

### 1. Fork o Repositório

Clique em "Fork" no GitHub e clone seu fork:

```bash
git clone https://github.com/SEU-USUARIO/Lloavece.git
cd Lloavece
```

### 2. Crie uma Branch

```bash
# Para features
git checkout -b feature/nome-da-feature

# Para bug fixes
git checkout -b fix/nome-do-bug

# Para documentação
git checkout -b docs/nome-da-doc
```

### 3. Faça Suas Alterações

- Siga o [estilo de código](#estilo-de-código)
- Adicione testes se aplicável
- Atualize a documentação se necessário
- Mantenha commits atômicos e descritivos

### 4. Teste Localmente

```bash
# Execute testes
pytest tests/

# Verifique qualidade do código
black . --check
flake8 .
mypy .

# Execute linting
pylint scripts/ training/
```

### 5. Commit Suas Alterações

```bash
git add .
git commit -m "tipo: descrição concisa"
```

**Formato de commits:**
- `feat:` Nova feature
- `fix:` Correção de bug
- `docs:` Mudanças na documentação
- `style:` Formatação, sem mudança de código
- `refactor:` Refatoração de código
- `test:` Adição ou correção de testes
- `chore:` Tarefas de manutenção

**Exemplos:**
```bash
git commit -m "feat: adicionar script de coleta de dados do GitHub"
git commit -m "fix: corrigir bug no carregamento do modelo"
git commit -m "docs: atualizar README com instruções de instalação"
```

### 6. Push para Seu Fork

```bash
git push origin feature/nome-da-feature
```

### 7. Abra um Pull Request

1. Vá para o repositório original no GitHub
2. Clique em "Pull Requests" → "New Pull Request"
3. Selecione sua branch
4. Preencha o template de PR
5. Aguarde revisão

---

## 💅 Estilo de Código

### Python

Seguimos [PEP 8](https://peps.python.org/pep-0008/) com algumas modificações:

- **Formatter**: Black (line length = 100)
- **Linter**: Flake8
- **Type hints**: Obrigatórios para funções públicas
- **Docstrings**: Google style

**Exemplo:**

```python
from typing import List, Optional


def process_data(
    data: List[dict],
    max_items: Optional[int] = None,
    filter_invalid: bool = True,
) -> List[dict]:
    """
    Processa uma lista de dados.
    
    Args:
        data: Lista de dicionários com dados brutos
        max_items: Número máximo de itens a processar
        filter_invalid: Se deve filtrar itens inválidos
    
    Returns:
        Lista de dados processados
    
    Raises:
        ValueError: Se data estiver vazio
    """
    if not data:
        raise ValueError("data não pode estar vazio")
    
    result = []
    
    for item in data[:max_items]:
        if filter_invalid and not is_valid(item):
            continue
        
        processed = transform(item)
        result.append(processed)
    
    return result
```

### Convenções

- **Nomes de arquivos**: snake_case (ex: `collect_data.py`)
- **Nomes de classes**: PascalCase (ex: `DataCollector`)
- **Nomes de funções**: snake_case (ex: `collect_github_repos`)
- **Constantes**: UPPER_SNAKE_CASE (ex: `MAX_REPOS`)
- **Imports**: Agrupados e ordenados (use isort)

### Testes

```python
import pytest
from scripts.collect_data import CodeDataCollector


def test_collect_github_repos():
    """Testa coleta de repositórios do GitHub"""
    collector = CodeDataCollector()
    
    repos = collector.collect_github_repos(
        languages=['Python'],
        min_stars=1000,
        max_repos=10,
    )
    
    assert len(repos) <= 10
    assert all('name' in repo for repo in repos)
    assert all('stars' in repo for repo in repos)


def test_collect_github_repos_empty():
    """Testa comportamento com resultado vazio"""
    collector = CodeDataCollector()
    
    repos = collector.collect_github_repos(
        languages=['LanguageInexistente'],
        max_repos=10,
    )
    
    assert len(repos) == 0
```

---

## 🐛 Reportando Bugs

### Antes de Reportar

1. **Verifique issues existentes** - O bug já foi reportado?
2. **Atualize para a última versão** - O bug ainda existe?
3. **Reproduza o problema** - Consiga reproduzir consistentemente?

### Template de Bug Report

```markdown
## Descrição do Bug

Descrição clara e concisa do bug.

## Passos para Reproduzir

1. Execute '...'
2. Use o comando '...'
3. Veja o erro

## Comportamento Esperado

O que deveria acontecer.

## Comportamento Atual

O que realmente acontece.

## Screenshots

Se aplicável, adicione screenshots.

## Ambiente

- OS: [ex: Ubuntu 22.04]
- Python: [ex: 3.11.5]
- PyTorch: [ex: 2.2.0]
- CUDA: [ex: 12.1]
- GPU: [ex: RTX 3090]

## Logs

Adicione logs relevantes aqui.

## Contexto Adicional

Qualquer outra informação relevante.
```

---

## 💡 Sugerindo Features

### Template de Feature Request

```markdown
## Descrição da Feature

Descrição clara e concisa da feature.

## Problema Relacionado

Este feature está relacionado a um problema? Descreva.
Ex: "Eu fico frustrado quando..."

## Solução Proposta

Descrição da solução que você gostaria.

## Alternativas Consideradas

Descrição de alternativas que você considerou.

## Contexto Adicional

Mockups, exemplos, referências, etc.
```

---

## 📊 Coleta de Dados

Se você quer contribuir com dados:

### Fontes de Dados

1. **GitHub**
   - Repositórios de qualidade (stars > 100)
   - Código limpo e bem documentado
   - Variedade de linguagens

2. **Stack Overflow**
   - Perguntas com respostas de alta qualidade
   - Score > 10
   - Respostas aceitas

3. **Documentação**
   - Docs oficiais (Python, JS, etc)
   - Tutoriais de qualidade
   - Blogs técnicos

4. **Repositórios Brasileiros**
   - Projetos open-source brasileiros
   - Documentação em PT-BR
   - Exemplos de código local

### Critérios de Qualidade

- ✅ Código que compila/executa
- ✅ Documentação clara
- ✅ Boas práticas
- ✅ Diversidade de problemas
- ❌ Código duplicado
- ❌ Código mal formatado
- ❌ Código inseguro

---

## 🔍 Revisão de Código

### Checklist do Revisor

- [ ] Código segue o estilo do projeto
- [ ] Testes foram adicionados/atualizados
- [ ] Documentação foi atualizada
- [ ] Não há código duplicado
- [ ] Performance é aceitável
- [ ] Segurança foi considerada
- [ ] Commits são atômicos e descritivos

### Feedback Construtivo

- Seja específico e objetivo
- Foque no código, não na pessoa
- Sugira melhorias, não apenas critique
- Reconheça bons aspectos do código

---

## 📝 Documentação

### Tipos de Documentação

1. **README** - Visão geral do projeto
2. **ARCHITECTURE** - Arquitetura técnica
3. **Tutoriais** - Guias passo-a-passo
4. **API Reference** - Documentação de funções
5. **Examples** - Exemplos de uso

### Ferramentas

- **Markdown** para documentação
- **Google-style docstrings** para código Python
- **Type hints** para melhor IDE support

---

## ❓ Perguntas Frequentes

### Q: Como configuro o ambiente de desenvolvimento?

A: Veja [Setup de Desenvolvimento](#setup-de-desenvolvimento).

### Q: Qual GPU eu preciso?

A: Para desenvolvimento, qualquer GPU moderna serve. Para treino, recomendamos A100 40GB+.

### Q: Posso usar Windows?

A: Sim, mas Linux é recomendado para desenvolvimento de ML.

### Q: Como testo minhas mudanças?

A: Execute `pytest tests/` e verifique com `black`, `flake8`, `mypy`.

---

## 🙏 Agradecimentos

Obrigado por contribuir! Cada contribuição, por menor que seja, faz diferença.

---

**Contato**: Se tiver dúvidas, abra uma issue ou entre em contato via GitHub Discussions.
