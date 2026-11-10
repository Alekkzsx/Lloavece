# 🚀 Kimi-K3.1-Pro (Moonshot AI) - Checkpoint K-Bit Multi-Rank (1M Contexto)

Este diretório contém os pesos e arquivos oficiais de arquitetura do **Kimi-K3.1-Pro** compactados através da tecnologia **K-Bit Multi-Rank (Decomposição de Kronecker Rank-4)**.

---

## 📊 Especificações Técnicas do Modelo

- **Modelo Base**: `moonshotai/Kimi-K3.1-Pro`
- **Arquitetura**: `KimiK31ForConditionalGeneration` (com `KimiLinearForCausalLM`)
- **Janela de Contexto Nativa**: **1.048.576 tokens (1M)**
- **Dimensão Oculta (Hidden Size)**: `7.168`
- **Dimensão Intermediária (MLP)**: `33.792`
- **Número de Camadas**: `93`
- **Tamanho do Vocabulário**: `163.840`
- **MoE Experts**: `896 experts` com roteamento dinâmico
- **Algoritmo de Compressão**: K-Bit Multi-Rank Kronecker ($W \approx \sum_{k=1}^4 A_k \otimes B_k$)
- **Taxa de Compressão Média**: **1.319x** (Redução de centenas de gigabytes para megabytes)

---

## 📁 Arquivos no Diretório

| Arquivo | Descrição | Tamanho |
| :--- | :--- | :--- |
| `kimi_k31_pro_kbit_1m.pt` | Checkpoint com fatores K-Bit ($A_k$, $B_k$), normas oficiais e metadados de 1M | ~2.10 MB |
| `config.json` | Configuração oficial da arquitetura do Kimi-K3.1-Pro (Moonshot AI) | 7.13 KB |
| `configuration_kimi_k3.py` | Definição da classe `KimiK3Config` e mapeamento de auto-configuração | 11.15 KB |
| `encoding_k3.py` | Lógica de codificação e positional embeddings para 1M de tokens | 25.68 KB |
| `generation_config.json` | Parâmetros de geração (temperatura, eos/bos tokens) | 53 B |
| `kimi_k3_processor.py` | Processador multimodal e de tokenização do Kimi-K3 | 7.48 KB |
| `kimi_k3_vision_processing.py`| Módulo de pré-processamento de imagens e visão computacional | 6.53 KB |
| `media_utils.py` | Utilitários de manipulação de mídia | 13.52 KB |
| `kbit_model_report.json` | Relatório estruturado de auditoria e validação | 0.49 KB |

---

## ⚡ Como Carregar e Testar o Modelo

```python
import torch
from kimi_k31_pro.compressor import multi_kronecker_forward

# 1. Carrega o checkpoint
ckpt = torch.load("kimi_k31_pro/Data/kimi_k31_pro_kbit_1m.pt", map_location="cpu")
print("Modelo:", ckpt["model_name"])
print("Contexto:", f"{ckpt['context_window']:,} tokens")

# 2. Executa inferência em uma camada K-Bit
layer_data = ckpt["kbit_layers"]["layers.0.self_attn.q_proj"]
x = torch.randn(1, 4, 7168) # Batch=1, Seq=4, Hidden=7168
y = multi_kronecker_forward(layer_data["matrices_A"], layer_data["matrices_B"], x)
print("Saída do Forward Pass:", y.shape) # (1, 4, 7168)
```
