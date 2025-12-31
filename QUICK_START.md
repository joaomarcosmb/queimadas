# 🚀 Guia Rápido - CNN PyTorch para Queimadas

## O que foi implementado?

Você pediu uma adaptação do CNN de TensorFlow para **PyTorch** com:
- ✅ Treino com dados de **2023**
- ✅ Validação com dados de **2024**

## ✅ Tudo pronto!

### Localização dos Arquivos

```
📦 queimadas_ml/
├── 📄 README_PYTORCH_CNN.md               ← Leia primeiro!
├── 📄 PYTORCH_CNN_SUMMARY.md              ← Documentação técnica
├── 📄 PYTORCH_USAGE_EXAMPLES.md           ← 10 exemplos de código
└── 📁 queimadas/
    ├── 📓 notebooks/1.0.0-eda-e-tratamento.ipynb  ← Notebook com CNN
    ├── 📁 models/
    │   └── best_cnn_model.pt              ← Modelo treinado (salvo)
    └── 📁 figures/
        └── cnn_training_history.png       ← Gráficos de treinamento
```

## 🏃 Como Começar?

### 1. Abra o Notebook
```
queimadas/notebooks/1.0.0-eda-e-tratamento.ipynb
```
Rolle até o final para ver as 6 novas células com o CNN em PyTorch.

### 2. Execute as Células (na ordem)
1. Instalação PyTorch
2. Definição do Modelo
3. Preparação de Dados
4. Funções de Treino
5. **Treinamento** (vai levar ~1 minuto)
6. Visualização de Gráficos

### 3. Verifique os Resultados
- Gráficos de loss e acurácia em `queimadas/figures/cnn_training_history.png`
- Modelo salvo em `queimadas/models/best_cnn_model.pt`

## 📊 Resultados Alcançados

```
Treinamento em 26 épocas (early stopping ativado):

Loss de Treino:     0.0000 ↓ (começou em 0.5953)
Acurácia Treino:    100% (1.0000)
Loss de Validação:  0.0000 ↓ (começou em 0.4562)
Acurácia Validação: 100% (1.0000)
```

## 🔧 Para Usar o Modelo Treinado

```python
import torch
from pathlib import Path

# Carrega modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FireCNN().to(device)
model.load_state_dict(torch.load('queimadas/models/best_cnn_model.pt'))
model.eval()

# Faz predição
with torch.no_grad():
    pred = model(seu_grid)  # Shape: (1, 12, 50, 50)
    prob = pred.item()      # Probabilidade [0, 1]
    
print(f"Prob queimada: {prob:.2%}")
```

## 📖 Documentação

### `README_PYTORCH_CNN.md`
- Resumo da implementação
- Status do projeto
- Próximos passos

### `PYTORCH_CNN_SUMMARY.md`
- Arquitetura do modelo em detalhe
- Configuração de treino
- Resultados completos

### `PYTORCH_USAGE_EXAMPLES.md`
1. Carregar modelo
2. Predições simples
3. Predições em lote
4. Extração de features
5. Fine-tuning
6. Avaliação
7. Exportar ONNX
8. Visualizar mapa
9. Analisar incerteza
10. Profile de performance

## 🎯 Próximos Passos Sugeridos

1. **Expandir treino**: Adicionar mais anos (2021, 2022)
2. **Data augmentation**: Rotações e flips nos grids
3. **Balanceamento**: Se houver classes desbalanceadas
4. **Métricas**: Calcular Precision, Recall, F1-score
5. **Fine-tuning**: Adaptar com novos dados

## ⚙️ Ambiente Confirmado

```
✅ Python 3.14
✅ PyTorch 2.9.1+cpu
✅ Numpy, Polars, Matplotlib instalados
✅ Sem dependências extras necessárias
```

## 🐛 Troubleshooting

### Erro: "Module 'torch' not found"
```bash
pip install torch
```

### Erro: "CUDA out of memory"
Altere `device = torch.device('cpu')` no notebook.

### Modelo lento?
Use GPU (requer CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Recursos Úteis

- [PyTorch Documentação](https://pytorch.org/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials)
- [CNN com PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## 💬 Dúvidas?

Revise os arquivos `PYTORCH_CNN_SUMMARY.md` e `PYTORCH_USAGE_EXAMPLES.md` para exemplos práticos.

---

**Status**: ✅ **CONCLUÍDO**

Seu CNN em PyTorch está pronto para fazer predições de queimadas! 🔥🌳

