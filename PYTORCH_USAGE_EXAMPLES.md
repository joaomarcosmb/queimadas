# Exemplos de Uso - CNN PyTorch para Queimadas

## 1. Carregar o Modelo Treinado

```python
import torch
from torch import nn

# Redefine a arquitetura do modelo
class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# Carrega o modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FireCNN().to(device)
model.load_state_dict(torch.load('models/best_cnn_model.pt'))
model.eval()
print("Modelo carregado com sucesso!")
```

## 2. Fazer Predições em Novos Dados

### Exemplo 1: Predição com um grid único

```python
# Criar um grid espacial-temporal (12 meses × 50×50)
import numpy as np

heat_map = np.random.randn(12, 50, 50)  # Seus dados de queimadas
heat_map = np.maximum(heat_map, 0)      # Garantir valores positivos
heat_map = heat_map / heat_map.max()    # Normalizar [0, 1]

# Converter para tensor e fazer predição
input_tensor = torch.FloatTensor(heat_map).unsqueeze(0).to(device)  # (1, 12, 50, 50)

with torch.no_grad():
    prediction = model(input_tensor)
    prob = prediction.item()

print(f"Probabilidade de queimada: {prob:.4f}")
print(f"Predição: {'Queimada detectada' if prob > 0.5 else 'Sem queimada'}")
```

### Exemplo 2: Predições em lote (batch)

```python
# Fazer múltiplas predições ao mesmo tempo
num_samples = 10
heat_maps_batch = np.random.randn(num_samples, 12, 50, 50)
heat_maps_batch = np.maximum(heat_maps_batch, 0)
heat_maps_batch = heat_maps_batch / heat_maps_batch.max(axis=(1, 2, 3), keepdims=True)

# Converter para tensor
input_tensor = torch.FloatTensor(heat_maps_batch).to(device)

with torch.no_grad():
    predictions = model(input_tensor)
    probs = predictions.cpu().numpy().flatten()

for i, prob in enumerate(probs):
    print(f"Amostra {i+1}: {prob:.4f} - {'Queimada' if prob > 0.5 else 'Sem queimada'}")
```

## 3. Extração de Features (Ativações Intermediárias)

```python
# Criar versão do modelo para extrair features da camada FC1
class FireCNNFeatures(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(
            base_model.conv1, base_model.relu1, base_model.pool1,
            base_model.conv2, base_model.relu2, base_model.pool2,
            base_model.flatten,
            base_model.fc1, base_model.relu3
        )
    
    def forward(self, x):
        return self.features(x)

# Extrair features
feature_extractor = FireCNNFeatures(model).to(device)
with torch.no_grad():
    features = feature_extractor(input_tensor)  # Shape: (batch_size, 128)

print(f"Features extraídas: shape {features.shape}")
# Usar para comparação de similaridade, clustering, etc.
```

## 4. Fine-tuning com Novos Dados

```python
# Descongelar apenas as últimas camadas para ajuste fino
for param in model.parameters():
    param.requires_grad = False

# Recongelar apenas FC1 e FC2
model.fc1.weight.requires_grad = True
model.fc1.bias.requires_grad = True
model.fc2.weight.requires_grad = True
model.fc2.bias.requires_grad = True

# Re-treinar com novos dados
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # LR menor
criterion = nn.BCELoss()

# Loop de treinamento (similar ao notebook)
for epoch in range(10):
    model.train()
    for X_batch, y_batch in new_train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 5. Avaliar no Conjunto de Validação

```python
def evaluate_model(model, val_loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = (outputs > 0.5).float().cpu().numpy()
            labels = y.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())
    
    # Calcular métricas
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return precision, recall, f1

# Usar
precision, recall, f1 = evaluate_model(model, val_loader, device)
```

## 6. Salvar e Carregar Checkpoint Completo

```python
# Salvar checkpoint (modelo + otimizador + estado)
checkpoint = {
    'epoch': 26,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_val_loss,
}
torch.save(checkpoint, 'models/checkpoint.pt')

# Carregar checkpoint
checkpoint = torch.load('models/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
print(f"Resumindo a partir da época {start_epoch}")
```

## 7. Exportar para ONNX (para usar em outras frameworks)

```python
# Exportar modelo para formato ONNX
import torch.onnx

dummy_input = torch.randn(1, 12, 50, 50).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "models/fire_cnn.onnx",
    opset_version=12,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Modelo exportado para ONNX")

# Agora pode ser usado com ONNX Runtime, TensorFlow, etc.
```

## 8. Visualizar Predições em um Mapa

```python
import matplotlib.pyplot as plt
import numpy as np

# Supor que temos um grid de predições para diferentes áreas
grid_predictions = np.random.rand(100, 100)  # 100×100 áreas geográficas

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(grid_predictions, cmap='RdYlGn_r', vmin=0, vmax=1)
ax.set_title('Probabilidade de Queimadas por Região')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probabilidade')
plt.savefig('fire_prediction_map.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 9. Analisar Incerteza das Predições

```python
# Usar dropout em teste para estimar incerteza (MC Dropout)
def predict_with_uncertainty(model, input_tensor, num_samples=10):
    """Faz múltiplas predições com dropout ativo para estimar incerteza"""
    model.train()  # Manter dropout ativo
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(input_tensor)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred

# Usar
mean_pred, uncertainty = predict_with_uncertainty(model, input_tensor)
print(f"Predição média: {mean_pred[0, 0]:.4f}")
print(f"Incerteza (std): {uncertainty[0, 0]:.4f}")
```

## 10. Profile de Performance

```python
import time

# Medir tempo de inferência
num_iterations = 100
start = time.time()

with torch.no_grad():
    for _ in range(num_iterations):
        output = model(input_tensor)

end = time.time()
avg_time = (end - start) / num_iterations * 1000  # em ms

print(f"Tempo médio de inferência: {avg_time:.2f} ms")
print(f"Throughput: {1000/avg_time:.0f} samples/segundo")

# Contagem de parâmetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal de parâmetros: {total_params:,}")
print(f"Parâmetros treináveis: {trainable_params:,}")
```

---

## Referências Úteis

- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **Transferência de Aprendizado**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **Data Loading**: https://pytorch.org/docs/stable/data.html
- **Modelos Pré-treinados**: https://pytorch.org/vision/stable/models.html

