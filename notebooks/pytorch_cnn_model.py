import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader, random_split

# ===== MODELO CNN EM PYTORCH =====

class FireCNN(nn.Module):
    """
    CNN para predição de ocorrência de queimadas
    Input: (batch_size, 6, 50, 50) - 6 canais (ex.: semestres ou features agregadas)
    Output: Probabilidade de queimada (sigmoid)
    """
    def __init__(self):
        super(FireCNN, self).__init__()
        
        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Segunda camada convolucional
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camadas totalmente conectadas
        self.flatten = nn.Flatten()
        # Entrada flatten: 16 canais * 12 * 12 (50 -> 25 -> 12 apos pooling)
        self.fc1 = nn.Linear(16 * 12 * 12, 32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Conv block 1: (batch, 6, 50, 50) -> (batch, 8, 25, 25)
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # Conv block 2: (batch, 8, 25, 25) -> (batch, 16, 12, 12)
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten: (batch, 16*12*12)
        x = self.flatten(x)
        
        # Fully connected: -> (batch, 32) -> (batch, 1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x

# ===== DATASET CUSTOMIZADO =====

class FireDataset(Dataset):
    """Dataset para queimadas em formato de grid espacial-temporal"""
    def __init__(self, heat_maps, labels):
        self.heat_maps = torch.FloatTensor(heat_maps)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.heat_maps)
    
    def __getitem__(self, idx):
        return self.heat_maps[idx], self.labels[idx]

# ===== PREPARAÇÃO DOS DADOS =====

def prepare_data_for_training(df, year_train=2023, year_val=2024):
    """
    Prepara grids espacial-temporais para treino e validação
    
    Args:
        df: DataFrame Polars com colunas 'ano', 'mes', 'latitude', 'longitude'
        year_train: Ano para treino (padrão: 2023)
        year_val: Ano para validação (padrão: 2024)
    
    Returns:
        train_loader, val_loader: DataLoaders do PyTorch
    """
    
    # Definir bins para latitude e longitude (Brasil)
    lat_bins = np.linspace(-33, 5, 51)  # 51 edges para 50 bins
    lon_bins = np.linspace(-75, -30, 51)
    
    # Função para criar heat_map de um ano
    def create_heat_maps(data_year):
        heat_maps = []
        num_months = 12
        
        # Agrupar por ano/mês
        for year in data_year['ano'].unique():
            monthly_grids = np.zeros((num_months, 50, 50))
            
            for month in range(1, 13):
                data_month = data_year.filter(
                    (pl.col('ano') == year) & 
                    (pl.col('mes') == month)
                )
                
                if len(data_month) > 0:
                    # Histograma 2D da densidade de queimadas
                    counts, _, _ = np.histogram2d(
                        data_month['latitude'].to_numpy(),
                        data_month['longitude'].to_numpy(),
                        bins=[lat_bins, lon_bins]
                    )
                    monthly_grids[month-1] = counts
            
            # Normalizar
            max_val = monthly_grids.max()
            if max_val > 0:
                monthly_grids = monthly_grids / max_val
            
            heat_maps.append(monthly_grids)
        
        return np.array(heat_maps)
    
    # Preparar dados de treino (2023)
    df_train = df.filter(pl.col('ano') == year_train)
    X_train = create_heat_maps(df_train)
    y_train = np.ones(len(X_train))  # Label: ocorrência de queimadas
    
    # Preparar dados de validação (2024)
    df_val = df.filter(pl.col('ano') == year_val)
    X_val = create_heat_maps(df_val)
    y_val = np.ones(len(X_val))
    
    # Criar datasets
    train_dataset = FireDataset(X_train, y_train)
    val_dataset = FireDataset(X_val, y_val)
    
    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

# ===== EXEMPLO DE USO =====
"""
# No notebook, use assim:

# Supondo que 'df' já está carregado do parquet
train_loader, val_loader = prepare_data_for_training(df, year_train=2023, year_val=2024)

# Criar modelo e otimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

model = FireCNN().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Treinar (ver arquivo train_pytorch_cnn.py para loop de treinamento)
"""
