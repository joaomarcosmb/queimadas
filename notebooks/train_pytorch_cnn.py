# Script para treinar o modelo CNN em PyTorch
# Copie este código em uma célula do notebook após definir o modelo

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Acurácia
        predicted = (outputs > 0.5).float()
        correct += (predicted == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Treinar o modelo
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("Iniciando treinamento...\n")

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Treino - Loss: {train_loss:.4f}, Acurácia: {train_acc:.4f}")
    print(f"  Validação - Loss: {val_loss:.4f}, Acurácia: {val_acc:.4f}\n")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Salvar modelo
        torch.save(model.state_dict(), '../models/best_cnn_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping na época {epoch+1}")
            break

print("\nTreinamento concluído!")
print(f"Melhor validação loss: {best_val_loss:.4f}")
