import os
import numpy as np
from pathlib import Path
import shutil
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score

# Ustawienia środowiska
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- AUTORSKA ARCHITEKTURA CNN ---
class BoneNetCNN(nn.Module):
    def __init__(self):
        super(BoneNetCNN, self).__init__()
        # Warstwa 1: Wykrywanie krawędzi (224x224 -> 112x112 po poolingu)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Warstwa 2: Wykrywanie tekstur (112x112 -> 56x56)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Warstwa 3: Złożone wzory pęknięć (56x56 -> 28x28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Warstwy klasyfikacyjne (Dense)
        # 128 filtrów * obraz 28x28 = 100352 wejścia
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1) # Binarna klasyfikacja

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # Spłaszczenie
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- DATASET I TRANSFORMACJE ---
CLASS_NAMES = ["fractured", "not fractured"]

class BoneXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths_dict, class_names, transform):
        self.transform = transform
        self.class_names = class_names
        self.all_image_data = []
        
        for index, c in enumerate(self.class_names):
            path = str(image_paths_dict[c])
            if os.path.exists(path):
                images_in_class = os.listdir(path)
                for img_name in images_in_class:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.all_image_data.append((os.path.join(path, img_name), index))
        
    def __len__(self):
        return len(self.all_image_data)

    def __getitem__(self, index):
        image_path, label = self.all_image_data[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Ścieżki
fracatlas_root_dir = Path("./FracAtlas")
output_dir = Path("./output")

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- FUNKCJE POMOCNICZE ---
def balance_dataset(folder_path):
    for folder in [folder_path, folder_path / 'test']:
        if not folder.exists(): continue
        counts = {c: len(os.listdir(folder / c)) for c in CLASS_NAMES if (folder / c).exists()}
        if not counts: continue
        min_count = min(counts.values())
        print(f"Balansowanie {folder}: cel {min_count} zdjęć na klasę.")
        for c in CLASS_NAMES:
            path = folder / c
            images = os.listdir(path)
            if len(images) > min_count:
                to_remove = random.sample(images, len(images) - min_count)
                for img in to_remove: os.remove(path / img)

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device).view(-1, 1).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total

def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.to(device))
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(targets.numpy())
            all_preds.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f} | Precision: {precision_score(all_labels, all_preds):.4f}")
    
    # Wyświetlanie Macierzy
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues')
    plt.title(f"Macierz Pomyłek - {model.__class__.__name__}")
    plt.show()
    return acc

# --- GŁÓWNA PĘTLA ---
if __name__ == "__main__":
    # 1. Przygotowanie danych (uproszczone)
    for c in CLASS_NAMES:
        os.makedirs(output_dir / c, exist_ok=True)
        os.makedirs(output_dir / 'test' / c, exist_ok=True)

    # 2. Inicjalizacja modelu i treningu
    model = BoneNetCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # (Tu powinna być Twoja logika kopiowania plików z FracAtlas do output/...)
    # Zakładamy, że pliki są już w folderze output po Twoich wcześniejszych operacjach
    
    balance_dataset(output_dir)
    
    train_dirs = {'not fractured': output_dir / 'not fractured', 'fractured': output_dir / 'fractured'}
    test_dirs = {'not fractured': output_dir / 'test' / 'not fractured', 'fractured': output_dir / 'test' / 'fractured'}

    train_ds = BoneXRayDataset(train_dirs, CLASS_NAMES, train_transform)
    test_ds = BoneXRayDataset(test_dirs, CLASS_NAMES, test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

    print("Rozpoczynanie treningu autorskiej sieci BoneNetCNN...")
    for epoch in range(5): # Przykładowo 5 epok
        loss, acc = train_model(model, train_loader, criterion, optimizer)
        print(f"Epoka {epoch+1} | Loss: {loss:.4f} | Acc: {acc:.4f}")
    
    evaluate_model(model, test_loader, criterion)
    
    # Bezpieczne zapisywanie (używamy .pth zamiast .hd5, aby uniknąć problemów)
    # I dodajemy do gitignore!
    model_path = "bone_model_custom.pth"
    torch.save(model.state_dict(), model_path)
    
    with open(".gitignore", "a") as f:
        f.write(f"\n{model_path}\n*.hd5\noutput/\n")
    print(f"Model zapisany jako {model_path}. Pamiętaj: NIE wysyłaj go na GitHub!")