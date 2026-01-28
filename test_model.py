import torch
import torchvision
import os
from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importujemy klasę Dataset z Twojego głównego pliku
# Upewnij się, że test_model.py jest w tym samym folderze co main.py
from main import BoneXRayDataset, CLASS_NAMES, test_transform

# Konfiguracja urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, model_type):
    """Wczytuje architekturę i zapisane wagi modelu."""
    if model_type == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
        num_filters = model.fc.in_features
        model.fc = torch.nn.Linear(num_filters, 1)
    elif model_type == 'vgg16':
        model = torchvision.models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, 1)
    elif model_type == 'dense_net':
        model = torchvision.models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, 1)
    
    # Wczytanie wag (state_dict)
    model_path = f"{model_name}.hd5"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Pomyślnie załadowano: {model_path}")
        return model
    else:
        print(f"Błąd: Nie znaleziono pliku {model_path}")
        return None

def run_final_test(model, dataloader):
    """Przeprowadza test na całym zbiorze i zwraca metryki."""
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().view(-1)
            
            all_labels.extend(targets.numpy())
            all_preds.extend(preds.numpy())
            
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds),
        'Recall': recall_score(all_labels, all_preds),
        'F1': f1_score(all_labels, all_preds)
    }
    return metrics

if __name__ == "__main__":
    # 1. Przygotowanie danych testowych (używając ścieżek z Twojego projektu)
    output_dir = Path("./output")
    test_dirs = {
        'not fractured': output_dir / 'test' / 'not fractured',
        'fractured': output_dir / 'test' / 'fractured'
    }
    
    test_dataset = BoneXRayDataset(test_dirs, CLASS_NAMES, test_transform)
    dl_test = DataLoader(test_dataset, batch_size=6, shuffle=False)
    
    # 2. Lista modeli do sprawdzenia
    models_to_test = [
        ('resnet50', 'resnet50'),
        ('vgg16', 'vgg16'),
        ('dense_net', 'dense_net')
    ]
    
    results = []

    print("\n--- ROZPOCZYNAM TESTOWANIE MODELI ---")
    for name, m_type in models_to_test:
        model = load_model(name, m_type)
        if model:
            print(f"Testowanie {name}...")
            m_metrics = run_final_test(model, dl_test)
            m_metrics['Model'] = name
            results.append(m_metrics)

    # 3. Wyświetlenie wyników w tabeli
    if results:
        df = pd.DataFrame(results)
        # Reorganizacja kolumn dla czytelności
        df = df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']]
        print("\nOstateczne wyniki porównawcze:")
        print(df.to_string(index=False))
        
        # Opcjonalnie zapis do CSV
        df.to_csv('porownanie_modeli.csv', index=False)
        print("\nWyniki zostały zapisane do pliku porownanie_modeli.csv")