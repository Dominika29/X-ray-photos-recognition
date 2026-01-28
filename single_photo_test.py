import torch
import torchvision
from PIL import Image, ImageOps
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
# Importujemy obiekty z Twojego pliku głównego
from main import resnet, vgg16, dense_net, device, test_transform, CLASS_NAMES

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Wybierz zdjęcie RTG",
        filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return file_path

def preprocess_image(image_path):
    """Wymusza skalę szarości i rozmiar 224x224."""
    try:
        img = Image.open(image_path)
        
        # 1. Konwersja na skalę szarości (usuwa niebieski/kolory)
        img = ImageOps.grayscale(img)
        
        # 2. Wymuszenie rozmiaru 224x224
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # 3. Powrót do RGB (model wymaga 3 kanałów, ale będą one teraz szare)
        img = img.convert('RGB')
        
        return img
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None

def predict_and_show(image_path, model_path, model_architecture):
    # 1. Załadowanie wag modelu
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku {model_path}")
        return
    
    model_architecture.load_state_dict(torch.load(model_path, map_location=device))
    model_architecture.eval()
    
    # 2. Przygotowanie zdjęcia (Szary + 224x224)
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return

    # Normalizacja zgodna z treningiem
    image_tensor = test_transform(processed_img).unsqueeze(0).to(device) 

    # 3. Predykcja
    with torch.no_grad():
        output = model_architecture(image_tensor)
        probability = torch.sigmoid(output).item()
        
        # Zakładając: 0 = fractured, 1 = not fractured
        prediction_idx = 0 if probability < 0.5 else 1
        conf = (1 - probability) if prediction_idx == 0 else probability

    # 4. Wyświetlanie wyniku
    plt.figure(figsize=(8, 8))
    plt.imshow(processed_img) # Pokazuje dokładnie to, co widzi model
    
    label = CLASS_NAMES[prediction_idx].upper()
    color = 'red' if prediction_idx == 0 else 'green'
    
    plt.text(112, 20, f"DIAGNOZA: {label}\nPewność: {conf*100:.2f}%", 
             fontsize=12, color='white', fontweight='bold',
             bbox=dict(facecolor=color, alpha=0.8, edgecolor='black'),
             ha='center')
    
    plt.axis('off')
    plt.title(f"Analiza (224x224, Grayscale): {os.path.basename(image_path)}")
    plt.show()

if __name__ == "__main__":
    wybrany_plik = select_file()
    if wybrany_plik:
        # Możesz zmienić model na vgg16 lub dense_net
        predict_and_show(wybrany_plik, "resnet50.hd5", resnet)