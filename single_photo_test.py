import torch
import torchvision
from PIL import Image, ImageOps
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from main import resnet, device, test_transform, CLASS_NAMES

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

        img = ImageOps.grayscale(img)
        

        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        

        img = img.convert('RGB')
        
        return img
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None

def predict_and_show(image_path, model_path, model_architecture):
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku {model_path}")
        return
    
    model_architecture.load_state_dict(torch.load(model_path, map_location=device))
    model_architecture.eval()
    
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return

    image_tensor = test_transform(processed_img).unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model_architecture(image_tensor)
        probability = torch.sigmoid(output).item()
        
        prediction_idx = 0 if probability < 0.5 else 1
        conf = (1 - probability) if prediction_idx == 0 else probability

    plt.figure(figsize=(8, 8))
    plt.imshow(processed_img)
    
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
        predict_and_show(wybrany_plik, "resnet50.hd5", resnet)