import os
import numpy as np
from pathlib import Path
import pandas as pd
import shutil
import random
import torch
import torchvision
import h5py
from PIL import Image, ImageFile
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix, ConfusionMatrixDisplay

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


CLASS_NAMES = ["fractured", "not fractured"]
class BoneXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, class_names, transform):
        self.transform = transform
        self.class_names = class_names
        self.all_image_data = []
        
        for index, c in enumerate(self.class_names):
            path = str(image_paths[c])
            images_in_class = os.listdir(path)
            print(f'Found {len(images_in_class)} {c} examples')
            
            for img_name in images_in_class:
                self.all_image_data.append((os.path.join(path, img_name), index))
        
    def __len__(self):
        return len(self.all_image_data)

    def __getitem__(self, index):
        image_path, label = self.all_image_data[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


fracatlas_root_dir = Path("./FracAtlas")
fracatlas_train_dir = Path(f"{fracatlas_root_dir}/train")
fracatlas_test_dir = Path(f"{fracatlas_root_dir}/test")
fracatlas_val_dir = Path(f"{fracatlas_root_dir}/val")
fracatlas_non_fractured_dir = Path(f"{fracatlas_root_dir}/not fractured")

output_dir = Path("./output")


train_image_filepaths = [x for x in os.listdir(f"{fracatlas_train_dir}/img")]
train_annotation_filepaths = [x for x in os.listdir(f"{fracatlas_train_dir}/ann")]

test_image_filepaths = [x for x in os.listdir(f"{fracatlas_test_dir}/img")]
test_annotation_filepaths = [x for x in os.listdir(f"{fracatlas_test_dir}/ann")]

train_image_filepaths.extend(test_image_filepaths)

# train + test combined datasets are all fractured
fractured_image_filepaths = train_image_filepaths

non_fractured_image_filepaths = [x for x in os.listdir(f"{fracatlas_non_fractured_dir}/img")]
non_fractured_annotation_filepaths = [x for x in os.listdir(f"{fracatlas_non_fractured_dir}/ann")]

for c in CLASS_NAMES:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)

for c in CLASS_NAMES:
    images = [x for x in os.listdir(os.path.join(output_dir, c))]
    selected_images = random.sample(images, int(.05*len(images)))
    
    for image in selected_images:
        source_path = os.path.join(output_dir, c, image)
        target_path = os.path.join(output_dir, 'test', c, image)
        shutil.move(source_path, target_path)


def main():
    print(f"count of fractured images: {len(fractured_image_filepaths)}")
    print(f"count of non fractured images: {len(non_fractured_image_filepaths)}")

os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

for c in CLASS_NAMES:
    os.makedirs(os.path.join(output_dir, 'test', c), exist_ok=True)

shutil.copytree(
    os.path.join(fracatlas_non_fractured_dir, 'img'), os.path.join(output_dir, "not fractured"), dirs_exist_ok=True
)

for i, src_dir in enumerate([fracatlas_test_dir, fracatlas_train_dir]):
    shutil.copytree(
        os.path.join(src_dir, 'img'), os.path.join(output_dir, "fractured"), dirs_exist_ok=True
    )

for c in CLASS_NAMES:
    images = [x for x in os.listdir(os.path.join(output_dir, c))]
    selected_images = random.sample(images, int(.05*len(images)))
    
    for image in selected_images:
        source_path = os.path.join(output_dir, c, image)
        target_path = os.path.join(output_dir, 'test', c, image)
        shutil.move(source_path, target_path)

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_transform = torchvision.transforms.Compose([
torchvision.transforms.Resize(size=(224,224)),
torchvision.transforms.RandomHorizontalFlip(),
torchvision.transforms.RandomRotation(15),
torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
torchvision.transforms.Resize(size=(224,224)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

train_dirs = {
    'not fractured': output_dir / 'not fractured',
    'fractured': output_dir / 'fractured'
}
print(train_dirs)
train_dataset = BoneXRayDataset(
train_dirs, CLASS_NAMES, train_transform
    )

test_dirs = {
    'not fractured': output_dir / 'test' / 'not fractured',
    'fractured': output_dir / 'test' / 'fractured'
}
print(test_dirs)
test_dataset = BoneXRayDataset(
test_dirs, CLASS_NAMES, test_transform
    )

batch_size = 6

dl_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
    )

dl_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
    )

print('Num of training batches', len(dl_train))
print('Num of test batches', len(dl_test))


def show_images(images, labels, preds):
    plt.figure(figsize=(8,4))
    
    for i, image in enumerate(images):
        plt.subplot(1,6, i + 1, xticks=[], yticks=[])
        
        image = image.numpy().transpose((1,2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        
        col = 'green' if preds[i] == labels[i] else 'red'
        
        plt.xlabel(f'{CLASS_NAMES[int(labels[i].numpy())]}')
        plt.ylabel(f'{CLASS_NAMES[int(preds[i].cpu().numpy())]}', color=col)
        
    plt.tight_layout()
    plt.show()


images, labels = next(iter(dl_train))
show_images(images, labels, labels)
image = images[0].numpy().transpose((1, 2, 0)) 
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
        
image = image * std + mean
image = np.clip(image, 0., 1.)
plt.imshow(image)

print(f"The following image has the following dimensions: {image.shape}\n")

resnet = torchvision.models.resnet50(weights='DEFAULT')
num_filters = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_filters, 1) 
resnet = resnet.to(device)
print(resnet) 

vgg16 = torchvision.models.vgg16(pretrained=True)
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = torch.nn.Linear(num_ftrs, 1) 
vgg16 = vgg16.to(device)

print(vgg16)

dense_net = torchvision.models.densenet121(pretrained=True)
num_ftrs = dense_net.classifier.in_features
dense_net.classifier = torch.nn.Linear(num_ftrs, 1) 
dense_net = dense_net.to(device)

print(dense_net)

def show_preds(model):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dl_test))
        outputs = model(images.to(device))
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().view(-1)
    show_images(images, labels, preds)

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for train_step, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs.to(device))
        targets = targets.to(device).view(-1, 1).float()
        
        loss = criterion(outputs.to(device), targets.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        preds = torch.sigmoid(outputs.to(device))
        preds = (preds > 0.5).float()
        
        correct += (preds == targets.to(device)).sum().item()
        total += targets.size(0)
        
        if train_step % 20 == 0:
            print('#', sep=' ', end='', flush=True)
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    print(f'\n\nTraining Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    return epoch_loss, epoch_accuracy

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for eval_step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Obliczanie predykcji (próg 0.5 dla Sigmoidy)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Zbieranie danych do macierzy (przenosimy na CPU i do numpy)
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            if eval_step % 20 == 0:
                print(f'Validation Step: {eval_step}, Accuracy: {(correct / total):.4f}')

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    # --- GENEROWANIE I WYŚWIETLANIE MACIERZY POMYŁEK ---
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Macierz Pomyłek - Model: {model.__class__.__name__}')
    plt.show()
    # --------------------------------------------------
    
    # Obliczanie dodatkowych metryk
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f'\nFinal Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    return epoch_loss, epoch_accuracy, epoch_accuracy, precision, recall, f1, roc_auc

def run_model(model, dl_train, dl_test, model_name: str, best_accuracy: float):
    weight_for_fractured = torch.tensor([1.0]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_for_fractured)
    total_epochs = 0
    
    learning_rate = 3e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while(True):
        print(f"\nCurrent Epoch: {total_epochs+1}")

        # TRAIN MODEL
        print("Training Model...")
        train_loss, train_accuracy = train_model(model, dl_train, criterion, optimizer)
        print("Model Training Complete")

        # EVALUATE MODEL
        print("Validating Model...")
        val_loss, val_accuracy, accuracy, precision, recall, f1, roc_auc = evaluate_model(
            model, dl_test, criterion
        )
        print("Model Validation Complete")

        print(f"\nEpoch {total_epochs+1} Summary")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        total_epochs += 1 # increment epochs

        # Save the best model
        if val_accuracy > best_accuracy:
            print(f'\n{model_name.upper()} Model performance condition met')
            
            best_accuracy = val_accuracy
            
            print(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )
            print(
                f"F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}"
            )
            break


    # Save the one with the best performance
    torch.save(model.state_dict(), f'{model_name}.hd5')

def balance_dataset(base_path):
    # Lista podfolderów do zbalansowania: główny i testowy
    folders_to_check = [base_path, base_path / 'test']
    
    for folder in folders_to_check:
        if not folder.exists():
            continue
            
        counts = {c: len(os.listdir(folder / c)) for c in CLASS_NAMES}
        min_count = min(counts.values())
        
        print(f"\nBalansowanie folderu: {folder}")
        for c in CLASS_NAMES:
            path = folder / c
            images = os.listdir(path)
            if len(images) > min_count:
                to_remove = random.sample(images, len(images) - min_count)
                for img in to_remove:
                    os.remove(path / img)
            print(f" Klasa {c}: pozostało {len(os.listdir(path))} zdjęć.")


if __name__ == "__main__":
    main()
    balance_dataset(output_dir)
    balance_dataset(output_dir / 'test')
    train_dataset = BoneXRayDataset(train_dirs, CLASS_NAMES, train_transform)
    test_dataset = BoneXRayDataset(test_dirs, CLASS_NAMES, test_transform)
    
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    run_model(resnet, dl_train, dl_test, "resnet50", .80)
    run_model(vgg16, dl_train, dl_test, "vgg16", .80)
    run_model(dense_net, dl_train, dl_test, "dense_net", .80)