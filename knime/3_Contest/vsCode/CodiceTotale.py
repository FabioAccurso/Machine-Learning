import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Normalize01(torch.nn.Module):
    """Normalizza i valori di un tensore tra 0 e 1."""
    def __init__(self):
        super().__init__()

    def forward(self, img):
        min_val, max_val = img.min(), img.max()
        return (img - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
def check_tensor_range(img):
    """ Controlla se i valori del tensore sono nel range [0,1] dopo ToTensor. """
    if img.min() < 0 or img.max() > 1:
        print("Warning: Tensor values are out of range [0,1] after ToTensor")

def lambda_check_range(x):
    return (check_tensor_range(x), x)[1]

class PollinateDataset(Dataset):
    def __init__(self, csv_file, dataset_folder, channels=('T', 'M', 'N'), selected_ids=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.dataset_folder = dataset_folder
        self.channels = channels
        self.transform = transform
        self.min_size = (100, 100)  # Dimensione minima richiesta
        # Use only selected IDs if provided
        if selected_ids is not None:  #lasciare a None per usare tutto il dataset
            self.df = self.df[self.df['folder'].isin(selected_ids)]

    def __len__(self):
        return len(self.df)

    def pad_image_to_min_size(self, image, min_size=(100, 100)):
        """Aggiunge padding all'immagine se è più piccola delle dimensioni minime richieste."""
        h, w = image.shape[:2]
        pad_h = max(0, min_size[0] - h)
        pad_w = max(0, min_size[1] - w)

        if pad_h > 0 or pad_w > 0:
            # Calcola il padding su ogni lato per centrare l'immagine
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row['folder'])
        label = row['class']
        folder_path = os.path.join(self.dataset_folder, sample_id)

        imgs = []
        for ch in self.channels:
            path = os.path.join(folder_path, f"{sample_id}_{ch}.npy")
            img = np.load(path)
            img = self.pad_image_to_min_size(img, self.min_size)
            imgs.append(img)

        img_stack = np.stack(imgs, axis=0)

        # Convert to float32 and normalize to [0,1] if not done already
        img_tensor = torch.from_numpy(img_stack).float()
        if self.transform:
            img_tensor = self.transform(img_tensor)
        else:
            #img_tensor = torch.from_numpy(img_stack).float()
            if img_tensor.max() > 1.0:
                img_tensor /= 255.0

        return row['folder'], img_tensor, label


dataset_folder = r"C:\Users\Fabio\Desktop\Magistrale\Machine_Learning\esercizi\knime\3_Contest"
channels = ('T', 'M', 'N') #da modificare in base alla scelta dei canali
train_transform = transforms.Compose([
    transforms.Lambda(lambda_check_range),
    transforms.Resize((224, 224)),  # Porta tutto a 224x224 per il modello (ATTENZIONE: Da modificare in base alla scelta della soluzione)
    Normalize01()
])
train_dataset = PollinateDataset(os.path.join(dataset_folder,'train.csv'), os.path.join(dataset_folder, "train"), channels, transform=train_transform)


print(f"Numero totale di campioni: {len(train_dataset)}")

# Analisi su un esempio
folder_id, img_tensor, label = train_dataset[0]
print(f"ID campione: {folder_id}")
print(f"Forma del tensore immagine: {img_tensor.shape}")
print(f"Tipo del tensore: {type(img_tensor)}")
print(f"Tipo dati contenuti: {img_tensor.dtype}")
print(f"Etichetta associata: {label}")

'''
img_np = img_tensor.numpy()
if img_np.shape[0] == 1:
    # Se canale singolo, togli dimensione canale
    img_to_show = img_np[0]
elif img_np.shape[0] == 3:
    # Trasponi [C,H,W] -> [H,W,C]
    img_to_show = np.transpose(img_np, (1, 2, 0))
else:
    # Per altri casi, mostra il primo canale
    img_to_show = img_np[0]

plt.imshow(img_to_show, cmap='gray')
plt.title(f"ID: {folder_id}, Label: {label}")
plt.axis('off')
plt.show()


fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(min(144, len(train_dataset))):  # fino a 144 o numero totale campioni
    folder_id, img_tensor, label = train_dataset[i]
    img_np = img_tensor.numpy()
    # Trasponi i canali per matplotlib (C,H,W) -> (H,W,C)
    if img_np.shape[0] == 1:
        img_to_show = img_np[0]
        cmap = 'gray'
    else:
        img_to_show = np.transpose(img_np, (1, 2, 0))
        cmap = None  # se RGB, non serve cmap
    ax = fig.add_subplot(12, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(img_to_show, cmap=cmap, interpolation='nearest')
    ax.set_title(str(label), fontsize=6)
plt.show()
'''

