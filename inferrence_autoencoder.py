import torch
import models
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision as tv
from wfdb.processing import normalize_bound
import glob

#params
map=torch.device("mps")
MODEL_PATH = "outputs/model_2025-01-01_21-27.pth"


model = models.ECG_Autoencoder()
checkpoint = torch.load(MODEL_PATH, map_location=map)
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)

BATCH_SIZE = 32
TRAIN_DATA_PATH = 'all_muse_records/csv_records/'

class UnsqueezeImage(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(0)
        return x   

transform_img = tv.transforms.Compose([
    tv.transforms.Lambda(lambda x: torch.tensor(normalize_bound(x.iloc[:,[0,1,6,7,8,9,10,11]].values, lb=0, ub=1), dtype=torch.float32)),
    UnsqueezeImage(),
    tv.transforms.Resize((5000,8), antialias=True),
])

train_dataset = tv.datasets.DatasetFolder(root=TRAIN_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

embeddings = []
x=0
for i, data in enumerate(train_loader):
    image, _ = data
    with torch.no_grad():
        output = model(image)[1]
    embeddings.append(output)
    x+=1
    if x==5000:
        break

img = pd.read_csv("all_muse_records/csv_records/allo_merged_csv/MUSE_20230629_173234_72000.csv", header=0)

img_norm = normalize_bound(img.iloc[:,[0,1,6,7,8,9,10,11]].values, lb=0, ub=1)

plt.scatter(range(5000), img_norm[:,0])
plt.show()

img = transform_img(img)
img = img.unsqueeze(0)
img_post = model(img)[0]
img_post = img_post.squeeze(0).squeeze(0).cpu().detach().numpy()

plt.scatter(range(5000), img_post[:,0])
plt.show()

embeddings = torch.cat(embeddings).cpu().detach().numpy()

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

fig = plt.figure(figsize=(12, 10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(embeddings_tsne[:,0], embeddings_tsne[:,1])
plt.colorbar(ticks=range(10), drawedges=True)
plt.show()

import os

files = [file for file in os.listdir(TRAIN_DATA_PATH) if file.endswith('.csv')]

pth = 'all_muse_records/csv_records/allo_merged_csv/*.csv'

files = files[:len(embeddings)]

df = pd.DataFrame({'file':files, 'x':embeddings_tsne[:,0], 'y':embeddings_tsne[:,1]})

df.to_csv('embeddings.csv', index=False)

df = pd.read_csv('embeddings.csv')

df['class_num'].unique()

fig = plt.figure(figsize=(12, 10))
cmap = plt.get_cmap('Spectral', 6)
plt.scatter(df['x'], df['y'], c=df['class_num'], cmap=cmap)
plt.colorbar()
plt.show()