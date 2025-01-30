import torch
import models
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#params
map=torch.device("mps")
MODEL_PATH = "/Users/shaffeb1/Documents/wfdb_pretrain/outputs_12_20/model_2024-12-20_12-56.pth"
FILE_PATH = "/Users/shaffeb1/Documents/wfdb_pretrain/data/test/norm/"

AFIB_PATH = "/Users/shaffeb1/Documents/wfdb_pretrain/data/test/afib/"


model = models.ECG_Model()
checkpoint = torch.load(MODEL_PATH, map_location=map)
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)

# generate file list
files = os.listdir(FILE_PATH)
files = [f for f in files if f.endswith('.csv')]

afib_files = os.listdir(AFIB_PATH)
afib_files = [f for f in afib_files if f.endswith('.csv')]



# inferrence loop
results = []
for file in files:
    data = pd.read_csv(FILE_PATH + file)
    data_t = torch.tensor(data.values, dtype=torch.float32)
    data_t = data_t.view(1,1,*data_t.shape)
    with torch.no_grad():
        out = model(data_t)
    results.append(out)
    
results_out = [result.item() for result in results]

results_afib = []
for file in afib_files:
    data = pd.read_csv(AFIB_PATH + file)
    data_t = torch.tensor(data.values, dtype=torch.float32)
    data_t = data_t.view(1,1,*data_t.shape)
    with torch.no_grad():
        out = model(data_t)
    results_afib.append(out)

results_afib_out = [result.item() for result in results_afib]


plt.figure(figsize=(10, 5))
plt.hist(results_out,bins=100,density=True,alpha=0.5,color='blue',label='inferrence')
plt.xlabel('Value')
plt.title("Density Plot")
plt.ylabel('Density')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))

#sns.kdeplot(results_out, ax=ax, label='normal')
sns.kdeplot(results_afib_out, ax=ax, label='afib')
plt.legend()
plt.show()

results_afib_out

df = zip(afib_files,results_afib_out)
df = pd.DataFrame(df,columns=['file','prob'])
df.to_csv('afib_prob.csv',index=False)