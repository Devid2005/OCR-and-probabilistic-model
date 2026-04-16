import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score
import numpy as np

from dataset import FoodDataset
from model import Model
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = FoodDataset(ARTIFACTS/"data.csv")
dl = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)

model = Model().to(device)

ce = nn.CrossEntropyLoss()
mse = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(),lr=LR)

for epoch in range(EPOCHS):

    preds,targets = [],[]
    reg_p,reg_t = [],[]

    for x,p,y,vis in dl:
        x,p,y,vis = x.to(device),p.to(device),y.to(device),vis.to(device)

        opt.zero_grad()

        prod,glob,v = model(x)

        loss = ce(prod,p)+ce(glob,y)+0.7*mse(v,vis)

        loss.backward()
        opt.step()

        preds += list(torch.argmax(glob,1).cpu().numpy())
        targets += list(y.cpu().numpy())

        reg_p.append(v.detach().cpu().numpy())
        reg_t.append(vis.cpu().numpy())

    acc = accuracy_score(targets,preds)
    f1 = f1_score(targets,preds,average="macro")

    reg_p = np.vstack(reg_p)
    reg_t = np.vstack(reg_t)

    mae = np.mean(np.abs(reg_p-reg_t))
    rmse = np.sqrt(np.mean((reg_p-reg_t)**2))

    print(f"Epoch {epoch} ACC:{acc:.3f} F1:{f1:.3f} MAE:{mae:.3f} RMSE:{rmse:.3f}")

torch.save(model.state_dict(),"model.pth")