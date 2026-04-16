import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from config import *

class FoodDataset(Dataset):
    def __init__(self,csv):
        self.df = pd.read_csv(csv)

        self.t = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self,i):
        r = self.df.iloc[i]

        x = self.t(Image.open(r["path"]).convert("RGB"))

        product = PRODUCT_MAP[r["product"]]

        y = int(r["global"])

        # 🔥 AQUÍ ESTABA EL ERROR
        vis = torch.tensor(
            [r["stains"], r["mold"], r["wilt"], r["color"]],
            dtype=torch.float32
        )

        return x,product,y,vis