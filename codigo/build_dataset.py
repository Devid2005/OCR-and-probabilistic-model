import random
import pandas as pd
from pathlib import Path

from config import *
from utils import extract_visual_metrics

rows = []

def get_product(name):
    name = name.lower()
    if "apple" in name: return "manzana"
    if "banana" in name: return "banano"
    if "pepper" in name: return "pimenton"
    if "carrot" in name: return "zanahoria"
    if "cucumber" in name: return "pepino"
    if "mango" in name: return "mango"
    if "orange" in name: return "naranja"
    if "potato" in name: return "papa"
    if "tomato" in name: return "tomate"
    return None

for folder in DATASET_ROOT.rglob("*"):
    if folder.is_dir():

        imgs = list(folder.glob("*.jpg"))
        if not imgs: continue

        imgs = random.sample(imgs, min(MAX_PER_CLASS,len(imgs)))

        product = get_product(folder.name)
        if product is None: continue

        for img in imgs:
            try:
                s,m,w,c = extract_visual_metrics(str(img))

                score = 0.8*s + 2*m + 1.2*w + c

                if score<0.3: g=0
                elif score<0.6: g=1
                elif score<1.0: g=2
                else: g=3

                rows.append({
                    "path":str(img),
                    "product":product,
                    "global":g,
                    "stains":s,"mold":m,"wilt":w,"color":c
                })
            except:
                continue

df = pd.DataFrame(rows)
df.to_csv(ARTIFACTS/"data.csv",index=False)
print("Dataset listo:",len(df))