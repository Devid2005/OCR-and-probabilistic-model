from pathlib import Path

DATASET_ROOT = Path("data/Dataset")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

MAX_PER_CLASS = 50  

PRODUCT_MAP = {
    "manzana":0,"banano":1,"pimenton":2,
    "zanahoria":3,"pepino":4,"mango":5,
    "naranja":6,"papa":7,"tomate":8
}

PRODUCT_MAP_INV = {v:k for k,v in PRODUCT_MAP.items()}

BASE_PARAMS = {
    "manzana":(14,2.0),"banano":(5,2.2),"pimenton":(8,2.0),
    "zanahoria":(10,1.8),"pepino":(6,1.9),"mango":(6,2.3),
    "naranja":(12,2.0),"papa":(20,1.5),"tomate":(7,2.0)
}