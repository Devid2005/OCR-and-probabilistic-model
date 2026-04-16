import torch
from torchvision import transforms
from PIL import Image

from model import Model
from weibull import predict_days
from config import *
from utils import extract_visual_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

t = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

GLOBAL_NAMES = {
    0: "good",
    1: "medium",
    2: "low",
    3: "critical"
}


def predict(img_path, env="ambiente"):

    img = t(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        prod, glob, v = model(img)

        prod = torch.argmax(prod).item()

        v = v.detach().cpu().numpy()

        if v.ndim == 0:
            s_model = m_model = w_model = c_model = float(v)

        elif v.ndim == 1:
            if len(v) == 4:
                s_model, m_model, w_model, c_model = v
            else:
                s_model = m_model = w_model = c_model = float(v[0])

        elif v.ndim == 2:
            s_model, m_model, w_model, c_model = v[0]

        else:
            raise ValueError(f"Forma inesperada: {v.shape}")

    s_img, m_img, w_img, c_img = extract_visual_metrics(img_path)

    alpha_model = 0.10
    alpha_img = 0.90

    s = alpha_model * float(s_model) + alpha_img * float(s_img)
    m = alpha_model * float(m_model) + alpha_img * float(m_img)
    w = alpha_model * float(w_model) + alpha_img * float(w_img)
    c = alpha_model * float(c_model) + alpha_img * float(c_img)

    score = 0.6*s + 1.5*m + 0.8*w + 0.5*c

    if score < 0.3:
        glob = 0
    elif score < 0.6:
        glob = 1
    elif score < 1.0:
        glob = 2
    else:
        glob = 3

    product = PRODUCT_MAP_INV[prod]

    days = predict_days(product, s, m, w, c, glob, env)

    return {
        "producto": product,
        "estado_global": GLOBAL_NAMES.get(glob, str(glob)),
        "ambiente": env,   
        "dias": round(float(days), 2),
        "stains": round(float(s), 4),
        "mold": round(float(m), 4),
        "wilt": round(float(w), 4),
        "color": round(float(c), 4),

        "metricas_modelo": {
            "stains": round(float(s_model), 4),
            "mold": round(float(m_model), 4),
            "wilt": round(float(w_model), 4),
            "color": round(float(c_model), 4),
        },

        "metricas_imagen": {
            "stains": round(float(s_img), 4),
            "mold": round(float(m_img), 4),
            "wilt": round(float(w_img), 4),
            "color": round(float(c_img), 4),
        }
    }


if __name__ == "__main__":

    print("\n--- AMBIENTE NORMAL ---")
    print(predict("test_6.jpeg", env="ambiente"))

    print("\n--- EN NEVERA ---")
    print(predict("test_6.jpeg", env="nevera"))