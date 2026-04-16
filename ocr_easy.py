import easyocr
import cv2
import re
import json
import unicodedata
import os
from collections import defaultdict
from difflib import SequenceMatcher


reader = easyocr.Reader(['es', 'en'], gpu=False)


FOOD_CATEGORIES = {
    "fruta": {
        "manzana", "platano", "pera", "limon", "uva",
        "fresa", "papaya", "sandia", "melon", "mango",
        "granadilla", "guanabana", "banano"
    },
    "verdura": {
        "lechuga", "tomate", "cebolla", "zanahoria", "papa", "pepino",
        "brocoli", "espinaca", "apio", "arveja"
    },
    "carne": {
        "pollo", "res", "cerdo", "carne", "chorizo", "jamon"
    },
    "lacteo": {
        "leche", "queso", "yogur", "yogurt", "mantequilla", "kumis"
    },
    "grano": {
        "arroz", "pasta", "frijol", "lenteja", "garbanzo"
    }
}

FOOD_TO_CATEGORY = {}
for category, foods in FOOD_CATEGORIES.items():
    for food in foods:
        FOOD_TO_CATEGORY[food] = category

ABBREVIATIONS = {
    "pqt", "pqte", "und", "kg", "gr", "g", "lb", "granel", "bja",
    "fv", "un", "u", "mr", "dj", "cj"
}

NOISE_WORDS = {
    "descripcion", "precio", "total", "subtotal", "descuento", "descuento",
    "nit", "factura", "electronica", "electronic", "caj", "caja", "vendedor",
    "regimen", "comun", "domicilio", "telefono", "tel", "fecha", "premiun",
    "premium", "sede", "mazuren", "mazuren", "cliente", "unid", "unidad",
    "tarjeta", "efectivo", "debito", "credito", "iva", "calle"
}

MONTHS_ES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "setiembre": "09", "octubre": "10",
    "noviembre": "11", "diciembre": "12"
}


def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_basic(text: str) -> str:
    text = text.lower().strip()
    text = strip_accents(text)
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_token(token: str) -> str:
    token = normalize_basic(token)

    if re.fullmatch(r'[\d\W_]+', token or ""):
        return token

    replacements = str.maketrans({
        '0': 'o',
        '1': 'l',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '6': 'g',
        '7': 't',
        '8': 'b'
    })
    token = token.translate(replacements)

    token = re.sub(r'[^a-zñ]', '', token)

    return token

def normalize_line(text: str) -> str:
    text = normalize_basic(text)
    text = text.replace(",", ".")
    text = re.sub(r'[|`´\'"~^_*;:(){}\[\]\\/+]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_line(text: str):
    text = normalize_line(text)
    raw_tokens = text.split()

    tokens = []
    for tok in raw_tokens:
        cleaned = normalize_token(tok)
        if not cleaned:
            continue
        if cleaned in ABBREVIATIONS:
            continue
        if cleaned in NOISE_WORDS:
            continue
        tokens.append(cleaned)

    return tokens

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def load_images_from_folder(folder_path):
    valid_ext = (".jpg", ".jpeg", ".png", ".webp")
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    ]
    files.sort()
    return files


def fix_orientation(image):
    h, w = image.shape[:2]
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")

    img = fix_orientation(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return thresh


def ocr_image(path):
    img = preprocess(path)
    return reader.readtext(
        img,
        detail=0,
        paragraph=False
    )

def ocr_multiple(image_paths):
    all_lines = []
    for img in image_paths:
        lines = ocr_image(img)
        all_lines.extend(lines)
    return all_lines


def extract_date(text):
    text_norm = normalize_basic(text)

    patterns = [
        r'(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(\d{4}[/-]\d{2}[/-]\d{2})'
    ]
    for pat in patterns:
        match = re.search(pat, text_norm)
        if match:
            return match.group(1).replace('-', '/')

    match = re.search(
        r'(\d{1,2})\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(\d{4})',
        text_norm
    )
    if match:
        day = int(match.group(1))
        month = MONTHS_ES[match.group(2)]
        year = match.group(3)
        return f"{day:02d}/{month}/{year}"

    return None


def find_best_food_match(tokens):
    best_food = None
    best_category = None
    best_score = 0.0

    for token in tokens:
        for food, category in FOOD_TO_CATEGORY.items():
            score = 0.0

            if token == food:
                score = 1.0
            elif food in token or token in food:
                score = 0.92
            else:
                score = similarity(token, food)

            if score > best_score:
                best_score = score
                best_food = food
                best_category = category

    if best_score >= 0.74:
        return best_food, best_category, best_score

    return None, None, 0.0

def classify_food(line):
    tokens = tokenize_line(line)
    if not tokens:
        return None, None

    food, category, _ = find_best_food_match(tokens)
    return food, category


def extract_measure(line):
    line_norm = normalize_line(line)

    decimal_matches = re.findall(r'\b\d+\.\d+\b', line_norm)
    for m in decimal_matches:
        try:
            val = float(m)
            if 0 < val <= 10:
                return "peso", val
        except ValueError:
            pass

    int_matches = re.findall(r'\b\d+\b', line_norm)
    small_ints = []
    for m in int_matches:
        try:
            val = int(m)
            if 1 <= val <= 24:
                small_ints.append(val)
        except ValueError:
            pass

    if small_ints:
        return "unidad", small_ints[0]

    return "unidad", 1


def should_skip_line(line):
    norm = normalize_basic(line)
    if not norm:
        return True

    alpha_count = len(re.findall(r'[a-zA-ZñÑ]', norm))
    if alpha_count < 3:
        return True

    for noise in NOISE_WORDS:
        if noise in norm:
            return True

    return False

def extract_products(lines):
    products = {}

    for line in lines:
        if should_skip_line(line):
            continue

        food_name, category = classify_food(line)
        if not food_name:
            continue

        measure_type, value = extract_measure(line)
        if measure_type is None:
            continue

        key = (food_name, measure_type)

        if key not in products:
            products[key] = {
                "name": food_name,
                "category": category
            }


    result = list(products.values())
    result.sort(key=lambda x: (x["category"], x["name"]))
    return result

def detect_completeness(lines):
    text = " ".join(lines).lower()
    score = 0

    if "total" in text:
        score += 1
    if any(x in text for x in ["efectivo", "tarjeta", "debito", "credito"]):
        score += 1
    if len(lines) > 15:
        score += 1

    return score >= 2


def process_receipt(input_data):
    if os.path.isdir(input_data):
        image_paths = load_images_from_folder(input_data)
    else:
        image_paths = [input_data]

    lines = ocr_multiple(image_paths)
    full_text = "\n".join(lines)

    return {
        "purchase_date": extract_date(full_text),
        "products": extract_products(lines),
        "is_complete": detect_completeness(lines),
        "images_processed": image_paths
    }


if __name__ == "__main__":
    result = process_receipt("recibo/")
    print(json.dumps(result, indent=4, ensure_ascii=False))