import importlib, subprocess, sys
import json
import os
import warnings
import re
import unicodedata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import cv2
from pathlib import Path
from PIL import Image
from joblib import dump, load
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
from nltk.stem.snowball import SnowballStemmer

import token
from wordfreq import zipf_frequency

from sentence_transformers import SentenceTransformer
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from TFM.Utils.ocr_utils import *

###############################################################
######################## OCR MODEL UTILS ######################
###############################################################

def ensure_package(pkg, import_name=None):
    module_name = import_name or pkg
    try:
        importlib.import_module(module_name)
        print(f'OK: {pkg} available')
    except ModuleNotFoundError:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])


def preprocess(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    # Resize (important for OCR)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Denoise
    gray = cv2.medianBlur(gray, 3)
    # Adaptive threshold works better than Otsu for uneven lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return Image.fromarray(thresh)


def configure_tesseract():
    candidates = [
        r'C:\Users\ibf\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break
    print('pytesseract version:', pytesseract.get_tesseract_version())


# Data loading and OCR extraction
def collect_samples(root_dir, image_extensions):
    root = Path(root_dir)
    classes = sorted([path.name for path in root.iterdir() if path.is_dir()])
    samples = []
    for class_name in classes:
        for image_path in (root / class_name).rglob('*'):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                samples.append((str(image_path), class_name))
    return classes, samples

######## COMMON FUNCTIONS FOR THE FILTERING ###

stemmer = SnowballStemmer("spanish")

def stem_token(token):
    return stemmer.stem(token)


SPANISH_STOPWORDS = {
    # TODO Add more stopwords
    'Jordi', 'caixa', 'caixabank', 'buenos', 'dias', 'rentabilidad', 
    'cuenta', 'cuentas', 'bizum', 'enviar bizum', 'resumen', 'mensual', 'recibo', 'recibos',
    'tarjeta', 'tarjetas', 'mostrar', 'todo', 'inicio', 'disfruta', 'operar', 'contratar', 'imaginbank', 
    'innmaginbank', 'trasnferencia', 'recibo', 'recibos', 'contatar', 'productos', 'mecardos', 'buscar',
    'favoritos', 'perfil', 'plusvalia', 'rentabilidad', 'posiciones', 'ordenes', 'visa', 'debit', 'certificado',
    'titularidad', 'solicitar', 'devolucion', 'buscar', 'movimientos', 'opciones', 'gastos', 'ingresos', 'todo',
    'historial', 'buenas', 'noches', 'Albert'
}

VOWELS = set("aeiou")

ALLOWED_WORDS = {   'no', 'si', 'es', 'en', 'el', 'la', 'lo', 'de', 'y', 'a', 'un', 'una', 'ok' }

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()

    # Normalize accents (é -> e, ñ -> n, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Remove special characters (keep letters, numbers, and spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text


def remove_numbers_and_symbols(text):
    # Remove digits and specified symbols
    text = re.sub(r"[0-9\.,\*\+\-/\\\[\]\(\)]", "", text)
    return text


def looks_like_repeated_noise(token):
    return len(token) > 4 and len(set(token)) <= 2


def vowel_ratio(token):
    vowels = sum(c in VOWELS for c in token)
    return vowels / len(token)


def looks_like_ocr_noise(token):
    if len(token) < 3 and token not in ALLOWED_WORDS:
        return True

    if looks_like_repeated_noise(token):
        return True

    # Spanish words usually contain at least some vowels
    if vowel_ratio(token) < 0.15:
        return True

    return False

def has_repeated_letters(token):
    for i in range(len(token) - 2):
        if token[i] == token[i + 1] == token[i + 2]:
            return True
    return False

def remove_duplicate_tokens(tokens):
    seen = set()
    unique_tokens = []

    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    return unique_tokens

def has_weird_consonant_cluster(token):
    return bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", token))

def normalize_repetitions(token):
    # coooool -> cool
    return re.sub(r"(.)\1{2,}", r"\1\1", token)

def is_real_word(token, min_freq=3):
    return zipf_frequency(token, 'es') > min_freq \
        or zipf_frequency(token, 'ca') > min_freq \
        or zipf_frequency(token, 'en') > min_freq

def vowel_ratio(token):
    vowels = "aeiouáéíóúàèìòù"
    v = sum(1 for c in token if c in vowels)
    return v / len(token)

def is_suspicious(token):
    ratio = vowel_ratio(token)
    return ratio < 0.2 or ratio > 0.9

def remove_numbers_and_symbols_2(text):
    # Keep letters (including accents) and spaces
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s]+", "", text)
    return text

def single_char(token):
    if len(token) == 1 and token.lower() not in "aeiou":
        return True
    return False

######## FILTER WITH NO BERT ##################

def clean_ocr_text(text, stopwords = SPANISH_STOPWORDS, min_token_length=3, min_tokens=3):
    if not text:
        return ""

    text = remove_numbers_and_symbols(text)
    text = normalize_text(text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    raw_tokens = text.split()
    cleaned_tokens = []

    for token in raw_tokens:
        token = normalize_repetitions(token)
        token = stem_token(token)
        if len(token) < min_token_length:
            continue
        # Paraula que no aporta valor
        if token in stopwords:
            continue
        # Soroll
        if looks_like_ocr_noise(token):
            continue
        # Paraules amb lletres repetides
        if has_repeated_letters(token):
            continue
        # Paraules amb clusters de consonants estranys
        if has_weird_consonant_cluster(token):
            continue    

        #  TODO ADD MORE STRATS
        cleaned_tokens.append(token)

    # unique_tokens = remove_duplicate_tokens(cleaned_tokens)

    # Reject almost-empty OCR
    if len(cleaned_tokens) < min_tokens:
        return ""
    
    return " ".join(cleaned_tokens)


############ FILTER WITH BERT ##################

def clean_for_bert(text):
    if not text:
        return ""

    text = text.lower()
    text = remove_numbers_and_symbols_2(text)
    text = re.sub(r"\s+", " ", text).strip()
    
    raw_tokens = text.split()
    cleaned_tokens = []

    for token in raw_tokens:
        token = normalize_repetitions(token)
        # Un caracter només no vocals
        if single_char(token):
            continue
        # Soroll
        if looks_like_ocr_noise(token):
            continue
        # Paraules amb lletres repetides
        if has_repeated_letters(token):
            continue
        # Paraules amb clusters de consonants estranys
        if has_weird_consonant_cluster(token):
            continue   
        # Paraules amb una proporció de vocals massa baixa o massa alta 
        if is_suspicious(token):
            continue
        # Paraula no real
        if not is_real_word(token):
            continue

        cleaned_tokens.append(token)
    
    if len(cleaned_tokens) <= 2:
            return ""

    
    valid_words = sum(is_real_word(t) for t in cleaned_tokens)

    if valid_words / len(cleaned_tokens) < 0.5:
        return ""

    return " ".join(cleaned_tokens)


#####################################################
############# OCR TEXT UTILS ########################
#####################################################

def extract_text_from_image(path, crop_top=120, crop_bottom=120):
    try:
        img = Image.open(path).convert('RGB')

        width, height = img.size
        if height > (crop_top + crop_bottom):
            top = int(crop_top)
            bottom = int(height - crop_bottom)
            if top < bottom:
                img = img.crop((0, top, width, bottom))

        # Opcional: aplicar tu preprocesado
        img = preprocess(img)

        # custom_config = r'--oem 3 --psm 6'
        custom_config = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'

        return pytesseract.image_to_string(
            img,
            lang='spa',
            config=custom_config
        ).strip()

    except Exception as exc:
        print(f'OCR failed for {path}: {exc}')
        return ''

def build_ocr_dataframe(root_dir, split_name):
    classes, samples = collect_samples(root_dir)
    records = []

    for image_path, label in tqdm(samples, desc=f'OCR {split_name}'):
        raw_text = extract_text_from_image(image_path)
        # Choose one strategy: bert or no bert
        # clean_text = clean_ocr_text(raw_text)
        clean_text_for_bert = clean_for_bert(raw_text)
        
        records.append(
            {
                'path': image_path,
                'label': label,
                'raw_text': raw_text,
                'text': clean_text_for_bert,   
            }
        )

    df = pd.DataFrame(records)

    print(f'{split_name} samples:', len(df))
    print(f'{split_name} empty OCR texts:',
          int((df['text'].fillna('').str.strip() == '').sum()))

    return classes, df


def build_ocr_dataframe_from_samples(samples, class_names, split_name):
    records = []

    for image_path, label in tqdm(samples, desc=f'OCR {split_name}'):
        raw_text = extract_text_from_image(image_path)
        clean_text_for_bert = clean_for_bert(raw_text)

        records.append({
            'path': image_path,
            'label': label,
            'raw_text': raw_text,
            'text': clean_text_for_bert,
        })

    df = pd.DataFrame(records)

    print(f'{split_name} samples:', len(df))
    print(f'{split_name} empty OCR texts:',
          int((df['text'].fillna('').str.strip() == '').sum()))

    return df