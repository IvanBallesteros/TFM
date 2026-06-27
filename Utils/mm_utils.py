import os
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from TFM.Utils.ocr_utils import predict_text_proba

IND_CLASS = "Uncertain"

# thresholds
CV_MIN_CONF = 0.85
TEXT_MIN_CONF = 0.95
COMB_THRESHOLD = 0.80

TEXT_STRONG = 0.97
CV_STRONG = 0.97

# weights
W_CV = 0.4
W_TEXT = 0.6

# ---------- HELPERS ----------
def get_true_labels_and_paths(test_dir):
    image_paths = []
    labels = []

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            image_paths.append(img_path)
            labels.append(class_name)

    return image_paths, labels


def dict_from_probs(prob_list):
    """Convert list of dicts into class->confidence dict"""
    return {item['class']: item['confidence'] for item in prob_list}


def combined_decision_3(text_pred, text_conf, text_probs,
                      cv_pred, cv_conf, cv_probs):

    # RULE 1: TEXT says Empty → trust CV
    if text_pred == "Empty":
        if cv_conf < CV_MIN_CONF:
            return IND_CLASS
        return cv_pred
    
    if text_conf >= TEXT_STRONG:
        return text_pred

    if cv_conf >= CV_STRONG:
        return cv_pred

    # RULE 2: CV confidence too low
    if cv_conf < CV_MIN_CONF:
        if text_conf > TEXT_MIN_CONF:
            return text_pred
    
    if text_pred == cv_pred:
        return text_pred

    if abs(text_conf - cv_conf) > 0.25:
        return text_pred if text_conf > cv_conf else cv_pred

    # RULE 3: Weighted fusion
    all_classes = set(text_probs.keys()).union(set(cv_probs.keys()))

    # Dynamic weights based on each model's confidence
    w_cv = cv_conf
    w_text = text_conf

    total = w_cv + w_text

    combined_scores = {}
    for c in all_classes:
        combined_scores[c] = (
            w_cv * cv_probs.get(c, 0.0) +
            w_text * text_probs.get(c, 0.0)
        ) / total

    best_class = max(combined_scores, key=combined_scores.get)
    scores = sorted(combined_scores.values(), reverse=True)
    margin = scores[0] - scores[1]

    if margin < 0.15:
        return IND_CLASS

    return best_class
