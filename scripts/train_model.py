"""Entrena el clasificador de categorías y lo guarda en models/ticket_classifier.joblib.

Uso (desde la raíz del proyecto):
    python scripts/train_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml import MODEL_PATH

DATA_PATH = "data/tickets.csv"


def build_pipeline() -> Pipeline:
    return Pipeline([
        # strip_accents: "contraseña" y "contrasena" cuentan como la misma palabra
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def main():
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset: {len(data)} tickets")
    print(data["category"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["category"],
        test_size=0.2,
        stratify=data["category"],
        random_state=42,
    )

    # 1) Evaluación honesta: entrenar con 80 % y medir con el 20 % que el modelo no vio
    model = build_pipeline()
    model.fit(X_train, y_train)
    print("=== Evaluación en el 20 % de prueba ===")
    print(classification_report(y_test, model.predict(X_test)))

    # 2) Modelo final: se reentrena con todos los datos para aprovechar cada ejemplo
    final_model = build_pipeline()
    final_model.fit(data["message"], data["category"])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    print("Prueba rápida:", final_model.predict(["Alguien entró a mi cuenta sin autorización"])[0])


if __name__ == "__main__":
    main()