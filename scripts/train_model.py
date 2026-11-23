"""Entrena el clasificador de categorías de tickets y lo guarda en disco.

Instalar:  pip install pandas scikit-learn joblib
Uso:       python train_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/tickets.csv"
MODEL_PATH = os.path.join("models", "ticket_classifier.joblib")


def build_pipeline() -> Pipeline:
    return Pipeline([
        # strip_accents: "contraseña" y "contrasena" cuentan como la misma palabra
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def main():
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset: {len(data)} tickets")
    print(data["category"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["category"],
        test_size=0.2, stratify=data["category"], random_state=42,
    )

    # 1) Evaluación: entrenar con 80 % y medir con el 20 % que el modelo no vio
    model = build_pipeline()
    model.fit(X_train, y_train)
    print("=== Evaluación en el 20 % de prueba ===")
    print(classification_report(y_test, model.predict(X_test)))

    # 2) Validación cruzada: con un test de ~36 ejemplos, un solo split es poco fiable
    scores = cross_val_score(build_pipeline(), data["message"], data["category"], cv=5)
    print(f"Validación cruzada (5 folds): {scores.mean():.2f} ± {scores.std():.2f}\n")

    # 3) Modelo final: reentrenar con todos los datos y guardarlo
    final_model = build_pipeline()
    final_model.fit(data["message"], data["category"])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    # 4) Prueba rápida con confianza
    texto = "Alguien entró a mi cuenta sin autorización"
    probs = final_model.predict_proba([texto])[0]
    best = probs.argmax()
    print(f"Prueba: '{texto}' -> {final_model.classes_[best]} ({probs[best]:.2f})")


if __name__ == "__main__":
    main()