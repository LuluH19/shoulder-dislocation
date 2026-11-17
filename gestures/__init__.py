"""Reconnaissance de gestes (circle / horizontal / vertical) à partir d'un capteur IMU 6 axes.

Modules, dans l'ordre où les données circulent :
    config         chemins, gestes, format des enregistrements
    data           lecture des CSV de tous les groupes
    preprocessing  normalisation, découpage, augmentation
    model          le CNN 1D
    training       entraînement, prédiction, sauvegarde
    metrics        matrice de confusion, précision, rappel, F1
    analysis       mesures sur les signaux (tours, sens, corrélations)
    plots          toutes les figures
    quantization   versions float16 / int8 du modèle
    export         génération de model_weights.h pour l'ESP32
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # évite un blocage de torch sur macOS
