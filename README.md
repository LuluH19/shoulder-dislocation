# Reconnaissance de gestes avec un capteur IMU

Un capteur de mouvement (accéléromètre + gyroscope, 6 axes) attaché au bras, un petit
réseau de neurones (CNN 1D, 9 715 paramètres) et une carte ESP32 qui reconnaît en temps
réel trois gestes : **circle**, **horizontal** et **vertical**.

```
geste → capteur MPU6050 → 75 mesures (1,5 s à 50 Hz) → normalisation → CNN 1D → geste prédit
```

L'ordinateur sert à **apprendre** (Python), la carte sert à **reconnaître** (C).
On ne transfère vers la carte que les poids du modèle.

## Organisation du dépôt

```
data/                     les enregistrements CSV, un dossier par groupe (voir data/README.md)
gestures/                 le code Python, un fichier par étape
├── config.py             chemins, gestes, format des enregistrements
├── data.py               lecture des CSV de tous les groupes
├── preprocessing.py      normalisation, découpage, augmentation
├── model.py              le CNN 1D
├── training.py           entraînement, prédiction, sauvegarde
├── metrics.py            matrice de confusion, précision, rappel, F1
├── analysis.py           mesures sur les signaux (tours, sens, corrélations)
├── plots.py              toutes les figures
├── quantization.py       versions float16 / int8 du modèle
└── export.py             génération de model_weights.h pour la carte
scripts/                  les commandes à lancer (une par tâche, voir plus bas)
firmware/movement_classifier/
├── movement_classifier.ino   le sketch : capteur, détection du geste, affichage
├── cnn.h                     le réseau réécrit en C, sans bibliothèque
└── model_weights.h           les poids, générés par scripts/export_firmware.py
models/                   modèles entraînés (movements.pt), non versionnés
results/                  figures et métriques produites par les scripts
tests/                    tests automatiques (pytest)
docs/                     site de documentation (Astro Starlight)
```

Le code va dans un seul sens : `scripts/` utilise `gestures/`, jamais l'inverse.
Chaque script commence par une explication de ce qu'il fait et de ce qu'il écrit.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # installe aussi le paquet gestures/ en mode éditable
```

## Utilisation

Toutes les commandes se lancent depuis la racine du dépôt.

### Du jeu de données à la carte

| Étape | Commande | Produit |
|---|---|---|
| 1. Entraîner | `python scripts/train.py` | `models/movements.pt`, `results/training/` |
| 2. Exporter | `python scripts/export_firmware.py` | `firmware/movement_classifier/model_weights.h` |
| 3. Téléverser | ouvrir `firmware/movement_classifier/movement_classifier.ino` dans l'IDE Arduino, puis Téléverser | |

Sur la carte, le moniteur série (115200 bauds) affiche le geste reconnu. Câblage du
MPU6050 : SDA sur la broche 8, SCL sur la broche 9.

Options utiles de `train.py` : `--holdout-group group3` (teste sur un groupe jamais vu),
`--epochs 80`, `--test-data autre/dossier`. Toutes les options : `python scripts/train.py --help`.

### Expériences

| Question | Commande | Résultats |
|---|---|---|
| Quel geste dans ce fichier ? | `python scripts/predict.py data/group3/circle/3_Circle_1.csv` | terminal |
| Le modèle marche-t-il sur un groupe jamais vu ? | `python scripts/compare_groups.py` | `results/groups/` |
| Pourquoi les groupes diffèrent-ils ? | `python scripts/analyze_groups.py` | `results/groups/` |
| Faut-il les 6 axes ? | `python scripts/compare_axes.py` | `results/axes/` |
| Peut-on compresser le modèle ? | `python scripts/quantize.py` | `results/quantization/` |

### Tests

```bash
pytest
```

`tests/test_firmware.py` compile `cnn.h` avec le compilateur C de l'ordinateur et vérifie
qu'il donne les mêmes probabilités que PyTorch.

## Points à garder en tête

- **La normalisation doit être la même partout.** `gestures/preprocessing.py::normalize`
  et `normalize_window` dans `cnn.h` font le même z-score par axe et par fenêtre.
- **Après chaque entraînement**, relancer `export_firmware.py` puis téléverser :
  sinon la carte garde les anciens poids.
- **La carte attend le modèle 16 / 32 / 64 filtres sur 6 axes.** `export_firmware.py`
  refuse un modèle d'une autre taille.
