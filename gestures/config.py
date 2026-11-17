"""Constantes partagées par tout le projet : chemins, gestes, format des enregistrements."""
from pathlib import Path

# ---- Chemins (tous relatifs à la racine du dépôt) ----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"                                # CSV bruts, un dossier par groupe
MODELS_DIR = ROOT / "models"                            # modèles entraînés (.pt)
RESULTS_DIR = ROOT / "results"                          # figures et métriques
FIRMWARE_DIR = ROOT / "firmware" / "movement_classifier"  # sketch Arduino de l'ESP32

DEFAULT_MODEL = MODELS_DIR / "movements.pt"
DEFAULT_HEADER = FIRMWARE_DIR / "model_weights.h"

# ---- Gestes à reconnaître ----
CLASSES = ["circle", "horizontal", "vertical"]
# Mot-clé cherché dans le nom du fichier puis du dossier -> geste.
# "ver" couvre ver / vert / vertical, "hor" couvre hor / horizontal.
CLASS_KEYWORDS = (("circ", "circle"), ("hor", "horizontal"), ("ver", "vertical"))

# ---- Format d'un enregistrement ----
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
# Noms de colonnes utilisés par les différents groupes (en minuscules), dans l'ordre de CHANNELS.
COLUMN_ALIASES = [
    ["ax", "acc_x", "x", "accx"],
    ["ay", "acc_y", "y", "accy"],
    ["az", "acc_z", "z", "accz"],
    ["gx", "gyro_x", "gyrox"],
    ["gy", "gyro_y", "gyroy"],
    ["gz", "gyro_z", "gyroz"],
]
SAMPLE_RATE_HZ = 50
N_SAMPLES = 75  # 1,5 s à 50 Hz

# ---- Modèle ----
# Nombre de filtres des 3 convolutions. firmware/movement_classifier/cnn.h attend (16, 32, 64).
MODEL_WIDTHS = (16, 32, 64)
