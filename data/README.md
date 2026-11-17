# Données

Un enregistrement = un geste = un fichier CSV de **75 lignes** (1,5 s à 50 Hz) et
**6 colonnes** : accéléromètre X/Y/Z puis gyroscope X/Y/Z.

| Groupe | Enregistrements | Dossiers | En-têtes | Déjà normalisé ? |
|---|---|---|---|---|
| group1 | 600 | `circle`, `hor`, `vert` | `aX,aY,aZ,gX,gY,gZ` | non |
| group2 | 600 | `circle`, `horizontal`, `vertical` | `aX,aY,aZ,gX,gY,gZ` | non |
| group3 | 300 | `circle`, `horizontal`, `vertical` | `acc_x,…,gyro_z` | oui, z-score |
| group5 | 600 | `circle`, `hor`, `ver` | `X,Y,Z,GX,GY,GZ` | oui, min-max (`group5/normalisation.json`) |

Les fichiers sont gardés tels que chaque groupe les a fournis. `gestures/data.py` unifie
les noms au chargement :

- **le geste** vient du nom du fichier, sinon du dossier (`hor` → horizontal, `ver`/`vert` → vertical) ;
- **le groupe** vient du dossier `groupN` ;
- **les colonnes** sont remises dans l'ordre `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`.

Pour ajouter un groupe : créer `data/groupN/` avec un sous-dossier par geste. Les scripts
le prennent en compte automatiquement.
