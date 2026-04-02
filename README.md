# TP04 — Reconnaissance Faciale par PCA (Eigenfaces) et Viola-Jones

Travaux pratiques de biométrie et tatouage numérique.  
Ce projet implémente un système de reconnaissance faciale basé sur l'Analyse en Composantes Principales (PCA / Eigenfaces) combinée à la détection Viola-Jones.

## Objectifs

- Détection de visage par la méthode de Viola-Jones (Cascades de Haar)
- Construction d'un modèle de reconnaissance par ACP (PCA – Eigenfaces)
- Projection des visages dans un sous-espace de dimension réduite
- Comparaison des vecteurs par la distance Euclidienne
- Prise de décision par seuillage de distance

## Prérequis

- Python 3.x
- Installer les dépendances :

```
pip install opencv-python numpy scipy matplotlib scikit-learn
```

## Dataset utilisé

Base AT&T (ORL) — 40 personnes × 10 images = 400 images au total.  
Téléchargeable sur : https://www.kaggle.com/datasets/kasikrit/att-database-of-faces

> Le dataset n'est pas inclus dans ce dépôt (trop volumineux).  
> Télécharge-le et place-le dans un dossier `dataset/` à la racine du projet.

## Structure du projet

```
tp4/
├── tp4_pca.py          # Script principal
├── test.jpg            # Image de test
├── README.md
├── .gitignore
└── dataset/            # Non inclus — à télécharger sur Kaggle
    ├── s1/
    ├── s2/
    └── ... (s1 à s40)
```

## Lancer le projet

```
python tp4_pca.py
```

## Architecture du code

### Classe `FaceRecognitionPCA`

| Méthode | Rôle |
|---------|------|
| `__init__(n_components)` | Initialisation Viola-Jones + paramètres PCA |
| `detect_face(image)` | Détection + recadrage visage (100×100 px) |
| `load_dataset(path)` | Chargement des images et vectorisation |
| `compute_pca(X)` | Calcul moyenne, centrage, covariance, vecteurs propres |
| `project(face_vector)` | Projection dans l'espace PCA |
| `fit(dataset_path)` | Pipeline complet d'entraînement |
| `recognize(image_path, threshold)` | Reconnaissance + décision |

## Pipeline de traitement

1. Chargement du dataset structuré par personne
2. Détection du visage (Viola-Jones) + redimensionnement 100×100
3. Vectorisation (100×100 = 10 000 dimensions)
4. Calcul PCA :
   - Moyenne des visages
   - Centrage de la matrice
   - Décomposition en valeurs propres
   - Sélection des n_components premiers vecteurs propres
5. Projection de tous les visages dans l'espace PCA
6. Pour une image test :
   - Détection + projection
   - Distance Euclidienne avec chaque vecteur de la base
   - Décision par seuillage

## Seuil de décision

| Distance | Décision |
|----------|----------|
| ≤ 3000 | MATCH |
| > 3000 | NO MATCH |

> Le seuil est expérimental et peut être ajusté dans `main()`.

## Expérimentations

### Effet du nombre de composantes k

| k | Distance | Décision |
|---|----------|----------|
| 10 | — | — |
| 20 | — | — |
| 50 | — | — |

> Remplis ce tableau après avoir lancé `experimenter_k()`.

### Réponses aux questions d'analyse

**Pourquoi PCA nécessite un bon alignement des visages ?**  
Les eigenfaces sont calculées pixel par pixel. Un mauvais alignement fait capturer des décalages géométriques plutôt que les traits du visage.

**Que se passe-t-il si k est trop faible ?**  
Perte d'information importante → les visages différents se ressemblent dans l'espace réduit → fausses acceptations.

**Que se passe-t-il si k est trop élevé ?**  
Capture du bruit et de variations inutiles → distances plus grandes → faux rejets.

**Pourquoi la distance Euclidienne est adaptée dans l'espace PCA ?**  
Les composantes principales sont orthogonales et décorrélées, la distance Euclidienne y est géométriquement significative.

**Limites face aux variations d'illumination ?**  
Les premières eigenfaces capturent surtout les variations d'éclairage (grande variance) plutôt que l'identité.

## Résultats générés dans `results/`

| Fichier | Contenu |
|---------|---------|
| `tp4_resultat.png` | Image test + rectangle + identité + décision |
| `tp4_effet_k.png` | Graphique distance vs k (10, 20, 50) |

## Technologies utilisées

- [OpenCV](https://opencv.org/) — détection Viola-Jones
- [NumPy](https://numpy.org/) — calcul PCA
- [SciPy](https://scipy.org/) — distance Euclidienne
- [Matplotlib](https://matplotlib.org/) — visualisation
