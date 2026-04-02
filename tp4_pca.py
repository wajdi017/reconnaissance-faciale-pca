import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean
import os

os.makedirs("results", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

class FaceRecognitionPCA:

    def __init__(self, n_components=30):
        """
        Initialise :
        - détecteur Viola-Jones
        - nombre de composantes principales
        - variables internes
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector     = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise IOError("Fichier cascade Haar introuvable.")

        self.n_components  = n_components
        self.mean          = None
        self.eigenvectors  = None
        self.projections   = None
        self.labels        = []
        print(f"Système PCA initialisé — k={n_components} composantes.")

    # ──────────────────────────────────────────────────────────────────────────
    def detect_face(self, image: np.ndarray):
        """
        Détecte le plus grand visage dans une image BGR.
        Retourne le visage en niveaux de gris redimensionné à 100×100.
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (30, 30)
        )

        if len(faces) == 0:
            return None, None

        # Garder le plus grand visage
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face       = gray[y:y+h, x:x+w]
        face       = cv2.resize(face, (100, 100))
        return face, (x, y, w, h)

    # ──────────────────────────────────────────────────────────────────────────
    def load_dataset(self, dataset_path: str):
        """
        Parcourt dataset/ structuré par personne.
        Retourne X (matrice des visages vectorisés) et y (labels).
        """
        X, y = [], []

        for person in sorted(os.listdir(dataset_path)):
            person_dir = os.path.join(dataset_path, person)
            if not os.path.isdir(person_dir):
                continue

            count = 0
            for fname in os.listdir(person_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")):
                    continue

                img_path = os.path.join(person_dir, fname)
                image    = cv2.imread(img_path)
                if image is None:
                    continue

                face, _ = self.detect_face(image)

                # Si pas de visage détecté → essayer directement en gris
                if face is None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(gray, (100, 100))

                X.append(face.flatten().astype(float))
                y.append(person)
                count += 1

            if count > 0:
                print(f"  {person} : {count} images chargées")

        X = np.array(X)
        y = np.array(y)
        print(f"\nDataset : {X.shape[0]} images, {len(set(y))} personnes.")
        return X, y

    # ──────────────────────────────────────────────────────────────────────────
    def compute_pca(self, X: np.ndarray):
        """
        Calcule la PCA sur la matrice X :
        1. Moyenne
        2. Centrage
        3. Matrice de covariance
        4. Valeurs/vecteurs propres
        5. Sélection des n_components premiers
        """
        # 1. Moyenne
        self.mean = np.mean(X, axis=0)

        # 2. Centrage
        X_centered = X - self.mean

        # 3. Covariance (transposé pour efficacité si n_pixels >> n_images)
        n = X_centered.shape[0]
        cov = np.dot(X_centered, X_centered.T) / n

        # 4. Valeurs propres et vecteurs propres
        eigenvalues, eigenvectors_small = np.linalg.eigh(cov)

        # Passage dans l'espace original
        eigenvectors = np.dot(X_centered.T, eigenvectors_small)

        # Normalisation
        for i in range(eigenvectors.shape[1]):
            norm = np.linalg.norm(eigenvectors[:, i])
            if norm > 0:
                eigenvectors[:, i] /= norm

        # 5. Tri décroissant + sélection
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, idx[:self.n_components]]

        print(f"PCA calculée — {self.n_components} composantes retenues.")

    # ──────────────────────────────────────────────────────────────────────────
    def project(self, face_vector: np.ndarray) -> np.ndarray:
        """
        Projette un vecteur visage dans l'espace PCA.
        """
        centered = face_vector - self.mean
        return np.dot(centered, self.eigenvectors)

    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, dataset_path: str):
        """
        Charge le dataset, calcule la PCA et projette toutes les images.
        """
        X, y             = self.load_dataset(dataset_path)
        self.labels      = y
        self.compute_pca(X)
        self.projections = np.array([self.project(x) for x in X])
        print("Modèle entraîné et projections calculées.")

    # ──────────────────────────────────────────────────────────────────────────
    def recognize(self, image_path: str, threshold: float = 3000.0):
        """
        Reconnaît le visage dans image_path.
        Retourne : identité, distance minimale, décision, (image, coords)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        face, coords = self.detect_face(image)
        if face is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (100, 100))
            coords = None

        vec        = face.flatten().astype(float)
        projection = self.project(vec)

        # Distance Euclidienne avec chaque projection du dataset
        distances  = [euclidean(projection, p) for p in self.projections]
        min_idx    = int(np.argmin(distances))
        min_dist   = distances[min_idx]
        identity   = self.labels[min_idx]

        decision   = "MATCH" if min_dist <= threshold else "NO MATCH"

        return identity, min_dist, decision, (image, coords)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPÉRIMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def experimenter_k(dataset_path, test_path, k_values=[10, 20, 50], threshold=3000):
    """
    Compare les résultats pour différentes valeurs de k.
    """
    print("\n" + "═"*50)
    print("EXPÉRIMENTATION — Effet du nombre de composantes k")
    print("═"*50)

    resultats = []
    for k in k_values:
        sys = FaceRecognitionPCA(n_components=k)
        sys.fit(dataset_path)
        identity, dist, decision, _ = sys.recognize(test_path, threshold)
        resultats.append((k, identity, dist, decision))
        print(f"k={k:3d} | Identité: {identity:12s} | Distance: {dist:8.2f} | {decision}")

    # Graphique distances vs k
    ks    = [r[0] for r in resultats]
    dists = [r[2] for r in resultats]

    plt.figure(figsize=(7, 4))
    plt.plot(ks, dists, marker="o", color="#1D9E75", linewidth=2)
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Seuil = {threshold}")
    plt.xlabel("Nombre de composantes k")
    plt.ylabel("Distance minimale")
    plt.title("Effet de k sur la distance de reconnaissance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/tp4_effet_k.png")
    plt.close()
    print("\nGraphique sauvegardé : results/tp4_effet_k.png")

    return resultats


def experimenter_seuil(dataset_path, test_path, n_components=30):
    """
    Trace un tableau Distance vs Décision pour différents seuils.
    """
    print("\n" + "═"*50)
    print("EXPÉRIMENTATION — Effet du seuil")
    print("═"*50)

    sys = FaceRecognitionPCA(n_components=n_components)
    sys.fit(dataset_path)
    _, dist, _, _ = sys.recognize(test_path)

    seuils    = [500, 1000, 2000, 3000, 5000, 8000]
    decisions = ["MATCH" if dist <= s else "NO MATCH" for s in seuils]

    print(f"\nDistance obtenue : {dist:.2f}")
    print(f"{'Seuil':>8} | {'Décision':>10}")
    print("-" * 22)
    for s, d in zip(seuils, decisions):
        print(f"{s:>8} | {d:>10}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def afficher_resultat(image, coords, identity, distance, decision):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)

    if coords:
        x, y, w, h  = coords
        couleur_rect = "green" if decision == "MATCH" else "red"
        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=2,
                                  edgecolor=couleur_rect,
                                  facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 10,
                f"{identity} | dist={distance:.0f} | {decision}",
                color=couleur_rect,
                fontsize=10,
                fontweight="bold")

    ax.axis("off")
    plt.title(f"TP04 — PCA Eigenfaces\n{decision} — {identity} (dist={distance:.2f})",
              fontsize=11)
    plt.tight_layout()
    plt.savefig("results/tp4_resultat.png")
    plt.close()
    print("Image sauvegardée : results/tp4_resultat.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    DATASET_PATH  = "dataset"
    TEST_PATH     = "test.jpg"
    N_COMPONENTS  = 30
    THRESHOLD     = 3000.0

    # ── Entraînement ──────────────────────────────────────────────────────────
    system = FaceRecognitionPCA(n_components=N_COMPONENTS)
    system.fit(DATASET_PATH)

    # ── Reconnaissance ────────────────────────────────────────────────────────
    identity, distance, decision, (image, coords) = system.recognize(
        TEST_PATH, threshold=THRESHOLD
    )

    # ── Résultats console ─────────────────────────────────────────────────────
    print("\n" + "─"*40)
    print(f"Identité prédite  : {identity}")
    print(f"Distance minimale : {distance:.4f}")
    print(f"Seuil             : {THRESHOLD}")
    print(f"Décision          : {decision}")
    print("─"*40)

    # ── Affichage visuel ──────────────────────────────────────────────────────
    afficher_resultat(image, coords, identity, distance, decision)

    # ── Expérimentations ──────────────────────────────────────────────────────
    experimenter_k(DATASET_PATH, TEST_PATH, k_values=[10, 20, 50], threshold=THRESHOLD)
    experimenter_seuil(DATASET_PATH, TEST_PATH, n_components=N_COMPONENTS)


if __name__ == "__main__":
    main()