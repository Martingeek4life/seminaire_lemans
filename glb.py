import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

# =======================
# 1. Charger les embeddings
# =======================
def load_embeddings(file_path):
    words = []
    vectors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Ignore la première ligne (header)
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    return words, np.array(vectors)

# =======================
# 2. Calculer les axes principaux (vecteurs directeurs) avec PCA
# =======================
def compute_pca_axes(vectors):
    mean = np.mean(vectors, axis=0)  # barycentre du nuage
    centered = vectors - mean
    pca = PCA(n_components=3)
    pca.fit(centered)
    # axes échelle par racine de la variance pour bien les visualiser
    axes = pca.components_ * np.sqrt(pca.explained_variance_).reshape(3,1)
    return mean, axes

# =======================
# 3. Dessiner un ellipsoïde basé sur la covariance pour montrer la forme réelle
# =======================
def plot_ellipsoid_cov(ax, mean, cov, color='blue', alpha=0.2):
    # Eigen decomposition pour obtenir axes et tailles exactes
    vals, vecs = eigh(cov)  # vals = valeurs propres, vecs = vecteurs propres
    # Génération d'une sphère unité
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.array([x.flatten(), y.flatten(), z.flatten()])
    # Transformation selon covariance
    ellipsoid = vecs @ np.diag(np.sqrt(vals)) @ sphere
    x_t = ellipsoid[0,:].reshape(x.shape) + mean[0]
    y_t = ellipsoid[1,:].reshape(y.shape) + mean[1]
    z_t = ellipsoid[2,:].reshape(z.shape) + mean[2]
    # Affichage de l'ellipsoïde
    ax.plot_surface(x_t, y_t, z_t, color=color, alpha=alpha)

# =======================
# 4. Fonction pour afficher l'espace avec points, vecteurs directeurs et ellipsoïde
# =======================
def plot_space(vectors, words, color='blue', title='Espace', filename='space.png'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')

    # Calcul du barycentre et covariance
    mean = np.mean(vectors, axis=0)
    cov = np.cov(vectors, rowvar=False)

    # Nuage de points
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,2], c=color, alpha=0.3, label='Nuage de mots')

    # Vecteurs directeurs via PCA
    mean_pca, axes_pca = compute_pca_axes(vectors)
    for i in range(3):
        ax.quiver(mean_pca[0], mean_pca[1], mean_pca[2],
                  axes_pca[i,0], axes_pca[i,1], axes_pca[i,2],
                  color=color, linewidth=4, arrow_length_ratio=0.15, label=f'Axe principal {i+1}' if i==0 else "")

    # Ellipsoïde exact basé sur covariance
    plot_ellipsoid_cov(ax, mean, cov, color=color, alpha=0.15)

    # Limites égales pour visualiser correctement la forme
    max_range = np.array([vectors[:,0].max()-vectors[:,0].min(),
                          vectors[:,1].max()-vectors[:,1].min(),
                          vectors[:,2].max()-vectors[:,2].min()]).max() / 2.0
    mid_x = (vectors[:,0].max()+vectors[:,0].min()) * 0.5
    mid_y = (vectors[:,1].max()+vectors[:,1].min()) * 0.5
    mid_z = (vectors[:,2].max()+vectors[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(title)
    ax.legend()
    plt.savefig(filename)
    plt.show()

# =======================
# 5. Partie principale
# =======================
if __name__ == "__main__":
    # Fichiers embeddings avant/après
    file_before = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec"
    file_after = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec"
    # Chargement
    words_before, vectors_before = load_embeddings(file_before)
    words_after, vectors_after = load_embeddings(file_after)

    # Vérification que les mots sont les mêmes
    assert words_before == words_after, "Les deux fichiers doivent avoir les mêmes mots dans le même ordre."

    # Réduction à 3D pour visualisation
    pca = PCA(n_components=3)
    vectors_before_3d = pca.fit_transform(vectors_before)
    vectors_after_3d = pca.transform(vectors_after)

    # Figure séparée pour l'espace avant (vecteurs directeurs en bleu)
    plot_space(vectors_before_3d, words_before, color='blue', title='Espace avant', filename='espace_avant_final.png')

    # Figure séparée pour l'espace après (vecteurs directeurs en rouge)
    plot_space(vectors_after_3d, words_after, color='red', title='Espace après', filename='espace_apres_final.png')
