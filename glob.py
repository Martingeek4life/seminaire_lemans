import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.linalg import svd

def load_embeddings(file_path):
    words = []
    vectors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # ignore la première ligne
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    return words, np.array(vectors)

def compute_pca_axes(vectors):
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    pca = PCA(n_components=3)
    pca.fit(centered)
    # axes échelle par racine de la variance (taille représentative)
    axes = pca.components_ * np.sqrt(pca.explained_variance_).reshape(3,1)
    return mean, axes, pca.explained_variance_

def plot_ellipsoid(ax, mean, axes, color='blue', alpha=0.1):
    # Générer un ellipsoïde approximatif pour montrer la forme de l'espace
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = axes[0,0] * np.outer(np.cos(u), np.sin(v)) + mean[0]
    y = axes[1,1] * np.outer(np.sin(u), np.sin(v)) + mean[1]
    z = axes[2,2] * np.outer(np.ones_like(u), np.cos(v)) + mean[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_spaces(vectors_before, vectors_after, words):
    fig = plt.figure(figsize=(14,12))
    ax = fig.add_subplot(111, projection='3d')

    # Calcul des axes PCA
    mean_before, axes_before, var_before = compute_pca_axes(vectors_before)
    mean_after, axes_after, var_after = compute_pca_axes(vectors_after)

    # Nuages de points
    ax.scatter(vectors_before[:,0], vectors_before[:,1], vectors_before[:,2], c='blue', alpha=0.2, label='Nuage avant')
    ax.scatter(vectors_after[:,0], vectors_after[:,1], vectors_after[:,2], c='red', alpha=0.2, label='Nuage après')

    # Ellipsoïdes pour forme géométrique
    plot_ellipsoid(ax, mean_before, axes_before, color='blue', alpha=0.1)
    plot_ellipsoid(ax, mean_after, axes_after, color='red', alpha=0.1)

    # Axes principaux
    for i in range(3):
        ax.quiver(mean_before[0], mean_before[1], mean_before[2],
                  axes_before[i,0], axes_before[i,1], axes_before[i,2],
                  color='blue', linewidth=3, arrow_length_ratio=0.1, label=f'Axe avant {i+1}' if i==0 else "")
        ax.quiver(mean_after[0], mean_after[1], mean_after[2],
                  axes_after[i,0], axes_after[i,1], axes_after[i,2],
                  color='red', linewidth=3, arrow_length_ratio=0.1, label=f'Axe après {i+1}' if i==0 else "")

    # Flèches entre axes correspondants (facultatif)
    for i in range(3):
        ax.plot([mean_before[0]+axes_before[i,0], mean_after[0]+axes_after[i,0]],
                [mean_before[1]+axes_before[i,1], mean_after[1]+axes_after[i,1]],
                [mean_before[2]+axes_before[i,2], mean_after[2]+axes_after[i,2]],
                color='purple', linestyle='--', alpha=0.5)

    # Mots annotés (optionnel, peut être brouillant si trop de mots)
    for i in range(len(words)):
        ax.text(vectors_before[i,0], vectors_before[i,1], vectors_before[i,2], words[i], color='blue', fontsize=8)
        ax.text(vectors_after[i,0], vectors_after[i,1], vectors_after[i,2], words[i], color='red', fontsize=8)

    ax.set_title("Transformation de l'espace vectoriel : forme, axes et nuage de mots")
    ax.legend()
    plt.savefig("espace_geom_avant_apres.png")
    plt.show()

if __name__ == "__main__":
    file_before = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec"  # ton fic>    
    file_after = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec"
    words_before, vectors_before = load_embeddings(file_before)
    words_after, vectors_after = load_embeddings(file_after)

    assert words_before == words_after, "Les deux fichiers doivent avoir les mêmes mots dans le même ordre."

    # Réduction à 3D pour visualisation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    vectors_before_3d = pca.fit_transform(vectors_before)
    vectors_after_3d = pca.transform(vectors_after)

    plot_spaces(vectors_before_3d, vectors_after_3d, words_before)
