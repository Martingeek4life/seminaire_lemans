import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def load_embeddings(file_path):
    words = []
    vectors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
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
    axes = pca.components_ * np.sqrt(pca.explained_variance_).reshape(3,1)
    return mean, axes

def plot_space(vectors, words, color='blue', title='Espace', filename='space.png'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')

    mean, axes = compute_pca_axes(vectors)

    # Nuage de points
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,2], c=color, alpha=0.3)

    # Ellipsoïde simple pour la forme globale
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = axes[0,0]*np.outer(np.cos(u), np.sin(v)) + mean[0]
    y = axes[1,1]*np.outer(np.sin(u), np.sin(v)) + mean[1]
    z = axes[2,2]*np.outer(np.ones_like(u), np.cos(v)) + mean[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.1)

    # Axes principaux
    for i in range(3):
        ax.quiver(mean[0], mean[1], mean[2],
                  axes[i,0], axes[i,1], axes[i,2],
                  color=color, linewidth=3, arrow_length_ratio=0.1)

    # Limites égales pour voir la forme correctement
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
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    file_before = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec"
    file_after = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec"

    words_before, vectors_before = load_embeddings(file_before)
    words_after, vectors_after = load_embeddings(file_after)

    assert words_before == words_after, "Les deux fichiers doivent avoir les mêmes mots dans le même ordre."

    # Réduction à 3D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    vectors_before_3d = pca.fit_transform(vectors_before)
    vectors_after_3d = pca.transform(vectors_after)

    # Figure séparée pour chaque espace
    plot_space(vectors_before_3d, words_before, color='blue', title='Espace avant', filename='espace_avant.png')
    plot_space(vectors_after_3d, words_after, color='red', title='Espace après', filename='espace_apres.png')
