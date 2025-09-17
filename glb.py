import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

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
    axes = pca.components_ * np.sqrt(pca.explained_variance_).reshape(3,1)
    return mean, axes

def plot_ellipsoid_cov(ax, mean, cov, color='blue', alpha=0.1):
    # Eigen decomposition pour obtenir les directions et échelles exactes
    vals, vecs = eigh(cov)
    # vals sont les valeurs propres, vecs les vecteurs propres
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    # Transformation de l'ellipsoïde unité selon covariance
    ellipsoid = np.array([x.flatten(), y.flatten(), z.flatten()])
    ellipsoid_transformed = vecs @ np.diag(np.sqrt(vals)) @ ellipsoid
    x_t = ellipsoid_transformed[0,:].reshape(x.shape) + mean[0]
    y_t = ellipsoid_transformed[1,:].reshape(y.shape) + mean[1]
    z_t = ellipsoid_transformed[2,:].reshape(z.shape) + mean[2]
    ax.plot_surface(x_t, y_t, z_t, color=color, alpha=alpha)

def plot_space(vectors, words, color='blue', title='Espace', filename='space.png'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')

    mean = np.mean(vectors, axis=0)
    cov = np.cov(vectors, rowvar=False)

    # Nuage de points
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,2], c=color, alpha=0.2, label='Nuage de mots')

    # Vecteurs directeurs principaux (PCA)
    mean_pca, axes_pca = compute_pca_axes(vectors)
    for i in range(3):
        ax.quiver(mean_pca[0], mean_pca[1], mean_pca[2],
                  axes_pca[i,0], axes_pca[i,1], axes_pca[i,2],
                  color=color, linewidth=4, arrow_length_ratio=0.15)

    # Ellipsoïde exact basé sur covariance
    plot_ellipsoid_cov(ax, mean, cov, color=color, alpha=0.15)

    # Limites égales
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

    # Réduction à 3D pour visualisation si nécessaire
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    vectors_before_3d = pca.fit_transform(vectors_before)
    vectors_after_3d = pca.transform(vectors_after)

    # Figure pour espace avant avec vecteurs directeurs en bleu
    plot_space(vectors_before_3d, words_before, color='blue', title='Espace avant', filename='espace_avant_cov.png')

    # Figure pour espace après avec vecteurs directeurs en rouge
    plot_space(vectors_after_3d, words_after, color='red', title='Espace après', filename='espace_apres_cov.png')
