import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns

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

def plot_2d_space(vectors1, vectors2, words, filename):
    # PCA pour réduire à 2D
    pca = PCA(n_components=2)
    combined = np.vstack([vectors1, vectors2])
    pca_result = pca.fit_transform(combined)
    pca1 = pca_result[:len(vectors1)]
    pca2 = pca_result[len(vectors1):]

    plt.figure(figsize=(12, 8))
    plt.scatter(pca1[:,0], pca1[:,1], label='Avant multilingue', alpha=0.6)
    plt.scatter(pca2[:,0], pca2[:,1], label='Après multilingue', alpha=0.6)
    plt.legend()
    plt.title("Visualisation 2D des embeddings avant/après multilingue")
    plt.savefig(filename)
    plt.close()

def plot_3d_space(vectors1, vectors2, words, filename):
    from mpl_toolkits.mplot3d import Axes3D
    # PCA pour réduire à 3D
    pca = PCA(n_components=3)
    combined = np.vstack([vectors1, vectors2])
    pca_result = pca.fit_transform(combined)
    pca1 = pca_result[:len(vectors1)]
    pca2 = pca_result[len(vectors1):]

    # Calcul des écarts pour colorer
    distances = np.linalg.norm(pca1 - pca2, axis=1)
    norm = (distances - distances.min()) / (distances.max() - distances.min())  # normalisation [0,1]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca1[:,0], pca1[:,1], pca1[:,2], c='blue', label='Avant multilingue', alpha=0.5)
    ax.scatter(pca2[:,0], pca2[:,1], pca2[:,2], c=plt.cm.Reds(norm), label='Après multilingue', alpha=0.7)
    plt.legend()
    plt.title("Visualisation 3D avec écart en rouge")
    plt.savefig(filename)
    plt.close()

def plot_distance_difference(vectors1, vectors2, filename):
    dist1 = cosine_distances(vectors1)
    dist2 = cosine_distances(vectors2)
    diff = np.abs(dist1 - dist2)

    plt.figure(figsize=(12, 10))
    sns.heatmap(diff, cmap='Reds')
    plt.title("Différence de distances cosinus entre les deux espaces")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    file1 = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec"  # ton fichier embeddings monolingue
    file2 = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec"  # ton fichier embeddings après multilingue

    words1, vectors1 = load_embeddings(file1)
    words2, vectors2 = load_embeddings(file2)

    assert words1 == words2, "Les deux fichiers doivent avoir les mêmes mots dans le même ordre."

    # Visualisations
    plot_2d_space(vectors1, vectors2, words1, "embeddings_2D.png")
    plot_3d_space(vectors1, vectors2, words1, "embeddings_3D.png")
    plot_distance_difference(vectors1, vectors2, "difference_distances.png")

    print("Visualisations générées : embeddings_2D.png, embeddings_3D.png, difference_distances.png")
