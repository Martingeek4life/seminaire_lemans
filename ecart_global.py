import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

def procrustes_alignment(X, Y):
    # Procrustes Analysis: retourne X-aligned à Y, et l'écart global
    mtx1, mtx2, disparity = procrustes(Y, X)  # Y = référence, X = à transformer
    return mtx2, disparity  # mtx2 = X aligné sur Y

def plot_3d_arrows(Y, X_aligned, words, filename):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Points référence
    ax.scatter(Y[:,0], Y[:,1], Y[:,2], c='blue', label='Avant (référence)', alpha=0.6)
    # Points alignés
    ax.scatter(X_aligned[:,0], X_aligned[:,1], X_aligned[:,2], c='green', label='Après aligné', alpha=0.6)

    # Flèches représentant le décalage
    for i in range(Y.shape[0]):
        ax.plot([X_aligned[i,0], Y[i,0]],
                [X_aligned[i,1], Y[i,1]],
                [X_aligned[i,2], Y[i,2]],
                c='red', alpha=0.6)

    ax.set_title("Décalage structurel après alignement (flèches rouges)")
    ax.legend()
    plt.savefig(filename)
    plt.close()

def plot_disparity_heatmap(Y, X_aligned, filename):
    # Calcul des distances des écarts après alignement
    distances = np.linalg.norm(Y - X_aligned, axis=1)
    plt.figure(figsize=(12, 6))
    sns.heatmap(distances.reshape(1, -1), cmap='Reds', cbar_kws={'label':'Écart'})
    plt.title("Écart structurel global après alignement")
    plt.xlabel("Mots")
    plt.ylabel("Écart")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    file1 = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec"   # embeddings avant
    file2 = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec"   # embeddings après

    words1, vectors1 = load_embeddings(file1)
    words2, vectors2 = load_embeddings(file2)

    assert words1 == words2, "Les deux fichiers doivent avoir les mêmes mots dans le même ordre."

    # On réduit à 3D pour la visualisation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    vectors1_3d = pca.fit_transform(vectors1)
    vectors2_3d = pca.transform(vectors2)

    # Alignement Procrustes
    vectors2_aligned, disparity = procrustes_alignment(vectors2_3d, vectors1_3d)
    print(f"Disparité globale après Procrustes: {disparity:.6f}")

    # Visualisation 3D avec flèches rouges montrant le décalage
    plot_3d_arrows(vectors1_3d, vectors2_aligned, words1, "decalage_3D.png")
    # Heatmap des écarts
    plot_disparity_heatmap(vectors1_3d, vectors2_aligned, "decalage_heatmap.png")

    print("Visualisations générées : decalage_3D.png, decalage_heatmap.png")
