import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    # Centrer l'espace
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    pca = PCA(n_components=3)
    pca.fit(centered)
    axes = pca.components_ * np.sqrt(pca.explained_variance_).reshape(3,1)  # échelle par variance
    return mean, axes

def plot_3d_spaces(vectors_before, vectors_after, words):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calcul PCA axes
    mean_before, axes_before = compute_pca_axes(vectors_before)
    mean_after, axes_after = compute_pca_axes(vectors_after)

    # Affichage nuages de points pour contexte
    ax.scatter(vectors_before[:,0], vectors_before[:,1], vectors_before[:,2], c='blue', alpha=0.3, label='Nuage avant')
    ax.scatter(vectors_after[:,0], vectors_after[:,1], vectors_after[:,2], c='green', alpha=0.3, label='Nuage après')

    # Affichage axes avant (pointillés)
    for i in range(3):
        ax.plot([mean_before[0], mean_before[0]+axes_before[i,0]],
                [mean_before[1], mean_before[1]+axes_before[i,1]],
                [mean_before[2], mean_before[2]+axes_before[i,2]],
                'b--', linewidth=3, label=f'Axe avant {i+1}' if i==0 else "")
    
    # Affichage axes après (pleins)
    for i in range(3):
        ax.plot([mean_after[0], mean_after[0]+axes_after[i,0]],
                [mean_after[1], mean_after[1]+axes_after[i,1]],
                [mean_after[2], mean_after[2]+axes_after[i,2]],
                'r-', linewidth=3, label=f'Axe après {i+1}' if i==0 else "")

    ax.set_title("Transformation de l'espace vectoriel avant/après")
    ax.legend()
    plt.savefig("transformation_espace_3D.png")
    plt.show()

if __name__ == "__main__":
    file_before = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec"   # embedd>    
    file_after = "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec"

    words_before, vectors_before = load_embeddings(file_before)
    words_after, vectors_after = load_embeddings(file_after)

    assert words_before == words_after, "Les deux fichiers doivent avoir les mêmes mots dans le même ordre."

    # Réduction à 3D si nécessaire (PCA)
    pca = PCA(n_components=3)
    vectors_before_3d = pca.fit_transform(vectors_before)
    vectors_after_3d = pca.transform(vectors_after)

    # Visualisation de la transformation des espaces
    plot_3d_spaces(vectors_before_3d, vectors_after_3d, words_before)
