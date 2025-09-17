import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

def load_embeddings(file_path, max_words=50):
    words = []
    vectors = []
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # ignorer la première ligne
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            vec = np.array(list(map(float, parts[1:])))
            words.append(word)
            vectors.append(vec)
            if len(words) >= max_words:
                break
    return words, np.array(vectors)

def plot_global_structure(before_file, after_file, output_file="comparaison_global.png", max_words=50):
    # Charger les embeddings
    words_before, vecs_before = load_embeddings(before_file, max_words=max_words)
    words_after, vecs_after = load_embeddings(after_file, max_words=max_words)

    if words_before != words_after:
        raise ValueError("❌ Les listes de mots ne correspondent pas entre les deux fichiers !")

    # Réduction PCA en 3D
    pca = PCA(n_components=3)
    reduced_before = pca.fit_transform(vecs_before)
    reduced_after = pca.fit_transform(vecs_after)

    # Appliquer l'alignement de Procrustes (rotation/translation/scale)
    mtx1, mtx2, disparity = procrustes(reduced_before, reduced_after)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Avant plongement (noir)
    ax.scatter(mtx1[:, 0], mtx1[:, 1], mtx1[:, 2], color="black", alpha=0.6, label="Avant")

    # Après plongement (bleu)
    ax.scatter(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2], color="blue", alpha=0.6, label="Après (aligné)")

    # On affiche quelques mots pour repères
    for word, (x, y, z) in zip(words_before[:10], mtx1[:10]):
        ax.text(x, y, z, word, fontsize=8, color="black")
    for word, (x, y, z) in zip(words_after[:10], mtx2[:10]):
        ax.text(x, y, z, word, fontsize=8, color="blue")

    ax.set_title("Comparaison globale des structures (Procrustes)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Graphique global sauvegardé sous {output_file} (disparity={disparity:.4f})")

# Exemple d’utilisation :
plot_global_structure("/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_monolingual.vec", "/lium/raid-b/matang/PHD/Experimentations/Testament/GMUD/rapport/categorical-modularity/en-grec/EN_SEMEVAL17_BINDER_crosslingual.vec", "global_result.png", max_words=50)
