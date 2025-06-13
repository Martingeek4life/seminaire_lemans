import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd, eigh

def parse_arguments():
    parser = argparse.ArgumentParser(description="λMUD: Méthode basée sur les pertubations des espaces sur les valeurs propres et sous espaces engendrés par les vecteurs propres pour mesurer les déformations lors des plongements multilingues de mots")
    parser.add_argument("--embeddings_before", required=True, help="Chemin vers les embeddings source")
    parser.add_argument("--embeddings_after", required=True, help="Chemin vers les embeddings cible")
    return parser.parse_args()

def extract_embeddings(path_vecteurs_avant, path_vecteurs_apres):
    vectors_avant = []
    vectors_apres = []

    with open(path_vecteurs_avant, 'r', encoding='utf-8') as file:
        next(file)  # Ignorer la première ligne
        for line in file:
            parts = line.strip().split()[1:]  # Ignorer le mot lui-même, prendre seulement les nombres
            vectors_avant.append([float(part) for part in parts])

    # Lire le fichier après le plongement et extraire les vecteurs
    with open(path_vecteurs_apres, 'r', encoding='utf-8') as file:
        next(file)  # Ignorer la première ligne
        for line in file:
            parts = line.strip().split()[1:]  # Ignorer le mot lui-même, prendre seulement les nombres
            vectors_apres.append([float(part) for part in parts])
    return vectors_avant, vectors_apres

def visualize_vectors_3D(vectors_avant, vectors_apres, legende):
    # Convertir en numpy arrays
    vectors_avant = np.array(vectors_avant)
    vectors_apres = np.array(vectors_apres)

    # Concaténer les deux matrices pour avoir un espace PCA commun
    all_vectors = np.vstack((vectors_avant, vectors_apres))

    # Centrer les données
    all_vectors_mean = np.mean(all_vectors, axis=0)
    all_vectors_centered = all_vectors - all_vectors_mean

    # Appliquer la PCA manuellement (ou utiliser sklearn)
    covariance_matrix = np.cov(all_vectors_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    indices_sorted = np.argsort(eigenvalues)[::-1]
    top3_eigenvectors = eigenvectors[:, indices_sorted[:3]]

    # Projeter les deux ensembles dans l’espace PCA
    avant_proj = (vectors_avant - all_vectors_mean).dot(top3_eigenvectors)
    apres_proj = (vectors_apres - all_vectors_mean).dot(top3_eigenvectors)

    # Visualisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(avant_proj[:, 0], avant_proj[:, 1], avant_proj[:, 2], color='blue', label='Avant', alpha=0.7)
    ax.scatter(apres_proj[:, 0], apres_proj[:, 1], apres_proj[:, 2], color='red', label='Après', alpha=0.7)

    ax.set_title(legende)
    ax.set_xlabel('Composante 1')
    ax.set_ylabel('Composante 2')
    ax.set_zlabel('Composante 3')
    ax.legend()
    ax.grid(True)

    # Sauvegarder la figure
    fig.savefig(legende + '.png')
    plt.close()

def matrices_covariances(vecteurs_avant, vecteurs_apres):
    
    print("---------------------------------Calcul de la matrice de covariance de l'espace avant le plongement ------------------------------")
    # Convertir la liste en un array NumPy
    vectors_avant = np.array(vecteurs_avant)
    print("les vecteurs avant: ", len(vectors_avant))
    cov_matrix_avant = np.cov(vectors_avant, rowvar=False)
    print("La matrice de covariace de l'espace avant: ", cov_matrix_avant)

    # Convertir la liste en un array NumPy
    print("---------------------------------Calcul de la matrice de covariance de l'espace apres le plongement ------------------------------")
    vectors_apres = np.array(vecteurs_apres)
    print("les vecteurs après: ", vectors_apres)
    cov_matrix_apres = np.cov(vectors_apres, rowvar=False)
    print("La matrice de covariace de l'espace apres: ", cov_matrix_apres)
    visualize_vectors_3D(vectors_avant, vectors_apres, legende=('Visualisation des Vecteurs avant et après le plongement'))

    # visualiser les différentes matrice de covariance avant et après le plongement
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # Heatmap de la matrice de covariance avant le plongement
    sns.heatmap(cov_matrix_avant, ax=ax[0], cmap='viridis')
    ax[0].set_title('Matrice de Covariance Avant le Plongement')

    # Heatmap de la matrice de covariance après le plongement
    sns.heatmap(cov_matrix_apres, ax=ax[1], cmap='viridis')
    ax[1].set_title('Matrice de Covariance Après le Plongement')
    fig.savefig('Matrice de Covariance Après le Plongement')
    # plt.show()
    plt.close()

    return cov_matrix_avant, cov_matrix_apres
def weyl_analysis(cov_A, cov_A_perturbed):
    
    # Matrice de perturbation
    E = cov_A_perturbed - cov_A
    
    # Valeurs propres
    eigvals_A = np.linalg.eigvalsh(cov_A)
    eigvals_perturbed = np.linalg.eigvalsh(cov_A_perturbed)
    eigval_diffs = np.abs(eigvals_perturbed - eigvals_A)
    
    # Norme opérateur de la perturbation
    op_norm_E = np.linalg.norm(E, ord=2)
    
    return {
        "eigvals_A": eigvals_A,
        "eigvals_perturbed": eigvals_perturbed,
        "eigval_diffs": eigval_diffs,
        "max_eigval_diff": np.max(eigval_diffs),
        "op_norm_E": op_norm_E,
        "weyl_inequality_satisfied": np.all(eigval_diffs <= op_norm_E + 1e-10)
    }

if __name__ == "__main__":
    # Analyser les arguments en ligne de commande
    args = parse_arguments()
    vectors_avant, vectors_apres = extract_embeddings(args.embeddings_before, args.embeddings_after)
    visualize_vectors_3D(vectors_avant, vectors_apres, legende=('Visualisation des Vecteurs avant et après le plongement'))
    cov_matrix_avant, cov_matrix_apres = matrices_covariances(vectors_avant, vectors_apres)
    results = weyl_analysis(cov_matrix_avant, cov_matrix_apres)
    print("Maximum des différences de valeurs propres :", results["max_eigval_diff"])
    print(results["op_norm_E"])
    print("\nL'inégalité de Weyl est-elle satisfaite ?")
    print("✅ Oui" if results["weyl_inequality_satisfied"] else "❌ Non")

