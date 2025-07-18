import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd, eigh, norm

def parse_arguments():
    parser = argparse.ArgumentParser(description="λMUD: Méthode basée sur les pertubations des espaces sur les valeurs propres et sous espaces engendrés par les vecteurs propres pour mesurer les déformations lors des plongements multilingues de mots")
    parser.add_argument("--embeddings_before", required=True, help="Chemin vers les embeddings source")
    parser.add_argument("--embeddings_after", required=True, help="Chemin vers les embeddings cible")
    parser.add_argument("--subspace_dimenson", type=int, default=100, required=True, help="dimension des sous-espaces à comparer")
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
    
def visualize_vectors_3D(vectors_avant, vectors_apres, legende_base):
    # Convertir en numpy arrays
    vectors_avant = np.array(vectors_avant)
    vectors_apres = np.array(vectors_apres)

    ### === PCA POUR "AVANT" === ###
    mean_avant = np.mean(vectors_avant, axis=0)
    centered_avant = vectors_avant - mean_avant
    cov_avant = np.cov(centered_avant.T)
    eigvals_avant, eigvecs_avant = np.linalg.eig(cov_avant)
    top3_avant = eigvecs_avant[:, np.argsort(eigvals_avant)[::-1][:3]]
    avant_proj = centered_avant @ top3_avant

    # Visualisation pour "avant"
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(avant_proj[:, 0], avant_proj[:, 1], avant_proj[:, 2], color='blue', label='Avant', alpha=0.7)
    ax1.set_title(f'{legende_base} - Espace AVANT')
    ax1.set_xlabel('Composante 1')
    ax1.set_ylabel('Composante 2')
    ax1.set_zlabel('Composante 3')
    ax1.legend()
    ax1.grid(True)
    fig1.savefig(f'{legende_base}_avant.png')
    plt.close(fig1)

    ### === PCA POUR "APRÈS" === ###
    mean_apres = np.mean(vectors_apres, axis=0)
    centered_apres = vectors_apres - mean_apres
    cov_apres = np.cov(centered_apres.T)
    eigvals_apres, eigvecs_apres = np.linalg.eig(cov_apres)
    top3_apres = eigvecs_apres[:, np.argsort(eigvals_apres)[::-1][:3]]
    apres_proj = centered_apres @ top3_apres

    # Visualisation pour "après"
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(apres_proj[:, 0], apres_proj[:, 1], apres_proj[:, 2], color='red', label='Après', alpha=0.7)
    ax2.set_title(f'{legende_base} - Espace APRÈS')
    ax2.set_xlabel('Composante 1')
    ax2.set_ylabel('Composante 2')
    ax2.set_zlabel('Composante 3')
    ax2.legend()
    ax2.grid(True)
    fig2.savefig(f'{legende_base}_apres.png')
    plt.close(fig2)

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

    # visualiser les différentes matrice de covariance avant et après le plongement
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # Heatmap de la matrice de covariance avant le plongement
    sns.heatmap(cov_matrix_avant, ax=ax[0], cmap='viridis')
    ax[0].set_title('Matrice de Covariance Avant le Plongement')
    # fig.savefig('Matrice de Covariance Avant le Plongement')

    # Heatmap de la matrice de covariance après le plongement
    sns.heatmap(cov_matrix_apres, ax=ax[1], cmap='viridis')
    ax[1].set_title('Matrice de Covariance Après le Plongement')
    fig.savefig('Matrice de Covariance Avant et Après Plongement')
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
def subspace_distance(E, F, r):
    """
    Calcule la distance entre deux sous-espaces de dimension r à partir de deux matrices.
    """
    # Décompose les deux matrices (SVD)
    U_e, _, _ = svd(E, full_matrices=False)
    U_f, _, _ = svd(F, full_matrices=False)
    
    # Garde seulement les r premiers vecteurs singuliers
    U_e_r = U_e[:, :r]
    U_f_r = U_f[:, :r]
    
    # Calcule les produits scalaire entre bases
    M = np.dot(U_e_r.T, U_f_r)
    
    # Valeurs singulières des produits croisés
    _, s, _ = svd(M)
    
    # Angles canoniques
    principal_angles = np.arccos(np.clip(s, -1.0, 1.0))
    
    # Distance sinΘ : norme de sin(angles)
    sin_theta = np.sin(principal_angles)
    distance = norm(sin_theta)

    return distance, principal_angles

def davis_kahan_info(Cov_before, Cov_after, r):
    """
    Calcule la norme de perturbation Δ, le gap spectral et la borne Davis-Kahan.

    Args:
        Cov_before (np.ndarray): covariance avant plongement.
        Cov_after (np.ndarray): covariance après plongement.
        r (int): Rang du sous-espace étudié.

    Returns:
        delta_norm (float): ||Δ||₂, norme de la différence de covariance.
        gap (float): λ_r - λ_{r+1}, gap spectral.
        bound (float or None): Borne Davis-Kahan si applicable, sinon None.
    """
    # Perturbation
    Delta = Cov_after - Cov_before
    delta_norm = norm(Delta, ord=2)

    # Valeurs propres triées décroissantes
    eigvals = np.linalg.eigvalsh(Cov_before)
    eigvals = np.sort(eigvals)[::-1]

    print("la taille de l'espace entier est: ", len(eigvals))

    if r > len(eigvals):
        print("⚠️ Le rang r est trop grand par rapport à la taille de l’espace.")
        return delta_norm, None, None

    # Gap spectral : λ_r - λ_{r+1}
    gap = eigvals[r-1] - eigvals[r]

    if gap > 0:
        bound = delta_norm / gap
    else:
        bound = None

    return delta_norm, gap, bound

if __name__ == "__main__":
    # Analyser les arguments en ligne de commande
    args = parse_arguments()
    print("-------------------------------------------------- Calcul de la pertubations engendrées par les valeurs propres -------------------------------------------------------\n")
    vectors_avant, vectors_apres = extract_embeddings(args.embeddings_before, args.embeddings_after)
    visualize_vectors_3D(vectors_avant, vectors_apres, legende_base=('Visualisation des Vecteurs plongement'))
    cov_matrix_avant, cov_matrix_apres = matrices_covariances(vectors_avant, vectors_apres)
    results = weyl_analysis(cov_matrix_avant, cov_matrix_apres)
    print("Maximum des différences de valeurs propres max_eigval_diff :", results["max_eigval_diff"])
    print("pertubations max sur les valeurs propres op_norm_E :", results["op_norm_E"])
    print("\nL'inégalité de Weyl est-elle satisfaite ?")
    print("✅ Oui" if results["weyl_inequality_satisfied"] else "❌ Non")

    print("-------------------------------------------------- Calcul de la pertubations engendrées par les vecteurs propres -------------------------------------------------------\n")
    r = args.subspace_dimenson  - 1
    distance, principal_angles = subspace_distance(vectors_avant, vectors_apres, r)
    delta, gap, bound = davis_kahan_info(cov_matrix_avant, cov_matrix_apres, r)
    print("la valeur de delta est: ", delta)
    print("La distance entre les deux sous espaces est: ", distance)
    if bound is not None:
        print(f"Borne Davis-Kahan : ||sin(Θ)|| ≤ {bound:.4f}")
        if distance <= bound:
            print("✅ L'inégalité de Davis-Kahan est respectée.")
        else:
            print("❌ L'inégalité de Davis-Kahan est violée (cela ne devrait pas arriver).")
