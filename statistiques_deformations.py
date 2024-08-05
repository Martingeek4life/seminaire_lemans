import numpy as np
import argparse

# Définir les arguments de la ligne de commande
parser = argparse.ArgumentParser(description='Analyse des déformations')
parser.add_argument('nom_fichier', type=str, help='Chemin vers le fichier de déformations')

# Parser les arguments
args = parser.parse_args()
nom_fichier = args.nom_fichier

# Ouvrir le fichier pour la lecture
with open(nom_fichier, 'r') as fichier:
    # Sauter les six premières lignes
    for _ in range(6):
        next(fichier)
    
    # Lire les données à partir de la 7ème ligne
    List_mots = []
    List_deformations = []
    for ligne in fichier:
        # Construire la liste de mots et la liste des valeurs de déformations
        List_mots.append(ligne.strip().split(":")[0])
        List_deformations.append(float(ligne.strip().split(":")[1]))
    
    # Convertir la liste en un array numpy pour les calculs
    deformations_array = np.array(List_deformations)
    
    # Calculer la moyenne
    moyenne = np.mean(deformations_array)
    print(f'Moyenne des déformations: {moyenne}')
    
    # Calculer l'écart-type
    ecart_type = np.std(deformations_array)
    print(f'Écart-type des déformations: {ecart_type}')
    
    # Calculer la médiane
    mediane = np.median(deformations_array)
    print(f'Médiane des déformations: {mediane}')
    
    # Calculer le minimum
    minimum = np.min(deformations_array)
    print(f'Minimum des déformations: {minimum}')
    
    # Calculer le maximum
    maximum = np.max(deformations_array)
    print(f'Maximum des déformations: {maximum}')
    
    # Calculer le quartile 1 (Q1)
    Q1 = np.percentile(deformations_array, 25)
    print(f'Premier quartile (Q1) des déformations: {Q1}')
    
    # Calculer le quartile 3 (Q3)
    Q3 = np.percentile(deformations_array, 75)
    print(f'Troisième quartile (Q3) des déformations: {Q3}')
    
    # Calculer l'intervalle interquartile (IQR)
    IQR = Q3 - Q1
    print(f'Intervalle interquartile (IQR) des déformations: {IQR}')
