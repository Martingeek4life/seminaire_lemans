from __future__ import print_function
import argparse
import string
import subprocess
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt
import os

# cette fonction charge les embeddings de mots à partir d'un fichier, les stocke dans un format approprié et renvoie à la fois les embeddings et les mots associés.

def reshap_embedding(word_embedding, word_dim):
    en_model = KeyedVectors.load_word2vec_format(word_embedding)

    # Getting tokens and vectors
    words = []
    embeddings = []
    
    # Itérer sur tous les mots dans le modèle
    for word in en_model.key_to_index:
        words.append(word)
        embeddings.append(en_model.get_vector(word))

    # Convertir la liste d'embeddings en tableau numpy
    embeddings = np.array(embeddings)
    return embeddings, words


# cette fonction plot les embedding dans un espace

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 10))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.title(filename)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)
    # Fermeture de la figure pour libérer les ressources
    plt.close()

# cette fonction calcule et retrourne les voisins d'un mot donné et les distance entre ce mot et ses différents voisins

# k = 51  50 voisins + le mot lui-même

# cette fonction prend entree les embedings entrée 
def search_Word_Nearest_Neighbors_embedding(embeddings, word_list, distance, k):
    # Calculer les distances cosinus/euclidiennes entre les embeddings
    distances = pairwise_distances(embeddings, metric= distance)

    # Initialiser l'algorithme des k plus proches voisins

    neighbors_algorithm = NearestNeighbors(n_neighbors=k, metric='precomputed', algorithm='brute')
    neighbors_algorithm.fit(distances)

    # Extraire les 50 voisins les plus proches pour chaque mot
    all_neighbors = neighbors_algorithm.kneighbors(distances, return_distance=True)
    word_data_for_graph_test = []
    word_data_for_graph = []
    j = 0
    # Parcourir chaque mot et ses voisins 
    for i, word in enumerate(word_list):
        neighbors_indices = all_neighbors[1][i][1:]  # Exclure le mot lui-même
        neighbors_distances = all_neighbors[0][i][1:]  # Exclure la distance au mot lui-même
        
        # Récupérer les mots voisins et leurs distances
        neighbor_words = [word_list[idx] for idx in neighbors_indices]
        neighbor_distances = neighbors_distances
        
        
        # je construis l'objet word_data_for_graph à envoyer au graphe, le nombre de mots est un hyperPrametre nb_words
        # juste pour afficher les graphes de 10 mots et visualiser
        if 200 < j <= 210:
            data_test = {
                'word': word,
                'neighbors': neighbor_words,
                'distances': neighbor_distances
            }
            word_data_for_graph_test.append(data_test)
            # Augmentation du compteur
        j += 1


        # word data for graph total sur tous les mots

        data = {
            'word': word,
            'neighbors': neighbor_words,
            'distances': neighbor_distances
        }
        word_data_for_graph.append(data)

        # Imprimer ou stocker les informations
        print(f"Voisins de '{word}':")
        for neighbor, distance in zip(neighbor_words, neighbor_distances):
            print(f"{neighbor} (Distance: {distance:.4f})")
        print()
    print("je construis l'objet word_data_for_graph à envoyer au graphe")
    print(word_data_for_graph)

    return word_data_for_graph, word_data_for_graph_test

# cette fonction permet de dessiner le graphe d'un mot avec ses voisins pondéré par la distance entre le mots et ses différents voisins
def Graph_for_word_embedding_neighbor(word_data_for_graph_test, state):
    # Création du graphe pondéré

    # ici c'est pour afficher le graphes des mots test juste pour visualiser
    for data in word_data_for_graph_test:
        G_test = nx.Graph()
        word = data['word']
        neighbors = data['neighbors']
        distances = data['distances']

        G_test.add_node(word)
        for neighbor, distance in zip(neighbors, distances):
            G_test.add_edge(word, neighbor, weight=distance)

        # Visualisation du graphe pondéré pour ce mot
        pos = nx.spring_layout(G_test)
        plt.figure(figsize=(18, 10))
        nx.draw(G_test, pos, with_labels=True, node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(G_test, 'weight')
        nx.draw_networkx_edge_labels(G_test, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"Graph for {word} and its Neighbors")
    
        # Génère un nom de fichier unique en utilisant le nom du mot
        filename = f"graphe_{word}_et_ses_voisins_{state}.png"
        plt.savefig(filename)
        # plt.show()
        plt.close()
        
    # print("----------- len word_data_for_graph -----------")
    # print(len(word_data_for_graph))
    # print(word_data_for_graph)


# cette fonction prend en entrée le mot central et sa liste des voisins et un word data graph contenant les liste des mots, voisins et disatances et retourne la liste des distances des voisins passés en parametre
def neighbor_distances_extraction(central_word, neighbors_list, word_data_for_graph):
    neighbors_distances = []
    for graph_data in word_data_for_graph:
        if graph_data['word'] == central_word:
            for i in range(0, len(neighbors_list)):
                for j in range(0, len(graph_data['neighbors'])):
                    if neighbors_list[i] == graph_data['neighbors'][j]:
                        neighbors_distances.append(graph_data['distances'][j])
    return neighbors_distances

# cette fonction me retourne dans un tableau les indices des n elements les plus petits du tableau nums
def indices_of_smallest(nums, n=10):
    if n >= len(nums):
        return list(range(len(nums)))
    return np.argpartition(nums, n)[:n]

# cette fonction prend en parametre un noeud et retourne ses 10 voisins les plus proches avec leurs distances
def Ten_nearest_neighbor(central_word, word_data_for_graph):
    for graph_data in word_data_for_graph:
        if graph_data['word'] == central_word:
            Ten_indices = indices_of_smallest(graph_data['distances'])
            neighbor = []
            distances = []
            for j in range(0, len(Ten_indices)):
                neighbor.append(graph_data['neighbors'][Ten_indices[j]])
                distances.append(graph_data['distances'][Ten_indices[j]])
            Ten_nearest = {
                'central': central_word,
                'neighbor': neighbor,
                'distances': distances 
            }
    return Ten_nearest

def common_lost_appeared_neighbors_extraction(word_data_for_graph_before, word_data_for_graph_after):
    # je Crée un dictionnaire pour chaque liste pour un accès plus facile par mot
    dict_liste_before = {element['word']: element for element in word_data_for_graph_before}
    dict_liste_after = {element['word']: element for element in word_data_for_graph_after}

    common_neighbors = []
    lost_neighbors =  []
    appeared_neighbors = []
    j = 0
    # je Parcours les éléments de même rang i des deux listes
    with open("log_common_lost_appear.txt", 'w') as file:
        for mot in dict_liste_before.keys() & dict_liste_after.keys():
            element_liste_before = dict_liste_before[mot]
            element_liste_after = dict_liste_after[mot]
    
            # je Trouve les voisins communs avec leurs distances
            voisins_communs = set(element_liste_before['neighbors']) & set(element_liste_after['neighbors'])
            print("nb voisins communs du mot: ", len(voisins_communs))
            print("voisins communs du mot: ", voisins_communs)
            # pour ces mots voisins communs j'extrait chacune de leurs distances avant et apres le plongement multilingue
            before_distance = neighbor_distances_extraction(mot, list(voisins_communs), word_data_for_graph_before)
            after_distance = neighbor_distances_extraction(mot, list(voisins_communs), word_data_for_graph_after)
    
            # Créez un dictionnaire pour le résultat
            noeud_constant = {
                'word': mot,
                'common_neighbors': list(voisins_communs),
                'before_PM_distances': before_distance,
                'after_PM_distances': after_distance,
            }
    
            common_neighbors.append(noeud_constant)
    
        # je Trouve les voisins présents dans avant le PM et absents dans Apres le PM
            voisins_absents = set(element_liste_before['neighbors']) - set(element_liste_after['neighbors'])
    
            lost_distances = neighbor_distances_extraction(mot, list(voisins_absents), word_data_for_graph_before)
            print("nombre de voisins disparut du mot: ", len(voisins_absents))
            print("voisins disparut du mot: ", voisins_absents)
            # Créez un dictionnaire pour le résultat
            noeud_abscent = {
                'word': mot,
                'lost_neighbors': list(voisins_absents),
                'lost_distances': lost_distances
            }
            lost_neighbors.append(noeud_abscent)
    
        # je Trouve les voisins absents avant le PM et apparut Apres le PM
            voisins_apparut = set(element_liste_after['neighbors']) - set(element_liste_before['neighbors']) 
            appeared_distances = neighbor_distances_extraction(mot, list(voisins_apparut), word_data_for_graph_after)
            print("nombre de voisins apparut du mot: ", len(voisins_apparut))
            print("voisins apparut du mot: ", voisins_apparut)
            # Créez un dictionnaire pour le résultat
            noeud_apparut = {
                'word': mot,
                'appeared_neighbors': list(voisins_apparut),
                'appeared_distances': appeared_distances
            }
            appeared_neighbors.append(noeud_apparut)
    
            # j'affiche les voisins communs, abscents et apparut des 10 mots situé entre le 200 ieme et le 210 ieme
            if 200 < j <= 210:
                print("les voisins conservés, apparut et disparrut du mots  \n", mot)
                print("mots voisins conservés après le plongement multilingue: \n", noeud_constant)
                print("mots voisins perdus apres le plongement multilingue: \n", noeud_abscent)
                print("mots voisins apparut après le plongement multilingue: \n", noeud_apparut)
            j = j + 1
            file.write(f'-------------------------------- analyse de la taille des lots, common, appear ------------------------------ \n')
            file.write(f'nb voisins communs du mot: {len(voisins_communs)}\n')
            for voisin in voisins_communs:
                file.write(f'voisin communs: {voisin}\n')
            file.write(f'nb voisins apparut du mot: {len(voisins_apparut)}\n')
            for voisin in voisins_apparut:
                file.write(f'voisin communs: {voisin}\n')
            file.write(f'nb voisins disparut du mot: {len(voisins_absents)}\n')
            for voisin in voisins_absents:
                file.write(f'voisin communs: {voisin}\n')

    return common_neighbors, lost_neighbors, appeared_neighbors


def mean_GMUD(common_neighbors, lost_neighbors, appeared_neighbors, beta1, beta2, beta3):
    # il me faut avoir un tableau de deformation Di de chaque mot Wi

    # D est le tableau des deformation de chaque mot Wi

    D = []
    somme = 0
    V_p = 0
    V_d = 0
    v_c = 0
    for i in range(0, len(common_neighbors)):
        common_i = 0
        lost_i = 0
        appeared_i = 0
        D_i = 0
        for j in range(0, len(common_neighbors[i]['common_neighbors'])):
            common_i = common_i + beta1 * abs(common_neighbors[i]['after_PM_distances'][j] - common_neighbors[i]['before_PM_distances'][j])

        lost_i = lost_i + beta2*len(lost_neighbors[i]['lost_neighbors'])
        
        appeared_i = appeared_i + beta3*len(appeared_neighbors[i]['appeared_neighbors'])

        D_i = common_i + lost_i + appeared_i
        V_p = V_p + appeared_i
        V_d = V_d + lost_i
        v_c = v_c + common_i
        somme = somme + D_i
        D.append(D_i)
        
        print(f"Word {i}: Common_i = {common_i}, Lost_i = {lost_i}, Appeared_i = {appeared_i}")
        print(f"Word {i}: V_p = {V_p}, V_d = {V_d}, v_c = {v_c}")
        
    # Calcul de l'écart type
    deviation_euclidian_GMUD = np.std(D)
    print("la taille du vocabulaire est v: \n", len(common_neighbors))
    moy = somme/len(common_neighbors)
    print("la somme euclidian_GMUD est: \n", somme)
    print("la moyenne euclidian_GMUD est: \n", moy)
    print("la moyenne vp est: \n", v_p/len(common_neighbors))
    print("la moyenne vd est: \n", v_d/len(common_neighbors))
    print("la moyenne vc est: \n", v_c/len(common_neighbors))
    print("L'écart type  euclidian_GMUD est: \n", deviation_euclidian_GMUD)

    return deviation_euclidian_GMUD, somme, moy, D, common_neighbors, V_p, V_d, v_c

# cette fonction ecrit les resultats(deformations, moyenne, la metric_distance, chaque mot avec sa deformations) dans un fichier pour analyse

def Analyse(Tab_deformations, list_mots, mean_Gmud, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, Apres_avant):
    fichier_sortie='analyse_deformations.txt'
    with open(fichier_sortie, 'w') as file:
        file.write(f'-------------------------------- Analyse des deformations GMUD ------------------------------ \n')
        file.write(f'Métrique de distance: {args.distance_metric}\n')
        file.write(f'Déformation moyenne: {mean_Gmud}\n')
        file.write(f'Somme des Déformations: {sum_GMUD}\n')
        file.write(f'EcartType des Déformations: {deviation_euclidian_GMUD}, v_p: {V_p}, v_d: {V_d}, v_c: {Apres_avant}\n')
        file.write('Déformations par mot:\n')
        
        for i in range(0, len(Tab_deformations)):
            file.write(f'{list_mots[i]["word"]}: {Tab_deformations[i]}\n')


# fonction pour gérer les arguments en ligne de commande
def parse_arguments():
    parser = argparse.ArgumentParser(description="GMUD: Méthode basée sur les graphes pour mesurer les déformations lors des plongements multilingues de mots")
    parser.add_argument("--distance_metric", default="euclidean", help="Métrique de distance à utiliser (cosine ou euclidean)")
    parser.add_argument("--nb_neighbors", type=int, default=101, help="Nombre de voisins à considérer")
    parser.add_argument("--beta1", type=float, default=1.0, help="Coefficient pour la déformation des voisins conservés")
    parser.add_argument("--beta2", type=float, default=1.0, help="Coefficient pour la déformation des voisins perdus")
    parser.add_argument("--beta3", type=float, default=1.0, help="Coefficient pour la déformation des voisins apparus")
    parser.add_argument("--embeddings_before", required=True, help="Chemin vers les embeddings source")
    parser.add_argument("--embeddings_after", required=True, help="Chemin vers les embeddings cible")
    # parser.add_argument("--lang", choices=["source", "target"], required=True, help="Langue sur laquelle effectuer le calcul de déformation (source ou target)")
    return parser.parse_args()

if __name__ == "__main__":
    # Analyser les arguments en ligne de commande
    args = parse_arguments()

    print("Calcul de la déformation moyenne GMUD")
    source_output_path = "source_embeddings"
    target_output_path = "target_embeddings"

    source_cross_path = "source_crosslingual.vec"
    target_cross_path = "target_crosslingual.vec"

    ext = ".vec"


    """     if args.lang == "source":
        embeddings_before, words_before = reshap_embedding(source_output_path+ext, 300)
        embeddings_after, words_after = reshap_embedding(source_cross_path, 300)
    else:
        embeddings_before, words_before = reshap_embedding(target_output_path+ext, 300)
        embeddings_after, words_after = reshap_embedding(target_cross_path , 300) """

    embeddings_before, words_before = reshap_embedding(args.embeddings_before, 300)
    embeddings_after, words_after = reshap_embedding(args.embeddings_after, 300)

    # Creating the tsne plot [Warning: will take time]
    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    # Charger les embeddings avant le plongement
    low_dim_embedding = tsne.fit_transform(embeddings_before)
    filename_target_visualisation_before = 'language words embeddings before PM'
    # Finally plotting and saving the fig
    print("--------------------- visualisation of language words embeddings space before PM -------------------------\n")
    plot_with_labels(low_dim_embedding, words_before, filename_target_visualisation_before)

    # Charger les embeddings après le plongement
    low_dim_embedding = tsne.fit_transform(embeddings_after)
    filename_target_visualisation_after= 'language words embeddings after PM'
    # Finally plotting and saving the fig
    print("--------------------- visualisation of language words embeddings space after PM -------------------------\n")
    plot_with_labels(low_dim_embedding, words_after, filename_target_visualisation_after)

    # Calculer les voisins pour les embeddings avant le plongement
    word_data_for_graph_before, word_data_for_graph_test_before = search_Word_Nearest_Neighbors_embedding(embeddings_before, words_before, args.distance_metric, args.nb_neighbors)
    print("--------------------- visualisation of Graph for language words embeddings neighbors before PM -------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_test_before, state= "before")

    # afficher dans un fichier chaque mot ses 10 voisins les plus proches avant le plongement
    fichier_sortie='analyse_voisins_avant.txt'
    with open(fichier_sortie, 'w') as file:
        for graph_data in word_data_for_graph_before:
            nearest_before = Ten_nearest_neighbor(graph_data['word'] , word_data_for_graph_before)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins avant le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_before['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_before['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')

    # Calculer les voisins pour les embeddings après le plongement
    word_data_for_graph_after, word_data_for_graph_test_after = search_Word_Nearest_Neighbors_embedding(embeddings_after, words_after, args.distance_metric, args.nb_neighbors)
    print("--------------------- visualisation of Graph for language words embeddings neighbors  after PM-------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_test_after, state= "after")

    # afficher dans un fichier chaque mot ses 10 voisins les plus proches après le plongement
    fichier_sortie='analyse_voisins_après.txt'
    with open(fichier_sortie, 'w') as file:
        for graph_data in word_data_for_graph_after:
            nearest_after = Ten_nearest_neighbor(graph_data['word'] , word_data_for_graph_after)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins après le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_after['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_after['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')
    # Extraire les voisins communs, perdus et apparus

    print("--------------------  Extraire les voisins communs, perdus et apparus ---------------------------")
    common_neighbors, lost_neighbors, appeared_neighbors = common_lost_appeared_neighbors_extraction(word_data_for_graph_before, word_data_for_graph_after)

    # Calculer la déformation moyenne GMUD
    # beta1, beta2, beta3 = 0.4, 0.3, 0.3
    deviation_euclidian_GMUD, sum_GMUD, mean, Tab_deformations, list_mots, V_p, V_d, Apres_avant = mean_GMUD(common_neighbors, lost_neighbors, appeared_neighbors, args.beta1, args.beta2, args.beta3)

    Analyse(Tab_deformations, list_mots, mean, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, Apres_avant)

    print("La déformation moyenne avec la métrique GMUD est de:", mean)
    print("La somme des déformations avec la métrique GMUD est de:", sum_GMUD)
    print("L'écart-type de la déformation avec la métrique GMUD est de:", deviation_euclidian_GMUD)
