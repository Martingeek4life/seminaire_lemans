from __future__ import print_function
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt

# Cette fonction charge les embeddings de mots à partir d'un fichier, les stocke dans un format approprié et renvoie à la fois les embeddings et les mots associés.
def reshap_embedding(word_embedding, word_dim, limit):
    en_model = KeyedVectors.load_word2vec_format(word_embedding)

    # Getting tokens and vectors
    words = []
    embeddings = []
    i = 0
    for word in en_model.key_to_index:  # Iterate over keys using key_to_index
        if i == limit:
            break
        words.append(word)
        embeddings.append(en_model.get_vector(word))
        i += 1

    embeddings = np.array(embeddings)
    return embeddings, words

# Cette fonction plot les embedding dans un espace
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
    plt.close()

# Cette fonction calcule et retourne les voisins d'un mot donné et les distances entre ce mot et ses différents voisins
def search_Word_Nearest_Neighbors_embedding(embeddings, word_list, distance, k, start=200, nb_words=210):
    distances = pairwise_distances(embeddings, metric=distance)
    neighbors_algorithm = NearestNeighbors(n_neighbors=k, metric='precomputed', algorithm='brute')
    neighbors_algorithm.fit(distances)

    all_neighbors = neighbors_algorithm.kneighbors(distances, return_distance=True)
    word_data_for_graph_test = []
    word_data_for_graph = []
    j = 0

    for i, word in enumerate(word_list):
        neighbors_indices = all_neighbors[1][i][1:]  # Exclure le mot lui-même
        neighbors_distances = all_neighbors[0][i][1:]  # Exclure la distance au mot lui-même

        neighbor_words = [word_list[idx] for idx in neighbors_indices]
        neighbor_distances = neighbors_distances

        if start < j <= nb_words:
            data_test = {
                'word': word,
                'neighbors': neighbor_words,
                'distances': neighbor_distances
            }
            word_data_for_graph_test.append(data_test)
        j += 1

        data = {
            'word': word,
            'neighbors': neighbor_words,
            'distances': neighbor_distances
        }
        word_data_for_graph.append(data)

        print(f"Voisins de '{word}':")
        for neighbor, distance in zip(neighbor_words, neighbor_distances):
            print(f"{neighbor} (Distance: {distance:.4f})")
        print()
    print("je construis l'objet word_data_for_graph à envoyer au graphe")
    print(word_data_for_graph)

    return word_data_for_graph, word_data_for_graph_test

# Cette fonction permet de dessiner le graphe d'un mot avec ses voisins pondéré par la distance entre le mot et ses différents voisins
def Graph_for_word_embedding_neighbor(word_data_for_graph, word_data_for_graph_test, before):
    for data in word_data_for_graph_test:
        G_test = nx.Graph()
        word = data['word']
        neighbors = data['neighbors']
        distances = data['distances']

        G_test.add_node(word)
        for neighbor, distance in zip(neighbors, distances):
            G_test.add_edge(word, neighbor, weight=distance)

        pos = nx.spring_layout(G_test)
        plt.figure(figsize=(18, 10))
        nx.draw(G_test, pos, with_labels=True, node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(G_test, 'weight')
        nx.draw_networkx_edge_labels(G_test, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"Graph for {word} and its Neighbors")
    
        filename = f"graphe_{word}_et_ses_voisins_{before}.png"
        plt.savefig(filename)
        plt.close()

# Cette fonction prend en entrée le mot central et sa liste des voisins et un word data graph contenant les listes des mots, voisins et distances et retourne la liste des distances des voisins passés en paramètre
def neighbor_distances_extraction(central_word, neighbors_list, word_data_for_graph):
    neighbors_distances = []
    for graph_data in word_data_for_graph:
        if graph_data['word'] == central_word:
            for neighbor in neighbors_list:
                if neighbor in graph_data['neighbors']:
                    index = graph_data['neighbors'].index(neighbor)
                    neighbors_distances.append(graph_data['distances'][index])
    return neighbors_distances

# Cette fonction retourne dans un tableau les indices des n éléments les plus petits du tableau nums
def indices_of_smallest(nums, n=10):
    if n >= len(nums):
        return list(range(len(nums)))
    return np.argpartition(nums, n)[:n]

# Cette fonction prend en paramètre un nœud et retourne ses 10 voisins les plus proches avec leurs distances
def Ten_nearest_neighbor(central_word, word_data_for_graph):
    for graph_data in word_data_for_graph:
        if graph_data['word'] == central_word:
            Ten_indices = indices_of_smallest(graph_data['distances'])
            neighbor = [graph_data['neighbors'][idx] for idx in Ten_indices]
            distances = [graph_data['distances'][idx] for idx in Ten_indices]
            Ten_nearest = {
                'central': central_word,
                'neighbor': neighbor,
                'distances': distances 
            }
            return Ten_nearest

def common_lost_appeared_neighbors_extraction(word_data_for_graph_before, word_data_for_graph_after):
    dict_liste_before = {element['word']: element for element in word_data_for_graph_before}
    dict_liste_after = {element['word']: element for element in word_data_for_graph_after}

    common_neighbors = []
    lost_neighbors = []
    appeared_neighbors = []
    j = 0
    for mot in dict_liste_before.keys() & dict_liste_after.keys():
        element_liste_before = dict_liste_before[mot]
        element_liste_after = dict_liste_after[mot]

        voisins_communs = set(element_liste_before['neighbors']) & set(element_liste_after['neighbors'])
        before_distance = neighbor_distances_extraction(mot, list(voisins_communs), word_data_for_graph_before)
        after_distance = neighbor_distances_extraction(mot, list(voisins_communs), word_data_for_graph_after)

        noeud_constant = {
            'word': mot,
            'common_neighbors': list(voisins_communs),
            'before_PM_distances': before_distance,
            'after_PM_distances': after_distance,
        }
        common_neighbors.append(noeud_constant)

        voisins_absents = set(element_liste_before['neighbors']) - set(element_liste_after['neighbors'])
        lost_distances = neighbor_distances_extraction(mot, list(voisins_absents), word_data_for_graph_before)
        noeud_abscent = {
            'word': mot,
            'lost_neighbors': list(voisins_absents),
            'lost_distances': lost_distances
        }
        lost_neighbors.append(noeud_abscent)

        voisins_apparut = set(element_liste_after['neighbors']) - set(element_liste_before['neighbors'])
        appeared_distances = neighbor_distances_extraction(mot, list(voisins_apparut), word_data_for_graph_after)
        noeud_apparut = {
            'word': mot,
            'appeared_neighbors': list(voisins_apparut),
            'appeared_distances': appeared_distances
        }
        appeared_neighbors.append(noeud_apparut)

        if 200 < j <= 210:
            print("les voisins conservés, apparut et disparrut du mots  \n", mot)
            print("mots voisins conservés après le plongement multilingue: \n", noeud_constant)
            print("mots voisins perdus apres le plongement multilingue: \n", noeud_abscent)
            print("mots voisins apparut après le plongement multilingue: \n", noeud_apparut)
        j += 1

    return common_neighbors, lost_neighbors, appeared_neighbors

def mean_GMUD(common_neighbors, lost_neighbors, appeared_neighbors, beta1, beta2, beta3):
    D = []
    somme = 0
    V_p = 0
    V_d = 0
    Apres_avant = 0
    for i in range(len(common_neighbors)):
        common_i = 0
        lost_i = 0
        appeared_i = 0
        D_i = 0
        for j in range(len(common_neighbors[i]['common_neighbors'])):
            common_i += beta1 * (common_neighbors[i]['after_PM_distances'][j] - common_neighbors[i]['before_PM_distances'][j])

        lost_i += beta2 * len(lost_neighbors[i]['lost_neighbors'])
        appeared_i += beta3 * len(appeared_neighbors[i]['appeared_neighbors'])

        D_i = common_i + lost_i + appeared_i
        V_p += appeared_i
        V_d += lost_i
        Apres_avant += common_i
        somme += D_i
        D.append(D_i)
    
    deviation_euclidian_GMUD = np.std(D)
    moy = somme / len(common_neighbors)
    print("la somme euclidian_GMUD est: \n", somme)
    print("la moyenne euclidian_GMUD est: \n", moy)
    print("L'écart type  euclidian_GMUD est: \n", deviation_euclidian_GMUD)

    return deviation_euclidian_GMUD, somme, moy, D, common_neighbors, V_p, V_d, Apres_avant

def Analyse(Tab_deformations, list_mots, mean_Gmud, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, Apres_avant):
    fichier_sortie = 'analyse_deformations.txt'
    with open(fichier_sortie, 'w') as file:
        file.write(f'-------------------------------- Analyse des deformations GMUD ------------------------------ \n')
        file.write(f'Métrique de distance: {args.distance_metric}\n')
        file.write(f'Déformation moyenne: {mean_Gmud}\n')
        file.write(f'Somme des Déformations: {sum_GMUD}\n')
        file.write(f'EcartType des Déformations: {deviation_euclidian_GMUD}, v_p: {V_p}, v_d: {V_d}, v_c: {Apres_avant}\n')
        file.write('Déformations par mot:\n')
        
        for i in range(len(Tab_deformations)):
            file.write(f'{list_mots[i]["word"]}: {Tab_deformations[i]}\n')

def parse_arguments():
    parser = argparse.ArgumentParser(description="GMUD: Méthode basée sur les graphes pour mesurer les déformations lors des plongements multilingues de mots")
    parser.add_argument("--distance_metric", default="euclidean", help="Métrique de distance à utiliser (cosine ou euclidean)")
    parser.add_argument("--nb_neighbors", type=int, default=101, help="Nombre de voisins à considérer")
    parser.add_argument("--beta1", type=float, default=0.4, help="Coefficient pour la déformation des voisins conservés")
    parser.add_argument("--beta2", type=float, default=0.3, help="Coefficient pour la déformation des voisins perdus")
    parser.add_argument("--beta3", type=float, default=0.3, help="Coefficient pour la déformation des voisins apparus")
    parser.add_argument("--embeddings_before", required=True, help="Chemin vers les embeddings source")
    parser.add_argument("--embeddings_after", required=True, help="Chemin vers les embeddings cible")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Calcul de la déformation moyenne GMUD")

    embeddings_before, words_before = reshap_embedding(args.embeddings_before, 300, 500)
    embeddings_after, words_after = reshap_embedding(args.embeddings_after, 300, 500)

    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    low_dim_embedding_before = tsne.fit_transform(embeddings_before)
    filename_target_visualisation_before = 'language_words_embeddings_before_PM'
    plot_with_labels(low_dim_embedding_before, words_before, filename_target_visualisation_before)

    low_dim_embedding_after = tsne.fit_transform(embeddings_after)
    filename_target_visualisation_after = 'language_words_embeddings_after_PM'
    plot_with_labels(low_dim_embedding_after, words_after, filename_target_visualisation_after)

    word_data_for_graph_before, word_data_for_graph_test_before = search_Word_Nearest_Neighbors_embedding(embeddings_before, words_before, args.distance_metric, args.nb_neighbors)
    print("--------------------- visualisation of Graph for language words embeddings neighbors before PM -------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_before, word_data_for_graph_test_before, before="before")

    fichier_sortie_avant = 'analyse_voisins_avant.txt'
    with open(fichier_sortie_avant, 'w') as file:
        for graph_data in word_data_for_graph_before:
            nearest_before = Ten_nearest_neighbor(graph_data['word'], word_data_for_graph_before)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins avant le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_before['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_before['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')

    word_data_for_graph_after, word_data_for_graph_test_after = search_Word_Nearest_Neighbors_embedding(embeddings_after, words_after, args.distance_metric, args.nb_neighbors)
    print("--------------------- visualisation of Graph for language words embeddings neighbors  after PM-------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_after, word_data_for_graph_test_after, before="after")

    fichier_sortie_apres = 'analyse_voisins_apres.txt'
    with open(fichier_sortie_apres, 'w') as file:
        for graph_data in word_data_for_graph_after:
            nearest_after = Ten_nearest_neighbor(graph_data['word'], word_data_for_graph_after)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins après le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_after['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_after['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')

    common_neighbors, lost_neighbors, appeared_neighbors = common_lost_appeared_neighbors_extraction(word_data_for_graph_before, word_data_for_graph_after)

    deviation_euclidian_GMUD, sum_GMUD, mean, Tab_deformations, list_mots, V_p, V_d, Apres_avant = mean_GMUD(common_neighbors, lost_neighbors, appeared_neighbors, args.beta1, args.beta2, args.beta3)

    Analyse(Tab_deformations, list_mots, mean, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, Apres_avant)

    print("La déformation moyenne avec la métrique GMUD est de:", mean)
    print("La somme des déformations avec la métrique GMUD est de:", sum_GMUD)
    print("L'écart-type de la déformation avec la métrique GMUD est de:", deviation_euclidian_GMUD)
