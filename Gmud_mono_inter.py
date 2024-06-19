from __future__ import print_function
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(vec1, vec2):
    # Reshape vectors to 2D if necessary
    if len(vec1.shape) == 1:
        vec1 = vec1.reshape(1, -1)
    if len(vec2.shape) == 1:
        vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Cette fonction charge les embeddings de mots à partir d'un fichier, les stocke dans un format approprié et renvoie à la fois les embeddings et les mots associés.
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
    
    # Calculer les distances entre les embeddings de la langue A et de la langue B
def search_Word_Nearest_Neighbors_embedding_lang_A_B(embeddings_langA_after, embeddings_langB_after, word_list_langA, word_list_langB, distance, k, start=200, nb_words=210):
    # Initialiser l'algorithme des k plus proches voisins avec la métrique cosinus
    neighbors_algorithm = NearestNeighbors(n_neighbors=k, metric=distance)
    neighbors_algorithm.fit(embeddings_langB_after)

    word_data_for_graph_test = []
    word_data_for_graph = []
    j = 0

    for i, word in enumerate(word_list_langA):
        # Obtenir les k plus proches voisins pour le vecteur de la langue A
        distances_i, indices_i = neighbors_algorithm.kneighbors([embeddings_langA_after[i]], return_distance=True)

        neighbor_words = [word_list_langB[idx] for idx in indices_i[0]]
        neighbor_distances = distances_i[0]

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
    print("je construis l'objet word_data_for_graph_lang_A_B à envoyer au graphe")
    print(word_data_for_graph)

    return word_data_for_graph, word_data_for_graph_test


# Cette fonction permet de dessiner le graphe d'un mot avec ses voisins pondéré par la distance entre le mot et ses différents voisins
def Graph_for_word_embedding_neighbor(word_data_for_graph_test, state):
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

        if(state == "before"):
        # Dessiner le graphe avec des nœuds noirs et des écritures blanches
            nx.draw(G_test, pos, with_labels=True, node_size=2000, font_size=10,
                    node_color='black', font_color='white')
        else:
        # Dessiner le graphe avec des nœuds bleu et des écritures noires
            nx.draw(G_test, pos, with_labels=True, node_size=2000, font_size=10,
                    node_color='blue', font_color='white')   
                    
        edge_labels = nx.get_edge_attributes(G_test, 'weight')
        nx.draw_networkx_edge_labels(G_test, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"Graph for {word} and its Neighbors")
        filename = f"graphe_{word}_et_ses_voisins_{state}.png"
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
# dans le contexte de GMUD_mono
def common_lost_appeared_neighbors_extraction(word_data_for_graph_before, word_data_for_graph_after, embeddings_source_before, embeddings_source_after, word_list):
    word_to_index = {word: idx for idx, word in enumerate(word_list)}

    common_neighbors = []
    lost_neighbors = []
    appeared_neighbors = []
    j = 0

    for mot in word_to_index.keys() & word_to_index.keys():
        element_liste_before = next(item for item in word_data_for_graph_before if item['word'] == mot)
        element_liste_after = next(item for item in word_data_for_graph_after if item['word'] == mot)

        # Voisins communs
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

        # Voisins perdus
        voisins_absents = set(element_liste_before['neighbors']) - set(element_liste_after['neighbors'])
        lost_distances_before = neighbor_distances_extraction(mot, list(voisins_absents), word_data_for_graph_before)
        lost_distances_after = []
        for neighbor in voisins_absents:
            distance_after = 1 - calculate_cosine_similarity(embeddings_source_after[word_to_index[mot]], embeddings_source_after[word_to_index[neighbor]])
            lost_distances_after.append(distance_after)
        
        noeud_abscent = {
            'word': mot,
            'lost_neighbors': list(voisins_absents),
            'lost_distances_before': lost_distances_before,
            'lost_distances_after': lost_distances_after
        }
        lost_neighbors.append(noeud_abscent)

        # Voisins apparus
        voisins_apparut = set(element_liste_after['neighbors']) - set(element_liste_before['neighbors'])
        appeared_distances_after = neighbor_distances_extraction(mot, list(voisins_apparut), word_data_for_graph_after)
        appeared_distances_before = []
        for neighbor in voisins_apparut:
            distance_before = 1 - calculate_cosine_similarity(embeddings_source_before[word_to_index[mot]], embeddings_source_before[word_to_index[neighbor]])
            appeared_distances_before.append(distance_before)
        
        noeud_apparut = {
            'word': mot,
            'appeared_neighbors': list(voisins_apparut),
            'appeared_distances_before': appeared_distances_before,
            'appeared_distances_after': appeared_distances_after
        }
        appeared_neighbors.append(noeud_apparut)

        if 200 < j <= 210:
            print("les voisins conservés, apparut et disparrut du mot  \n", mot)
            print("mots voisins conservés après le plongement multilingue: \n", noeud_constant)
            print("mots voisins perdus apres le plongement multilingue: \n", noeud_abscent)
            print("mots voisins apparut après le plongement multilingue: \n", noeud_apparut)
        j += 1

    return common_neighbors, lost_neighbors, appeared_neighbors


def mean_GMUD_mono(common_neighbors, lost_neighbors, appeared_neighbors, beta1, beta2, beta3):
    D = []
    somme = 0
    V_p = 0
    V_d = 0
    v_c = 0

    for i in range(len(common_neighbors)):
        common_i = 0
        lost_i = 0
        appeared_i = 0
        D_i = 0

        # Calculer les différences pour les voisins communs
        for j in range(len(common_neighbors[i]['common_neighbors'])):
            common_i += beta1 * (common_neighbors[i]['after_PM_distances'][j] - common_neighbors[i]['before_PM_distances'][j])

        # Calculer les différences pour les voisins perdus
        for j in range(len(lost_neighbors[i]['lost_neighbors'])):
            distance_before = lost_neighbors[i]['lost_distances_before'][j]
            distance_after = lost_neighbors[i]['lost_distances_after'][j]
            lost_i += beta2 * (distance_after - distance_before)

        # Calculer les différences pour les voisins apparus
        for j in range(len(appeared_neighbors[i]['appeared_neighbors'])):
            distance_before = appeared_neighbors[i]['appeared_distances_before'][j]
            distance_after = appeared_neighbors[i]['appeared_distances_after'][j]
            appeared_i += beta3 * (distance_after - distance_before)

        D_i = common_i + lost_i + appeared_i
        V_p += appeared_i
        V_d += lost_i
        v_c += common_i
        somme += D_i
        D.append(D_i)

    deviation_euclidian_GMUD = np.std(D)
    moy = somme / len(common_neighbors)
    print("la somme euclidian_GMUD est: \n", somme)
    print("la moyenne euclidian_GMUD est: \n", moy)
    print("L'écart type  euclidian_GMUD est: \n", deviation_euclidian_GMUD)

    return deviation_euclidian_GMUD, somme, moy, D, common_neighbors, V_p, V_d, v_c

def mean_GMUD_inter(word_data_for_graph_lang_A_B, k, Dmax=2.0):
    D = []
    somme = 0
    V_p = 0  # Total des distances apparues
    V_d = 0  # Total des distances disparues
    v_c = 0  # Total des distances communes

    for i, data in enumerate(word_data_for_graph_lang_A_B):
        common_i = 0
        D_i = 0

        # Calculer les déformations pour les voisins
        for j in range(min(k, len(data['distances']))):
            common_i += (Dmax - data['distances'][j])

        D_i = common_i
        v_c += common_i
        somme += D_i
        D.append(D_i)

    deviation_euclidian_GMUD = np.std(D)
    moy = somme / len(word_data_for_graph_lang_A_B)
    print("La somme GMUD_inter est: \n", somme)
    print("La moyenne GMUD_inter est: \n", moy)
    print("L'écart type GMUD_inter est: \n", deviation_euclidian_GMUD)

    return deviation_euclidian_GMUD, somme, moy, D, word_data_for_graph_lang_A_B, V_p, V_d, v_c

def Analyse(Tab_deformations, list_mots, mean_GMUD_mono, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, v_c):
    fichier_sortie = 'analyse_deformations.txt'
    with open(fichier_sortie, 'w') as file:
        file.write(f'-------------------------------- Analyse des deformations GMUD ------------------------------ \n')
        file.write(f'Métrique de distance: {args.distance_metric}\n')
        file.write(f'Déformation moyenne: {mean_GMUD_mono}\n')
        file.write(f'Somme des Déformations: {sum_GMUD}\n')
        file.write(f'EcartType des Déformations: {deviation_euclidian_GMUD}, v_p: {V_p}, v_d: {V_d}, v_c: {v_c}\n')
        file.write('Déformations par mot:\n')
        
        for i in range(len(Tab_deformations)):
            file.write(f'{list_mots[i]["word"]}: {Tab_deformations[i]}\n')

def Analyse_GMUD_inter(Tab_deformations, list_mots, mean_GMUD_inter, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, v_c):
    fichier_sortie = 'analyse_deformations_inter.txt'
    with open(fichier_sortie, 'w') as file:
        file.write(f'-------------------------------- Analyse des deformations GMUD_inter ------------------------------ \n')
        file.write(f'Métrique de distance: cosine\n')  # Supposons que la distance est cosinus pour cet exemple
        file.write(f'Déformation moyenne: {mean_GMUD_inter}\n')
        file.write(f'Somme des Déformations: {sum_GMUD}\n')
        file.write(f'EcartType des Déformations: {deviation_euclidian_GMUD}, v_p: {V_p}, v_d: {V_d}, v_c: {v_c}\n')
        file.write('Déformations par mot:\n')
        
        for i in range(len(Tab_deformations)):
            file.write(f'{list_mots[i]["word"]}: {Tab_deformations[i]}\n')

def parse_arguments():
    parser = argparse.ArgumentParser(description="GMUD: Méthode basée sur les graphes pour mesurer les déformations lors des plongements multilingues de mots")
    parser.add_argument("--distance_metric", default="euclidean", help="Métrique de distance à utiliser (cosine ou euclidean)")
    parser.add_argument("--nb_neighbors", type=int, default=101, help="Nombre de voisins à considérer")
    parser.add_argument("--beta1", type=float, default=1.0, help="Coefficient pour la déformation des voisins conservés")
    parser.add_argument("--beta2", type=float, default=1.0, help="Coefficient pour la déformation des voisins perdus")
    parser.add_argument("--beta3", type=float, default=1.0, help="Coefficient pour la déformation des voisins apparus")
    parser.add_argument("--embeddings_source_before", required=True, help="Chemin vers les embeddings source avant le PM")
    parser.add_argument("--embeddings_source_after", required=True, help="Chemin vers les embeddings source après le PM")
    parser.add_argument("--embeddings_target_after", required=True, help="Chemin vers les embeddings cible après le PM")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Calcul de la déformation moyenne GMUD")

    # Charger les embeddings avant et après
    embeddings_source_before, words_source_before = reshap_embedding(args.embeddings_source_before, 300)
    embeddings_source_after, words_source_after = reshap_embedding(args.embeddings_source_after, 300)

    # Créer la visualisation TSNE pour les embeddings avant
    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    low_dim_embedding_before = tsne.fit_transform(embeddings_source_before)
    filename_target_visualisation_before = 'language_words_embeddings_source_before_PM'
    plot_with_labels(low_dim_embedding_before, words_source_before, filename_target_visualisation_before)

    # Créer la visualisation TSNE pour les embeddings après
    low_dim_embedding_after = tsne.fit_transform(embeddings_source_after)
    filename_target_visualisation_after = 'language_words_embeddings_source_after_PM'
    plot_with_labels(low_dim_embedding_after, words_source_after, filename_target_visualisation_after)

    # Trouver les voisins les plus proches avant le processus
    word_data_for_graph_before, word_data_for_graph_test_before = search_Word_Nearest_Neighbors_embedding(embeddings_source_before, words_source_before, args.distance_metric, args.nb_neighbors)
    print("--------------------- Visualisation of Graph for language words embeddings neighbors before PM -------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_test_before, state="before")

    # Sauvegarder l'analyse des voisins avant le processus
    fichier_sortie_avant = 'analyse_voisins_avant.txt'
    with open(fichier_sortie_avant, 'w') as file:
        for graph_data in word_data_for_graph_before:
            nearest_before = Ten_nearest_neighbor(graph_data['word'], word_data_for_graph_before)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins avant le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_before['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_before['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')

    # Trouver les voisins les plus proches après le processus
    word_data_for_graph_after, word_data_for_graph_test_after = search_Word_Nearest_Neighbors_embedding(embeddings_source_after, words_source_after, args.distance_metric, args.nb_neighbors)
    print("--------------------- Visualisation of Graph for language words embeddings neighbors after PM -------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_test_after, state="after")

    # Sauvegarder l'analyse des voisins après le processus
    fichier_sortie_apres = 'analyse_voisins_apres.txt'
    with open(fichier_sortie_apres, 'w') as file:
        for graph_data in word_data_for_graph_after:
            nearest_after = Ten_nearest_neighbor(graph_data['word'], word_data_for_graph_after)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins après le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_after['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_after['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')


    # Extraire les voisins communs, perdus et apparus
    common_neighbors, lost_neighbors, appeared_neighbors = common_lost_appeared_neighbors_extraction(
        word_data_for_graph_before, word_data_for_graph_after, embeddings_source_before, embeddings_source_after, words_source_before)

    # Calculer la déformation moyenne GMUD
    deviation_euclidian_GMUD, sum_GMUD, mean, Tab_deformations, list_mots, V_p, V_d, v_c = mean_GMUD_mono(
        common_neighbors, lost_neighbors, appeared_neighbors, args.beta1, args.beta2, args.beta3)

    # Analyser et sauvegarder les résultats
    Analyse(Tab_deformations, list_mots, mean, sum_GMUD, deviation_euclidian_GMUD, V_p, V_d, v_c)

    print("La déformation moyenne avec la métrique GMUD est de:", mean)
    print("La somme des déformations avec la métrique GMUD est de:", sum_GMUD)
    print("L'écart-type de la déformation avec la métrique GMUD est de:", deviation_euclidian_GMUD)

    embeddings_target_after, words_target_after = reshap_embedding(args.embeddings_target_after, 300)
    word_data_for_graph_A_B, word_data_for_graph_test_A_B = search_Word_Nearest_Neighbors_embedding_lang_A_B(
    embeddings_source_after, embeddings_target_after, words_source_after, words_target_after, args.distance_metric, args.nb_neighbors)

    print("--------------------- Visualisation of Graph for words embeddings source with its target neighbors after PM -------------------------\n")
    Graph_for_word_embedding_neighbor(word_data_for_graph_test_A_B, state = "source_target_after")
        # Sauvegarder l'analyse des voisins target de chaque mot source après le processus
    fichier_sortie_apres = 'analyse_voisins_source_target_after.txt'
    with open(fichier_sortie_apres, 'w') as file:
        for graph_data in word_data_for_graph_A_B:
            nearest_after = Ten_nearest_neighbor(graph_data['word'], word_data_for_graph_A_B)
            central_word = graph_data['word']
            file.write(f'-------------------------------- Analyse des voisins source_target_after après le plongement ------------------------------ \n')
            voisins = " ".join(voisin for voisin in nearest_after['neighbor'])
            distances = " ".join(str(distance) for distance in nearest_after['distances'])
            file.write(f'mot central: {central_word}, voisins_proches: {voisins}, distances: {distances}\n')

    # sauvegarder resultat Gmud_inter
    deviation_euclidian_GMUD, somme, moy_inter, D, word_data_for_graph_lang_A_B, V_p, V_d, v_c = mean_GMUD_inter(word_data_for_graph_A_B, args.nb_neighbors)
    Analyse_GMUD_inter(D, word_data_for_graph_lang_A_B, moy_inter, somme, deviation_euclidian_GMUD, V_p, V_d, v_c)
    print("ratio mono/inter est: ", mean/moy_inter)
