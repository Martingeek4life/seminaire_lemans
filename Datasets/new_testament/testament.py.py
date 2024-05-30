import os
import shutil

def concatenate_files(base_directory, output_file):
    # Ouvrir le fichier de sortie en mode écriture
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Construire le chemin complet vers le fichier
        for file in base_directory:
            with open(file + '_en.txt', 'r', encoding='utf-8') as readfile:
                # Copier le contenu du fichier source dans le fichier de sortie
                shutil.copyfileobj(readfile, outfile)
                # Ajouter une nouvelle ligne pour séparer le contenu des différents fichiers
                outfile.write('\n')



testament = ['40_matiu', '41_maaku', '42_luuku', '43_johanu', '44_ise_awon_apositeli', '45_roomu', '46_1_korinti', '47_2_korinti', '48_galatia', '49_efesu', '50_filipi', '51_kolose', '52_1_tesalonika', '53_2_tesalonika', '54_1_timotiu', '55_2_timotiu', 
'56_titu', '57_filimoni', '58_awon_heberu', '59_jakobu', '60_1_peteru', '61_2_peteru', '62_1_johanu', '63_2_johanu', '64_3_johanu', '65_juda', '66_ifihan']
# Spécifier le chemin du dossier de base et le chemin du fichier de sortie
output_file = 'testament_en.txt'

# Appeler la fonction
concatenate_files(testament, output_file)

""" for livre in testament:

    input_directory = livre
    output_file_text = livre + '_yo.txt'
    output_file_text_en = livre + '_en.txt' """