import os

def extract_text_pairs(input_directory, output_file_text, output_file_text_en):
    # Parcourir tous les fichiers dans le dossier spécifié
    for filename in os.listdir(input_directory):
        if filename.endswith(".conllu"):  # Assurez-vous de traiter uniquement les fichiers texte
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            with open(output_file_text, 'a', encoding='utf-8') as out_text, \
                 open(output_file_text_en, 'a', encoding='utf-8') as out_text_en:
                for i in range(len(lines)):
                    if lines[i].startswith('# text ='):
                        out_text.write(lines[i][8:])  # Écrire le contenu après '# text ='
                    elif lines[i].startswith('# text_en ='):
                        out_text_en.write(lines[i][10:])  # Écrire le contenu après '# text_en ='

# Spécifier le dossier contenant les fichiers, et les chemins des fichiers de sortie

testament = ['40_matiu', '41_maaku', '42_luuku', '43_johanu', '44_ise_awon_apositeli', '45_roomu', '46_1_korinti', '47_2_korinti', '48_galatia', '49_efesu', '50_filipi', '51_kolose', '52_1_tesalonika', '53_2_tesalonika', '54_1_timotiu', '55_2_timotiu', 
'56_titu', '57_filimoni', '58_awon_heberu', '59_jakobu', '60_1_peteru', '61_2_peteru', '62_1_johanu', '63_2_johanu', '64_3_johanu', '65_juda', '66_ifihan']

for livre in testament:

    input_directory = livre
    output_file_text = livre + '_yo.txt'
    output_file_text_en = livre + '_en.txt'

    # Appeler la fonction
    extract_text_pairs(input_directory, output_file_text, output_file_text_en)