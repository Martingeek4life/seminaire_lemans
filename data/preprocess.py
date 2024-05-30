import subprocess
import argparse
import string
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        rawdata = file.read()
        result = chardet.detect(rawdata)
        return result['encoding']

def preprocess_text(corpus_path, output_file):
    print("enter")
    encodage = detect_encoding(corpus_path)
    with open(corpus_path, 'r', encoding=encodage) as file:
        text = file.read().lower()
        print(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        print(text)
    with open(output_file, 'w', encoding='utf-8') as clean_file:
        print("clean")
        clean_file.write(text)

preprocess_text("./testament_en.txt", "./new_testament_en.txt")