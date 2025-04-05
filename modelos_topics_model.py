# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:50:40 2025

@author: Andres
"""

# =============================
# INICIO: IMPORTACIÓN DE LIBRERÍAS
# =============================
print("Iniciando importación de librerías...")
import sklearn
import numpy
import pandas
import sys

import pandas as pd
import numpy as np
import unicodedata
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)

# =======================
# VERSIONES DE EJECUCIÓN NORMAL
# Python: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]
# scikit-learn: 1.5.1
# numpy: 1.26.4
# pandas: 2.2.2
# =======================

print("Librerías importadas correctamente.")
# =============================
# FIN: IMPORTACIÓN DE LIBRERÍAS
# =============================


# =============================
# INICIO: DESCARGA DE RECURSOS NLTK
# =============================
print("Descargando recursos de NLTK...")
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
print("Recursos NLTK descargados.")
# =============================
# FIN: DESCARGA DE RECURSOS NLTK
# =============================


# =============================
# INICIO: CARGA DEL DATASET
# =============================
print("Cargando archivo CSV...")
ruta_csv = 'predicciones.csv'
df = pd.read_csv(ruta_csv, encoding='utf8', sep=';')
print(f"Archivo cargado. Total de registros: {df.shape[0]}")
# =============================
# FIN: CARGA DEL DATASET
# =============================


# =============================
# INICIO: DEFINICIÓN DE STOPWORDS PERSONALIZADAS
# =============================
print("Definiendo stopwords personalizadas...")
stop_words = set(unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8')
                 for word in nltk.corpus.stopwords.words('spanish'))

lista_stopwords = ['muchisimas', 'muchas', 'gracias', 'eh', 'ah', 'okay',
                   'pues', 'javeriana', 'cali', 'javeriana cali', 'alo',
                   'senora', 'senor', 'entonces', 'usted','queria','maria']

for word in lista_stopwords:
    stop_words.add(word)

print(f"Total de stopwords utilizadas: {len(stop_words)}")
# =============================
# FIN: DEFINICIÓN DE STOPWORDS PERSONALIZADAS
# =============================


# =============================
# INICIO: PREPROCESAMIENTO DE TEXTO
# =============================
print("Iniciando preprocesamiento de texto...")

wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()
papers = df['Text'].dropna().tolist()

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def normalize_corpus(papers):
    norm_papers = []
    for paper in papers:
        paper = paper.lower()
        paper = remove_accents(paper)
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        if paper_tokens:
            norm_papers.append(paper_tokens)
    return norm_papers

norm_papers = normalize_corpus(papers)
print(f"Preprocesamiento finalizado. Total de documentos procesados: {len(norm_papers)}")
print(f"Ejemplo de documento tokenizado: {norm_papers[0][:10]}")
# =============================
# FIN: PREPROCESAMIENTO DE TEXTO
# =============================


# =============================
# INICIO: VECTORIZACIÓN
# =============================
print("Iniciando vectorización con CountVectorizer...")
cv = CountVectorizer(min_df=200, max_df=0.6, ngram_range=(1, 2),
                     token_pattern=None, tokenizer=lambda doc: doc,
                     preprocessor=lambda doc: doc)

cv_features = cv.fit_transform(norm_papers)
print(f"Vectorización completada. Forma de la matriz: {cv_features.shape}")
# =============================
# FIN: VECTORIZACIÓN
# =============================


# =============================
# INICIO: MODELADO NMF
# =============================
print("Iniciando modelado NMF...")

TOTAL_TOPICS = 5
nmf_model = NMF(n_components=TOTAL_TOPICS, solver='cd', max_iter=500,
                random_state=42, alpha_W=0.1, alpha_H=0.1, l1_ratio=0.85)

document_topics = nmf_model.fit_transform(cv_features)
print("Modelado NMF completado.")
# =============================
# FIN: MODELADO NMF
# =============================


# =============================
# INICIO: TÉRMINOS CLAVE POR TEMA
# =============================
print("Extrayendo términos clave por tema...")

top_terms = 20
vocabulary = np.array(cv.get_feature_names_out())
topic_terms = nmf_model.components_
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
topic_keyterms = vocabulary[topic_key_term_idxs]
topics = [', '.join(topic) for topic in topic_keyterms]

pd.set_option('display.max_colwidth', 1)
topics_df = pd.DataFrame(topics,
                         columns=['Terms per Topic'],
                         index=['Topic' + str(t) for t in range(1, TOTAL_TOPICS + 1)])

print("Términos clave extraídos exitosamente:")
print(topics_df)
# =============================
# FIN: TÉRMINOS CLAVE POR TEMA
# =============================
