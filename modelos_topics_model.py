# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:50:40 2025

@author: Andres
"""

# =============================
# INICIO: IMPORTACIÓN DE LIBRERÍAS
# =============================
print("\n=== Iniciando importación de librerías... ===")

import sys
import unicodedata
import pandas as pd
import numpy as np
import nltk
import sklearn
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

print("=== Librerías importadas correctamente ===")
# =============================
# FIN: IMPORTACIÓN DE LIBRERÍAS
# =============================


# =============================
# INICIO: DEFINICIÓN DE FUNCIONES
# =============================

def evaluar_modelo_topics(document_topics, total_topics):
    """
    Evalúa el modelo de tópicos basado en:
    - Dominancia promedio de los documentos
    - Distribución de documentos por tema
    - Gráfico de distribución de documentos
    """
    print("\n===== EVALUACIÓN DEL MODELO =====")

    if document_topics.size == 0 or np.all(document_topics == 0):
        print("⚠️ Advertencia: No hay documentos procesados o la matriz de tópicos es vacía.")
        return

    # 1. Dominancia de tópicos
    dominancia = (document_topics.max(axis=1) / document_topics.sum(axis=1)).mean()
    print(f"Dominancia promedio de temas: {dominancia:.4f}")

    if dominancia >= 0.7:
        print("✔️ Buena dominancia: los documentos tienden a pertenecer claramente a un solo tema.")
    else:
        print("⚠️ Baja dominancia: documentos mezclados en varios temas.")

    # 2. Distribución de temas
    tema_dominante = document_topics.argmax(axis=1)
    tema_counts = np.bincount(tema_dominante, minlength=total_topics)

    print("\nDistribución de documentos por tema:")
    for idx, count in enumerate(tema_counts):
        print(f" - Tema {idx+1}: {count} documentos")

    # 3. Gráfico de distribución
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, total_topics + 1), tema_counts)
    plt.xlabel('Tema dominante')
    plt.ylabel('Cantidad de documentos')
    plt.title('Distribución de documentos por tema')
    plt.xticks(range(1, total_topics + 1))
    plt.grid(True)
    plt.show()

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def normalize_corpus(papers, stop_words, wtk, wnl):
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

# =============================
# FIN: DEFINICIÓN DE FUNCIONES
# =============================


# =============================
# INICIO: CONFIGURACIÓN Y DATOS
# =============================
print("\n=== Descargando recursos de NLTK... ===")
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
print("=== Recursos de NLTK descargados ===")

print("\n=== Cargando archivo CSV... ===")
ruta_csv = 'predicciones.csv'
df = pd.read_csv(ruta_csv, encoding='utf8', sep=';')
print(f"Archivo cargado. Total de registros: {df.shape[0]}")

print("\n=== Definiendo stopwords personalizadas... ===")
stop_words = set(unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8')
                 for word in nltk.corpus.stopwords.words('spanish'))

lista_stopwords = [
    'muchisimas', 'muchas', 'gracias', 'eh', 'ah', 'okay',
    'pues', 'javeriana', 'cali', 'javeriana cali', 'alo',
    'senora', 'senor', 'entonces', 'usted', 'queria', 'maria'
]

for word in lista_stopwords:
    stop_words.add(word)

print(f"Total de stopwords utilizadas: {len(stop_words)}")
# =============================
# FIN: CONFIGURACIÓN Y DATOS
# =============================


# =============================
# INICIO: PROCESAMIENTO PRINCIPAL
# =============================
print("\n=== Iniciando procesamiento de sentimientos... ===")

wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()
lista_sentimientos = ['Positivo', 'Negativo', 'Neutral', 'Todas']

for prediccion in lista_sentimientos:
    
    print(f"\n********** SENTIMIENTO: {prediccion} **********")
    
    if prediccion == 'Todas':
        papers = df['Text'].dropna().tolist()
    else:
        papers = df[df['Prediccion'] == prediccion]['Text'].dropna().tolist()

    # Configuración manual según sentimiento
    if prediccion == 'Negativo':
        TOTAL_TOPICS = 2
        val_min_df = 60
        val_max_df = 0.7
    elif prediccion == 'Neutral':
        TOTAL_TOPICS = 2
        val_min_df = 10
        val_max_df = 0.8
    elif prediccion == 'Positivo':
        TOTAL_TOPICS = 3
        val_min_df = 40
        val_max_df = 0.6
    else:  # 'Todas'
        TOTAL_TOPICS = 4
        val_min_df = 180
        val_max_df = 0.6

    print(f"Configuración -> TOTAL_TOPICS: {TOTAL_TOPICS}, min_df: {val_min_df}, max_df: {val_max_df}")

    # =============================
    # INICIO: PREPROCESAMIENTO
    # =============================
    print("\nIniciando preprocesamiento de texto...")
    norm_papers = normalize_corpus(papers, stop_words, wtk, wnl)
    print(f"Preprocesamiento finalizado. Total de documentos procesados: {len(norm_papers)}")
    if norm_papers:
        print(f"Ejemplo de documento tokenizado: {norm_papers[0][:10]}")
    else:
        print("⚠️ No hay documentos procesados.")
    # =============================
    # FIN: PREPROCESAMIENTO
    # =============================

    if not norm_papers:
        continue

    # =============================
    # INICIO: VECTORIZACIÓN
    # =============================
    print("\nIniciando vectorización con CountVectorizer...")
    cv = CountVectorizer(
        min_df=val_min_df,
        max_df=val_max_df,
        ngram_range=(1, 2),
        token_pattern=None,
        tokenizer=lambda doc: doc,
        preprocessor=lambda doc: doc
    )
    
    cv_features = cv.fit_transform(norm_papers)
    print(f"Vectorización completada. Forma de la matriz: {cv_features.shape}")
    # =============================
    # FIN: VECTORIZACIÓN
    # =============================

    # =============================
    # INICIO: MODELADO NMF
    # =============================
    print("\nIniciando modelado NMF...")
    nmf_model = NMF(
        n_components=TOTAL_TOPICS,
        solver='cd',
        max_iter=500,
        random_state=42,
        alpha_W=0.1,
        alpha_H=0.1,
        l1_ratio=0.85
    )
    
    document_topics = nmf_model.fit_transform(cv_features)
    print("Modelado NMF completado.")
    # =============================
    # FIN: MODELADO NMF
    # =============================

    # Evaluar el modelo
    evaluar_modelo_topics(document_topics, TOTAL_TOPICS)

    # =============================
    # INICIO: EXTRACCIÓN DE TÉRMINOS CLAVE
    # =============================
    print("\nExtrayendo términos clave por tema...")
    
    top_terms = 20
    vocabulary = np.array(cv.get_feature_names_out())
    topic_terms = nmf_model.components_
    topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
    topic_keyterms = vocabulary[topic_key_term_idxs]
    topics = [', '.join(topic) for topic in topic_keyterms]

    pd.set_option('display.max_colwidth', 1)
    topics_df = pd.DataFrame(
        topics,
        columns=['Terms per Topic'],
        index=[f'Topic {t+1}' for t in range(TOTAL_TOPICS)]
    )

    print("Términos clave extraídos exitosamente:")
    print(topics_df)
    # =============================
    # FIN: EXTRACCIÓN DE TÉRMINOS CLAVE
    # =============================

print("\n=== PROCESAMIENTO FINALIZADO ===")
# =============================
# FIN: PROCESAMIENTO PRINCIPAL
# =============================
