# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# =======================
# IMPORTACIÓN DE LIBRERÍAS
# =======================

import sklearn
import numpy
import scipy
import pandas
import sys


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from importlib.metadata import version

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


from gensim.models import Word2Vec
from sklearn.preprocessing import label_binarize

from lime.lime_text import LimeTextExplainer

from IPython.display import display, HTML


# =======================
# DESCARGA DE RECURSOS NLTK
# =======================

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# =======================
# DICCIONARIO DE JERGAS
# =======================

jerga_dict_es = {
    "pa'": "para",
    "pana": "amigo"
}
# =======================
# VISUALIZACIÓN DE VERSIONES
# =======================

print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("lime:", version("lime"))
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)


# =======================
# VERSIONES DE EJECUCIÓN NORMAL
# Python: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]
# scikit-learn: 1.5.1
# lime: 0.2.0.1
# numpy: 1.26.4
# scipy: 1.13.1
# pandas: 2.2.2
# =======================


# =======================
# CLASES DE PREPROCESAMIENTO
# =======================

class ReemplazarJergas(BaseEstimator, TransformerMixin):
    def __init__(self, diccionario=jerga_dict_es):
        self.diccionario = diccionario

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._reemplazar(texto) for texto in X]

    def _reemplazar(self, texto):
        palabras = texto.split()
        palabras_reemplazadas = [self.diccionario.get(p, p) for p in palabras]
        return ' '.join(palabras_reemplazadas)

class LimpiarTexto(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._limpiar(texto) for texto in X]

    def _limpiar(self, texto):
        texto = texto.lower()
        texto = self._quitar_acentos(texto)
        texto = re.sub(r'[^a-zñ\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def _quitar_acentos(self, texto):
        acentos_dict = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N'
        }
        for acentuada, sin_acento in acentos_dict.items():
            texto = texto.replace(acentuada, sin_acento)
        return texto

class TokenizarYLematizar(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [' '.join(self._tokenizar_lemmatizar(texto)) for texto in X]

    def _tokenizar_lemmatizar(self, texto):
        tokens = word_tokenize(texto)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, X, y=None):
        tokenized_texts = [texto.split() for texto in X]
        self.model = Word2Vec(sentences=tokenized_texts, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4)
        return self

    def transform(self, X):
        tokenized_texts = [texto.split() for texto in X]
        vectors = []
        for tokens in tokenized_texts:
            if not tokens:
                vectors.append(np.zeros(self.vector_size))
            else:
                vec = np.mean([self.model.wv[token] for token in tokens if token in self.model.wv], axis=0)
                vectors.append(vec)
        return np.array(vectors)

# =======================
# PIPELINE DE PREPROCESAMIENTO
# =======================

print("INICIA PREPROCESAMIENTO")

preprocessing_pipeline = Pipeline([
    ('reemplazar_jergas', ReemplazarJergas()),
    ('limpiar_texto', LimpiarTexto()),
    ('tokenizar_lematizar', TokenizarYLematizar()),
])

# =======================
# CARGA DE DATOS
# =======================

ruta_csv = '3.csv'
df = pd.read_csv(ruta_csv, encoding='latin1', sep=';')

X = df['Text']
y = df['Etiqueta']

# =======================
# VECTORIZACIÓN Y CODIFICACIÓN
# =======================

vectorizer = CountVectorizer()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_procesado = preprocessing_pipeline.fit_transform(X)
X_vectorized = vectorizer.fit_transform(X_procesado)
X_ohe = encoder.fit_transform(X_vectorized.toarray())
print("FIN PREPROCESAMIENTO OK")

# =======================
# ENTRENAMIENTO DEL MODELO
# =======================

print("INICIA ENTRENAMIENTO DEL MODELO")

X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

f1 = f1_score(y_test, y_pred, average=None)
for etiqueta, valor in zip(np.unique(y_test), f1):
    print(f"Etiqueta: SVM {etiqueta}, F1-score: {valor}")

print("FIN ENTRENAMIENTO DEL MODELO OK")
# =======================
# BÚSQUEDA DE HIPERPARÁMETROS
# =======================

print("INICIA BÚSQUEDA DE HIPERPARÁMETROS DEL MODELO")
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf'],
    'probability': [True]
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")

svm_model_best = grid_search.best_estimator_
svm_model_best.fit(X_train, y_train)


print("FIN BÚSQUEDA DE HIPERPARÁMETROS DEL MODELO OK")
# =======================
# CURVAS ROC
# =======================

print("INICIA CURVA DE ROC")
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_proba = svm_model_best.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(y_test_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para cada clase')
plt.legend(loc='lower right')
plt.show()

print("FIN CURVA DE ROC OK")
# =======================
# ETIQUETADO DE NUEVOS DATOS
# =======================


print("INICIA ETIQUETADO DE LOS DATOS")
ruta_csv_se = 'sin_etiquetas_all.csv'
df_se = pd.read_csv(ruta_csv_se, encoding='latin1', sep=';')

X_se = df_se['Text']
X_procesado_se = preprocessing_pipeline.transform(X_se)
X_vectorized_se = vectorizer.transform(X_procesado_se)
X_ohe_se = encoder.transform(X_vectorized_se.toarray())

y_pred_se = svm_model_best.predict(X_ohe_se)

df_predicciones = pd.DataFrame({
    'Text': X_se,
    'Prediccion': y_pred_se
})

df_predicciones.to_csv('predicciones.csv', index=False, sep=';', encoding='utf-8')


print("FIN ETIQUETADO DE LOS DATOS OK")

print("INICIA EVALUACIÓN GENERAL DEL MODELO")


X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

X_test2_vec = vectorizer.transform(X_test2)

X_test2_ohe = encoder.transform(X_test2_vec.toarray())

y_pred = svm_model_best.predict(X_test2_ohe)

cm = confusion_matrix(y_test2, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model_best.classes_)
disp.plot(cmap='Blues')

print(classification_report(y_test2, y_pred, digits=4))

accuracy = accuracy_score(y_test2, y_pred)
print("Accuracy:", accuracy)

print("FIN EVALUACIÓN GENERAL DEL MODELO OK")



print("INICIA EXPLICABILIDAD CON LIME")
# =======================
# EXPLICABILIDAD CON LIME
# =======================

def predict_fn(texts):
    X_vec = vectorizer.transform(texts)
    X_ohe = encoder.transform(X_vec.toarray())
    return svm_model_best.predict_proba(X_ohe)

idx = 0
text_instance = X_test2.iloc[idx]

explainer = LimeTextExplainer(class_names=np.unique(y).tolist())
explanation = explainer.explain_instance(text_instance, predict_fn, num_features=10)

# Mostrar resultados
explanation.show_in_notebook()
html_explanation = explanation.as_html()
display(HTML(html_explanation))

# Si también quieres imprimir los pesos (por consola)
print(explanation.as_list())

with open("explicacion_lime.html", "w", encoding="utf-8") as f:
    f.write(html_explanation)
print("FIN EXPLICABILIDAD CON LIME (SE GUARDA ARCHIVO EN EL DIRECTORIO) OK")    
