import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score

#Cargamos el dataset
dataset = pd.read_csv('train.csv', encoding='latin-1', sep=',')

# Como no hay nombres de columnas asignadas, se asignan los nombres de las columnas
dataset.columns = ['clasificacion', 'id', 'fecha', 'NO_QUERY', 'usuario', 'tweet']

# Se eliminan las columnas que no se utilizarán
dataset = dataset.drop(columns=['clasificacion','id','fecha', 'NO_QUERY', 'usuario'])

# Se eliminan los registros duplicados
dataset = dataset.drop_duplicates()

#Mostramos las primeras filas del dataset
print(dataset.head(20), '\n')

# Creamos una función para clasificar los tweets según su contenido
def clasificar_tweet(tweet):
    if 'upset' in tweet.lower():
        return 'Negativo'
    elif 'happy' in tweet.lower():
        return 'Positivo'
    else:
        return 'Neutral'

# Aplicamos la función a la columna de tweets
dataset['clasificacion'] = dataset['tweet'].apply(clasificar_tweet)

# Mostramos las primeras filas del dataset
print(dataset.head(20), '\n')

#Vamos a limpiar los datos que no necesitamos por ejemplo las menciones, los acentos, etc
#Para ello vamos a utilizar expresiones regulares
import re

#Función para limpiar los mensaje
def eliminar_dataset(mensaje):
    #Eliminamos las menciones
    mensaje = re.sub(r'@[A-Za-z0-9]+', '', mensaje)
    #Eliminamos los links
    mensaje = re.sub(r'https?://[A-Za-z0-9./]+', '', mensaje)
    #Eliminamos los hashtags
    mensaje = re.sub(r'#', '', mensaje)
    #Eliminamos los signos de puntuación
    mensaje = re.sub(r'[^\w\s]', '', mensaje)
    #Eliminamos los guin bajo
    mensaje = re.sub(r'_', '', mensaje)
    return mensaje

#Aplicamos la función a la columna de los mensajes
dataset['tweet'] = dataset['tweet'].apply(lambda x: eliminar_dataset(x))

# Eliminar la columna de sentimiento
dataset = dataset.drop(columns=['clasificacion'])

#Mostramos las primeras filas del dataset con el borrado
print(dataset.head(20), '\n')