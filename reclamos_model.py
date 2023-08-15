#importar las librerias necesarias
import pickle
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# Carga el modelo preentrenado para análisis de sentimiento en español
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Carga el modelo preentrenado para clasificacion de texto
classifier = pipeline("zero-shot-classification", model="Recognai/bert-base-spanish-wwm-cased-xnli", multi_class=True)


