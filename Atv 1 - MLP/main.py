import os
import pandas as pd
import kaggle

# Nome do dataset no Kaggle
dataset_name = 'johnsmith88/heart-disease-dataset'
file_name = 'heart.csv'

# Baixando o dataset (descomentar para baixar o dataset, caso não tenha o arquivo heart.csv)
#kaggle.api.dataset_download_files(dataset_name, path='.', unzip=True)

# Lista com os nomes das colunas
colunas = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Carregando e Visualizando as primeiras linhas do dataset
df = pd.read_csv(file_name, names=colunas, skiprows=1)
df.head()