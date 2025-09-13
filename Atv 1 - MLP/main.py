import os
import pandas as pd
import kaggle

# Nome do dataset no Kaggle
dataset_name = 'johnsmith88/heart-disease-dataset'
file_name = 'heart.csv'

# Baixar o dataset
kaggle.api.dataset_download_files(dataset_name, path='.', unzip=True)

# Carregar o dataset para um DataFrame do Pandas
df = pd.read_csv(file_name)

# Visualizar as primeiras linhas
print(df.head())