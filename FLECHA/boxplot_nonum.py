# Importar las bibliotecas necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar el archivo CSV
df = pd.read_csv("cars.csv", delimiter=";")

# Eliminar la columna "CODE" ya que no aporta al análisis
df = df.drop(columns=['CODE'])

# Convertir la variable objetivo 'Mas_1_coche' a categórica (si es numérica)
df['Mas_1_coche'] = df['Mas_1_coche'].astype('category')

# Seleccionar solo las columnas categóricas
df_categorico = df.select_dtypes(include=['object'])

# Mantener la columna objetivo 'Mas_1_coche' en el DataFrame categórico
df_categorico['Mas_1_coche'] = df['Mas_1_coche']

# Inicializar el LabelEncoder para convertir variables categóricas en numéricas
label_encoder = LabelEncoder()

# Iterar sobre las variables categóricas (excluyendo 'Mas_1_coche') para hacer un boxplot
for column in df_categorico.columns:
    if column != 'Mas_1_coche':
        # Convertir las categorías en valores numéricos
        df_categorico[column] = label_encoder.fit_transform(df_categorico[column])
        
        # Crear el boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Mas_1_coche', y=column, data=df_categorico)
        plt.title(f'Boxplot de {column} en función de Mas_1_coche')
        plt.show()
