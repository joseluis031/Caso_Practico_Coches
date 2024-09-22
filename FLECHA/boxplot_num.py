# Importar las bibliotecas necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("cars.csv", delimiter=";")

# Eliminar la columna "CODE" ya que no aporta al análisis
df = df.drop(columns=['CODE'])

# Convertir la variable objetivo 'Mas_1_coche' a categórica (si es numérica)
df['Mas_1_coche'] = df['Mas_1_coche'].astype('category')

# Seleccionar solo las columnas numéricas
df_numerico = df.select_dtypes(include=['number'])

# Mantener la columna objetivo 'Mas_1_coche' en el DataFrame numérico
df_numerico['Mas_1_coche'] = df['Mas_1_coche']

# Crear un boxplot para cada variable numérica en función de la variable objetivo
for column in df_numerico.columns:
    if column != 'Mas_1_coche':  # Asegurarse de que no sea la variable objetivo
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Mas_1_coche', y=column, data=df_numerico)
        plt.title(f'Boxplot de {column} en función de Mas_1_coche')
        plt.show()
