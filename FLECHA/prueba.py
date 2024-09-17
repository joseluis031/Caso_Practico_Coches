# Importamos las bibliotecas necesarias para CORRELACION
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
archivo = "cars.csv"
df = pd.read_csv(archivo, delimiter=";")

# Seleccionar solo las primeras 100 filas
df = df.head(1000)

# Eliminar la columna "CODE" y "PROVINCIA" (no aportan información relevante)
df.drop(columns=['CODE', 'PROVINCIA', 'Tiempo'], inplace=True)

# Reemplazar valores de "Mas_1_coche" para convertirlo en numérico (0: No ha comprado más de un coche, 1: Sí ha comprado)
df['Mas_1_coche'] = df['Mas_1_coche'].apply(lambda x: 1 if x > 0 else 0)

# Convertir variables categóricas a numéricas utilizando pd.get_dummies (codificación one-hot)
df = pd.get_dummies(df, drop_first=True)

# Limpiar los valores faltantes (podemos llenar con el promedio o eliminar filas con NA)
df.fillna(df.mean(), inplace=True)

# Calcular la correlación
corr = df.corr()

# Visualizar la correlación usando un heatmap de Seaborn
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Mostrar la gráfica
plt.title('Matriz de Correlación con Heatmap (100 primeras filas, sin CODE)', size=15)
plt.show()
