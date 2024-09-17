# Importar las bibliotecas necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
archivo = "cars.csv"
df = pd.read_csv(archivo, delimiter=";")

# Seleccionar solo las primeras 100 filas
df = df.head(100)

# Eliminar la columna "CODE" ya que no aporta al análisis
df = df.drop(columns=['CODE'])

# Convertir variables categóricas a numéricas utilizando pd.get_dummies (one-hot encoding)
df = pd.get_dummies(df, drop_first=True)

# Seleccionar solo las columnas numéricas
df_numerico = df.select_dtypes(include=['int64', 'float64'])

# Calcular la correlación
corr = df_numerico.corr()

# Ajustar el tamaño de la figura para que quepan las etiquetas
plt.figure(figsize=(16, 12))

# Crear el heatmap con las correlaciones
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Rotar las etiquetas del eje X y Y para evitar que se sobrepongan
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Ajustar los márgenes para que las etiquetas no se corten
plt.subplots_adjust(bottom=0.3, left=0.2)

# Mostrar el gráfico
plt.title('Matriz de Correlación para Variables Numéricas', size=15)
plt.show()
