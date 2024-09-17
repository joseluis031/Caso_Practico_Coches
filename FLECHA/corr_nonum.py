# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("cars.csv", delimiter=";")

# Eliminar la columna "CODE" ya que no aporta al análisis
df = df.drop(columns=['CODE'])

# Seleccionar solo las columnas categóricas (tipo 'object')
df_categoricas = df.select_dtypes(include=['object'])

# Crear una copia del DataFrame categórico para aplicar Label Encoding
df_categorico_numerico = df_categoricas.copy()

# Inicializar el LabelEncoder
label_encoder = LabelEncoder()

# Iterar sobre las columnas categóricas y aplicar LabelEncoder a cada una
for column in df_categorico_numerico.columns:
    df_categorico_numerico[column] = label_encoder.fit_transform(df_categorico_numerico[column])

# Calcular la correlación para las variables categóricas transformadas
corr = df_categorico_numerico.corr()

# Visualizar la correlación con un heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Rotar las etiquetas del eje X y Y para evitar que se sobrepongan
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Ajustar los márgenes para que las etiquetas no se corten
plt.subplots_adjust(bottom=0.3, left=0.2)

# Mostrar el gráfico
plt.title('Matriz de Correlación de Variables Categóricas Transformadas', size=15)
plt.show()
