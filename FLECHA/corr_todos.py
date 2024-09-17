# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar el archivo CSV
df = pd.read_csv("cars.csv", delimiter=";")



# Eliminar la columna "CODE" ya que no aporta al análisis
df = df.drop(columns=['CODE'])

# Crear una copia del DataFrame
df_numerico = df.copy()

# Inicializar el LabelEncoder
label_encoder = LabelEncoder()

# Iterar sobre las columnas que son de tipo object (categóricas)
for column in df_numerico.select_dtypes(include=['object']).columns:
    # Aplicar LabelEncoder a cada columna categórica
    df_numerico[column] = label_encoder.fit_transform(df_numerico[column])


# Calcular la correlación para las variables numéricas
corr = df_numerico.corr()

# Visualizar la correlación con un heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Rotar las etiquetas del eje X y Y para evitar que se sobrepongan
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Ajustar los márgenes para que las etiquetas no se corten
plt.subplots_adjust(bottom=0.3, left=0.2)

# Mostrar el gráfico
plt.title('Matriz de Correlación para Variables noNuméricas y Numericas', size=15)
plt.show()
