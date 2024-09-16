import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
df = pd.read_csv('cars.csv', sep=';')

# Eliminar las filas donde la columna 'Mas_1_coche' esté vacía o tenga valores NaN

df = df.dropna(subset=['Mas_1_coche'])

# Comprobar el resultado
print(df())

# Verificar si la columna 'Mas_1_coche' está en el DataFrame
print(df.columns)

# Convertir la columna Mas_1_coche a valores numéricos (asumiendo que 0 = No, 1 = Sí)
df['Mas_1_coche'] = df['Mas_1_coche'].astype(int)

# Convertir columnas booleanas como Campanna1, Campanna2, etc., a valores numéricos (SI = 1, NO = 0)
boolean_columns = ['Campanna1', 'Campanna2', 'Campanna3', 'REV_Garantia', 'QUEJA_CAC']
for col in boolean_columns:
    df[col] = df[col].apply(lambda x: 1 if x == 'SI' else 0)

# Verificar si la columna 'Mas_1_coche' se ha convertido correctamente
print(df[['Mas_1_coche'].head()])

# Seleccionar las columnas numéricas para la matriz de correlación
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Calcular la matriz de correlación
correlation_matrix = df[numeric_columns].corr()

# Verificar si la columna 'Mas_1_coche' está en la matriz de correlación
print(correlation_matrix.columns)

# Dibujar la matriz de correlación usando seaborn y matplotlib
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['Mas_1_coche']], annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de correlación con Mas_1_coche')
plt.show()
