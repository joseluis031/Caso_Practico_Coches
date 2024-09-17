import pandas as pd

# Cargar los datos desde el archivo CSV
df = pd.read_csv('cars.csv', sep=';')

# Separar por género y calcular el porcentaje de hombres y mujeres
gender_count = df['GENERO'].value_counts(normalize=True) * 100
print("Porcentaje de Hombres y Mujeres:")
print(gender_count)

# Calcular la media del resto de variables para cada género
# Excluir las columnas no numéricas
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
gender_means = df.groupby('GENERO')[numeric_columns].mean()

print("\nMedia de las variables numéricas por género:")
print(gender_means)
