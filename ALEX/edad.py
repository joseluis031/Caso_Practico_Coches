import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('cars.csv', delimiter=';')

# Filtrar datos con más de un coche
df_con_mas_1_coche = df[df['Mas_1_coche'] == 1]

# Filtrar datos sin más de un coche
df_sin_mas_1_coche = df[df['Mas_1_coche'] == 0]

# Calcular la edad promedio para personas con más de un coche
edad_promedio_con_mas_1_coche = df_con_mas_1_coche['Edad Cliente'].mean()

# Calcular la edad promedio para personas sin más de un coche
edad_promedio_sin_mas_1_coche = df_sin_mas_1_coche['Edad Cliente'].mean()

print(f"Edad promedio para personas con más de un coche: {edad_promedio_con_mas_1_coche:.2f}")
print(f"Edad promedio para personas sin más de un coche: {edad_promedio_sin_mas_1_coche:.2f}")
