import matplotlib.pyplot as plt
import pandas as pd

df_con_mas_1_coche = pd.read_csv('cars.csv', sep=';')
df_sin_mas_1_coche = pd.read_csv('cars_input.csv', sep=';')
# Histograma para edades con más de un coche
plt.hist(df_con_mas_1_coche['Edad Cliente'], bins=range(0, 100, 5), alpha=0.5, label='Con más de 1 coche')

# Histograma para edades sin más de un coche
plt.hist(df_sin_mas_1_coche['Edad Cliente'], bins=range(0, 100, 5), alpha=0.5, label='Sin más de 1 coche')

plt.xlabel('Edad')
plt.ylabel('Número de personas')
plt.title('Distribución de Edad según Número de Coches')
plt.legend(loc='upper right')
plt.show()


# Percentiles para edades con más de un coche
percentiles_con_mas_1_coche = df_con_mas_1_coche['Edad Cliente'].quantile([0.25, 0.5, 0.75])

# Percentiles para edades sin más de un coche
percentiles_sin_mas_1_coche = df_sin_mas_1_coche['Edad Cliente'].quantile([0.25, 0.5, 0.75])

print("Percentiles para personas con más de un coche:")
print(percentiles_con_mas_1_coche)

print("\nPercentiles para personas sin más de un coche:")
print(percentiles_sin_mas_1_coche)

