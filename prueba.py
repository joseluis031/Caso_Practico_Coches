
import pandas as pd






# Cargar el archivo CSV
df = pd.read_csv("cars_input.csv", delimiter=";")

# Verificar si hay filas con valores nulos
null_data = df.isnull().sum()

# Mostrar el resultado de cuántos valores nulos hay en cada columna
print("Valores nulos por columna:")
print(null_data)



# Eliminar las filas con valores nulos
df2 = df.dropna()

# Guardar el DataFrame limpio en un nuevo archivo CSV
df2.to_csv("cars_input_cleaned.csv", index=False)

# Filtrar solo las columnas categóricas (tipo object)
categorical_columns = df2.select_dtypes(include=['object']).columns

# Mostrar los valores únicos de cada columna categórica
for column in categorical_columns:
    print(f"Valores únicos en la columna '{column}':")
    print(df2[column].unique())
    print("\n")
    
