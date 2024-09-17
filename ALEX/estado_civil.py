import pandas as pd
import statsmodels.api as sm

# Cargar los datos desde el archivo CSV
df = pd.read_csv('cars.csv', sep=';')

# Codificar 'ESTADO_CIVIL' como 1 (CASADO) y 0 (NO CASADO)
df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].apply(lambda x: 1 if x == 'CASADO' else 0)

# Convertir 'Mas_1_coche' a valores numéricos (asumiendo que 0 = No, 1 = Sí)
df['Mas_1_coche'] = df['Mas_1_coche'].astype(int)

# Definir las variables dependiente (Mas_1_coche) e independiente (ESTADO_CIVIL)
X = df['ESTADO_CIVIL']
y = df['Mas_1_coche']

# Agregar constante (intercepto) para la regresión
X = sm.add_constant(X)

# Crear el modelo de regresión logística
logit_model = sm.Logit(y, X)

# Entrenar el modelo
result = logit_model.fit()

# Imprimir el resumen del modelo
print(result.summary())
