import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd

# Cargar el CSV con la columna 'Mas_1_coche'
csv_con_mas_1_coche = pd.read_csv('cars.csv')

# Cargar el CSV que no tiene la columna 'Mas_1_coche' para conocer la longitud
csv_sin_mas_1_coche = pd.read_csv('cars_input.csv')

# Seleccionar una muestra aleatoria del CSV más grande con la misma cantidad de filas que el CSV más pequeño
csv_con_mas_1_coche_sample = csv_con_mas_1_coche.sample(n=len(csv_sin_mas_1_coche), random_state=42)

# Guardar la muestra en un nuevo archivo CSV para su entrenamiento
csv_con_mas_1_coche_sample.to_csv('ALEX/muestra_para_entrenamiento.csv', index=False)

print("La muestra ha sido guardada como 'muestra_para_entrenamiento.csv'.")
# Cargar ambos CSVs
csv_con_mas_1_coche = pd.read_csv('ALEX/muestra_para_entrenamiento.csv')
csv_sin_mas_1_coche = pd.read_csv('cars_input.csv')

# Cargar los datos de entrenamiento (con columna Mas_1_coche) y test (sin columna Mas_1_coche)
df_train = pd.read_csv('ALEX/muestra_para_entrenamiento.csv', sep=';')
df_test = pd.read_csv('cars_input.csv', sep=';')

df_train = df_train.drop(columns=['CODE','TIPO_CARROCERIA'])
df_test = df_test.drop(columns=['CODE','TIPO_CARROCERIA'])

# Preprocesamiento: Codificar variables categóricas
df_train['ESTADO_CIVIL'] = df_train['ESTADO_CIVIL'].apply(lambda x: 1 if x == 'CASADO' else 0)
df_train['GENERO'] = df_train['GENERO'].apply(lambda x: 1 if x == 'M' else 0)
df_test['ESTADO_CIVIL'] = df_test['ESTADO_CIVIL'].apply(lambda x: 1 if x == 'CASADO' else 0)
df_test['GENERO'] = df_test['GENERO'].apply(lambda x: 1 if x == 'M' else 0)

# Definir características (X) y variable objetivo (y)
X = df_train[['ESTADO_CIVIL', 'GENERO', 'Edad Cliente', 'km_anno', 'COSTE_VENTA', 'Revisiones']]
y = df_train['Mas_1_coche'].astype(int)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
print("Evaluación del modelo en el conjunto de prueba:")
print(classification_report(y_test, y_pred))

# Predecir 'Mas_1_coche' en el segundo CSV (df_test)
X_new = df_test[['ESTADO_CIVIL', 'GENERO', 'Edad Cliente', 'km_anno', 'COSTE_VENTA', 'Revisiones']]
df_test['Mas_1_coche_Pred'] = clf.predict(X_new)

# Guardar las predicciones en un nuevo archivo
df_test.to_csv('ALEX/segundo_csv_con_predicciones.csv', index=False, sep=';')

# Cargar el CSV que tiene la columna Mas_1_coche para hacer la comparación
df_test_real = pd.read_csv('ALEX/muestra_para_entrenamiento.csv', sep=';')

# Comparar predicciones con los valores reales
predicciones = df_test['Mas_1_coche_Pred']
reales = df_test_real['Mas_1_coche'].astype(int)

# Calcular la precisión, exactitud y otras métricas
accuracy = accuracy_score(reales, predicciones)
print(f"Exactitud de las predicciones: {accuracy * 100:.2f}%")

print("Reporte de clasificación para la comparación:")
print(classification_report(reales, predicciones))
