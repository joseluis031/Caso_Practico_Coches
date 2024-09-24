import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

'''
# 1. Cargar el archivo CSV
df = pd.read_csv("cars_cleaned.csv", delimiter=",")

# 2. Eliminar la columna 'REV_Garantia' si no la has eliminado antes
df = df.drop(columns=['CODE', 'PRODUCTO', 'TIPO_CARROCERIA', 'COMBUSTIBLE'])
#print(df.columns)

# 3. Transformar variables categóricas a numéricas
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# 4. Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['Mas_1_coche'])  # Features (sin la variable objetivo)
y = df['Mas_1_coche']  # Target (la variable objetivo)

# 5. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Crear el modelo XGBoost
model = xgb.XGBClassifier()

# 7. Entrenar el modelo
model.fit(X_train, y_train)

# 8. Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# 9. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo XGBoost: {accuracy:.4f}")


print(classification_report(y_test, y_pred))
'''
'''
# Cargar los archivos CSV de entrenamiento y test
df_train = pd.read_csv('cars_cleaned.csv', delimiter=',')
df_test = pd.read_csv('cars_input_cleaned.csv', delimiter=',')

# Eliminar columnas innecesarias del conjunto de entrenamiento y test
df_train = df_train.drop(columns=['CODE', 'PRODUCTO', 'TIPO_CARROCERIA', 'COMBUSTIBLE','EDAD_COCHE', 'Tiempo'])
df_test = df_test.drop(columns=['CODE', 'PRODUCTO', 'TIPO_CARROCERIA', 'COMBUSTIBLE'])



# Identificar columnas categóricas
categorical_columns = df_train.select_dtypes(include=['object']).columns

# Convertir columnas categóricas a numéricas usando LabelEncoder
label_encoder = LabelEncoder()

for column in categorical_columns:
    df_train[column] = label_encoder.fit_transform(df_train[column])
    # Asegurarse de que las mismas transformaciones se apliquen al conjunto de test
    if column in df_test.columns:
        # Asegurar que se manejen los valores no vistos en el entrenamiento, incluyendo NaNs
        df_test[column] = df_test[column].apply(lambda x: x if x in label_encoder.classes_ else "Missing")
        df_test[column] = label_encoder.transform(df_test[column])

# Separar las características (X) y la variable objetivo (y) en el conjunto de entrenamiento
X_train = df_train.drop(columns=['Mas_1_coche'])
y_train = df_train['Mas_1_coche']

# Asegurarse de que las columnas de test sean las mismas que las de entrenamiento
X_test = df_test

# Crear el modelo XGBoost
model = xgb.XGBClassifier()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de test
y_pred = model.predict(X_test)

# Mostrar las predicciones
print("Predicciones para el conjunto de test:")
print(y_pred)



# Si deseas guardar las predicciones en un archivo CSV
#output = pd.DataFrame({'Predicción_Mas_1_coche': y_pred})
#output.to_csv('predicciones_test.csv', index=False)
#print("Predicciones guardadas en 'predicciones_test.csv'.")
'''
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Cargar los archivos CSV de entrenamiento y test
df_train = pd.read_csv('cars_cleaned.csv', delimiter=',')
df_test = pd.read_csv('cars_input_cleaned.csv', delimiter=',')

# Eliminar columnas innecesarias del conjunto de entrenamiento
df_train = df_train.drop(columns=['CODE', 'PRODUCTO', 'TIPO_CARROCERIA', 'COMBUSTIBLE', 'EDAD_COCHE', 'Tiempo'])

# Identificar columnas categóricas en el conjunto de entrenamiento y test
categorical_columns_train = df_train.select_dtypes(include=['object']).columns
categorical_columns_test = df_test.select_dtypes(include=['object']).columns

# Convertir las columnas categóricas a tipo "category"
for column in categorical_columns_train:
    df_train[column] = df_train[column].astype('category')
for column in categorical_columns_test:
    df_test[column] = df_test[column].astype('category')

# Convertir las columnas categóricas en etiquetas numéricas usando LabelEncoder
label_encoder = LabelEncoder()

for column in categorical_columns_train:
    df_train[column] = label_encoder.fit_transform(df_train[column])
for column in categorical_columns_test:
    df_test[column] = label_encoder.fit_transform(df_test[column])

# Separar las características (X) y la variable objetivo (y) en el conjunto de entrenamiento
X_train = df_train.drop(columns=['Mas_1_coche'])
y_train = df_train['Mas_1_coche']

# Asegurarse de que las columnas de test sean las mismas que las de entrenamiento
X_test = df_test

# Crear el modelo XGBoost
model = xgb.XGBClassifier()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de test
y_pred = model.predict(X_test)

# Añadir las predicciones al dataframe original del test
df_test['Predicción_Mas_1_coche'] = y_pred

# Guardar el nuevo archivo CSV con las predicciones añadidas
df_test.to_csv('cars_input_cleaned_con_predicciones.csv', index=False)

print("Predicciones guardadas en 'cars_input_cleaned_con_predicciones.csv'.")
