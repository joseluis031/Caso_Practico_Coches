import sys 
sys.path.append(r'C:/Users/Germán Llorente/Desktop/germiprogramer/Caso_Practico_Coches')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd

# Paso 1: Cargar los datos
data = pd.read_csv('cars.csv', delimiter=';')  # Ajusta la ruta y delimitador según tus datos

# Define mappings for categorical columns
mappings = {
    "PRODUCTO": {"A": 1},
    "TIPO_CARRO": {"TIPO1": 1},
    "COMBUSTIBLE": {"FUEL 1": 1},
    "Potencia_": {"Baja": 1},
    "TRANS": {"M": 1},
    "FORMA_PAGO": {"Contado": 1, "Otros": 2, "Financiera Marca": 3},
    "ESTADO_CIVIL": {"CASADO": 1},
    "GENERO": {"M": 1, "F": 2},
    "OcupaciOn": {"Empresa": 1, "Funcionario": 2},
    "PROVINCIA": {"Asturias": 1, "Toledo": 2, "Lerida": 3, "Madrid": 4, "Santa Cruz de Tenerife": 5},
    "Campanna1": {"SI": 1, "NO": 0},
    "Campanna2": {"SI": 1, "NO": 0},
    "Campanna3": {"SI": 1, "NO": 0},
    "Zona_Renta": {"Medio-Bajo": 1, "Medio": 2, "Alto": 3},
    "REV_Garantia": {"NO DATA": 0, "SI": 1},
    "Averia_grave": {"Averia muy grave": 3, "Averia leve": 1, "No": 0},
    "QUEJA_CAC": {"SI": 1, "NO": 0}
}

# Apply mappings to the DataFrame
for column, mapping in mappings.items():
    data[column] = data[column].map(mapping)

print(data.head())
'''
# Paso 2: Preprocesamiento
# Separar las características y la variable objetivo
X = data.drop(columns=['Mas_1_coche'])
y = data['Mas_1_coche']

# Dividir las columnas en numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Crear un transformador para las variables categóricas y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Passthrough para columnas numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # OneHotEncoding para las categóricas
    ])

# Crear un pipeline que incluya el preprocesamiento y el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Paso 3: División de los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Entrenar el modelo
pipeline.fit(X_train, y_train)

# Paso 5: Evaluación del modelo
y_pred = pipeline.predict(X_test)
print(2)
# Calcular la precisión y mostrar la matriz de confusión
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Precisión del modelo: {accuracy:.2f}')
print('Matriz de confusión:')
print(conf_matrix)
'''