import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, recall_score
import matplotlib.pyplot as plt

# Cargar los archivos CSV de entrenamiento
df_train = pd.read_csv('cars_cleaned.csv', delimiter=',')
df_train = df_train.drop(columns=['CODE', 'PRODUCTO', 'TIPO_CARROCERIA', 'COMBUSTIBLE','EDAD_COCHE', 'Tiempo'])


# Identificar columnas categóricas
categorical_columns = df_train.select_dtypes(include=['object']).columns

# Convertir columnas categóricas a numéricas usando LabelEncoder
label_encoder = LabelEncoder()

for column in categorical_columns:
    df_train[column] = label_encoder.fit_transform(df_train[column])

# Separar las características (X) y la variable objetivo (y) en el conjunto de entrenamiento
X = df_train.drop(columns=['Mas_1_coche'])
y = df_train['Mas_1_coche']

# Dividir el conjunto de datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo XGBoost
model = xgb.XGBClassifier()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de validación
y_val_pred = model.predict(X_val)

# Evaluar el modelo
accuracy = accuracy_score(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)
class_report = classification_report(y_val, y_val_pred)

# Mostrar resultados
print(f"Precisión: {accuracy:.2f}")
print("Matriz de Confusión:")
print(conf_matrix)
print("Reporte de Clasificación:")
print(class_report)