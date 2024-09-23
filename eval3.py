import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    recall_score
)
import matplotlib.pyplot as plt

# Cargar los archivos CSV de entrenamiento
df_train = pd.read_csv('cars_cleaned.csv', delimiter=',')
df_train = df_train.drop(columns=['CODE', 'PRODUCTO', 'TIPO_CARROCERIA', 'COMBUSTIBLE', 'EDAD_COCHE', 'Tiempo'])

# Imputar valores faltantes si es necesario
df_train.fillna("Missing", inplace=True)

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

# Calcular probabilidades para la curva ROC
y_val_proba = model.predict_proba(X_val)[:, 1]  # Probabilidades de la clase positiva

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)

# Calcular Recall para cada clase
recall_per_class = recall_score(y_val, y_val_pred, average=None)

# Calcular precisión del modelo
accuracy = accuracy_score(y_val, y_val_pred)

# Graficar Recall y la curva ROC
plt.figure(figsize=(15, 6))

# Gráfico de Recall
plt.subplot(1, 2, 1)
plt.bar(range(len(recall_per_class)), recall_per_class, color='skyblue')
plt.xticks(range(len(recall_per_class)), label_encoder.classes_)
plt.xlabel('Clases')
plt.ylabel('Recall')
plt.title('Recall por Clase')
plt.ylim([0, 1])
plt.grid(axis='y')

# Gráfico de la curva ROC
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()

plt.tight_layout()
plt.show()

# Mostrar resultados
print(f"Precisión: {accuracy:.2f}")
class_report = classification_report(y_val, y_val_pred, target_names=label_encoder.classes_)
print("Reporte de Clasificación:")
print(class_report)
