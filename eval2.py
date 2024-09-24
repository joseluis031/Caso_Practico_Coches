import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, recall_score, confusion_matrix
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

# Separar las características (X) y la variable objetivo (y)
X = df_train.drop(columns=['Mas_1_coche'])
y = df_train['Mas_1_coche']

# Dividir el conjunto de datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo XGBoost
model = xgb.XGBClassifier()

# Entrenar el modelo
model.fit(X_train, y_train)

# Obtener predicciones de probabilidad
y_val_prob = model.predict_proba(X_val)[:, 1]

# Obtener predicciones (clase final)
y_val_pred = model.predict(X_val)

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_val, y_val_prob)

# Graficar la curva Precision-Recall
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.', label='XGBoost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Calcular y mostrar el Recall por clase
recall_per_class = recall_score(y_val, y_val_pred, average=None)
print(f"Recall por Clase: {recall_per_class}")

# Curva ROC
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Mostrar el reporte de clasificación
class_report = classification_report(y_val, y_val_pred)
print("Reporte de Clasificación:")
print(class_report)

# Matriz de confusión
conf_matrix = confusion_matrix(y_val, y_val_pred)
print("Matriz de Confusión:")
print(conf_matrix)
