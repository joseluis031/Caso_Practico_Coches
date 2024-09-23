#dividir analisis entre cat y numericas
#hacer boxplot de esto para ver como esta distribuido varianza num
#muchas cajas significa q hay distribucion, si hay puntos significa q esta muy dispersa(hay q ver si me interesa esa var o no)
#NUMERICAS hacer heatmap para correlacion
#pruebo feche engenerign limpiando y no limpiando pa ver q sale en el modelo
#CLASIFIERS 80 85
#curva de precision recall CRM1 2 Y 3 PARA VER LA % DE Q COMPRE OTRO COCHE, ME TENGO Q CENTRAR EN LOS Q TIENEN 50% PRO DE COMPRAR O NO, PARA INCITARLE Y LLAMAR A SU COMPRA
#Y LUEGO METO LOS CLIENTES NUEVOS DEL INPUT PARA VER EL % Q TIENEN DE COMPRA
#BUSCAR PRECISION DE 85 Y RECALL DE 55
#CALCULO ROI HASTA Q LA CURVA ESTE SATURADA(CUANDO YA ES RECTA) ROI CON LA INVERSION
#que significa balancear

import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("cars.csv", delimiter=";")

# Verificar si hay filas con valores nulos
null_data = df.isnull().sum()

# Mostrar el resultado de cuántos valores nulos hay en cada columna
#print("Valores nulos por columna:")
#print(null_data)

# Eliminar las filas con valores nulos
df2 = df.dropna()

# Guardar el DataFrame limpio en un nuevo archivo CSV
#df2.to_csv("cars_cleaned.csv", index=False)

# Eliminar la columna 'REV_Garantia'
df3 = df2.drop(columns=['REV_Garantia'])

# Verificar que la columna ha sido eliminada
print(df3.columns)