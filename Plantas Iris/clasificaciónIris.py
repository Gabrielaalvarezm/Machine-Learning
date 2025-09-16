# ================================================
#   REGRESIÓN LINEAL PARA CLASIFIACIÓN DE IRIS
# ================================================
# Gabriela Alvarez Martinez - Ivan Yesid Camargo Bocachica

# Se importan librerías necesarias para el ejercicio
# - Pandas y numpy sirven para manejar Plantitas en tablas y en arreglos numéricos
# - Matplotlib sirve para hacer gráficos de los Plantitas
# - Sklearn tiene herramientas para dividir los Plantitas, entrenar el modelo y medir resultados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

# ====================================================
# 1. Cargar los Plantitas del archivo de las plantas Iris 
# ====================================================
# Este archivo tiene información de las flores Iris (largo y ancho de pétalos y sépalos, y la especie)
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(ruta_base, "Iris.csv")

Plantitas = pd.read_csv(ruta)
# La columna "Id" solo tiene números consecutivos, no aporta nada para clasificar
# por eso la quitamos
Plantitas = Plantitas.drop("Id", axis=1)

# ====================================================
# 2. Separar los Plantitas en dos partes
# ====================================================
# X = las características numéricas (medidas de pétalos y sépalos)
# y = la especie (Setosa, Versicolor o Virginica)
X = Plantitas.drop("Species", axis=1).values
y = Plantitas["Species"].values

# Como los nombres de las flores son palabras, las convertimos en números:
# Setosa = 0, Versicolor = 1, Virginica = 2
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ====================================================
# 3. Dividir el conjunto de Plantitas en dos grupos:
# ====================================================
# - Entrenamiento (70%): para que el modelo "aprenda"
# - Prueba (30%): para verificar si el modelo aprendió bien
Xentreno, Xprueba, Yentreno, Yprueba = train_test_split(X, y, test_size=0.30, random_state=42)

# ====================================================
# 4. Crear el modelo de "Regresión Lineal"
# ====================================================
# Este modelo intenta encontrar una fórmula matemática que relacione las medidas
# de los pétalos y sépalos con la especie de flor
model = LinearRegression()

# ====================================================
# 5. Entrenar el modelo con los Plantitas de entrenamiento
# ====================================================
# Aquí el modelo "aprende" la relación entre las medidas y la especie
model.fit(Xentreno, Yentreno)

# ====================================================
# 6. Usar el modelo para predecir la especie de las flores del grupo de prueba
# ====================================================
# El resultado son valores decimales (por ejemplo 0.8, 1.9, etc.)
Ypredice = model.predict(Xprueba)

# ====================================================
# 7. Como necesitamos clases enteras (0, 1 o 2), redondeamos esos valores decimales
# ====================================================
Yprediceclase = np.rint(Ypredice).astype(int)

# ====================================================
# 8. Nos aseguramos de que las predicciones no se salgan del rango permitido (0, 1, 2)
# ====================================================
Yprediceclase = np.clip(Yprediceclase, 0, 2)

# ====================================================
# 9. Medir qué tan bien funcionó el modelo:
# ====================================================
# Se calcula la precisión comparando lo que predijo con la especie real
print("Precisión:", accuracy_score(Yprueba, Yprediceclase))

# ====================================================
# 10. Función para que el usuario ingrese valores manualmente
# ====================================================
# Pide al usuario las medidas de la flor y devuelve la especie predicha

# Primero sacamos los rangos válidos de las medidas (del dataset Iris)
rangos = {
    "sepal_length": (float(Plantitas["SepalLengthCm"].min()), float(Plantitas["SepalLengthCm"].max())),
    "sepal_width":  (float(Plantitas["SepalWidthCm"].min()),  float(Plantitas["SepalWidthCm"].max())),
    "petal_length": (float(Plantitas["PetalLengthCm"].min()), float(Plantitas["PetalLengthCm"].max())),
    "petal_width":  (float(Plantitas["PetalWidthCm"].min()),  float(Plantitas["PetalWidthCm"].max()))
}

def Pidevalorcito(mensaje, minimo, maximo):
    # Pide un valor al usuario y repite hasta que ingrese un número válido
    # dentro del rango [minimo, maximo].
    while True:
        entrada = input(f"{mensaje} ({minimo:.1f} - {maximo:.1f} cm): ")
        try:
            valor = float(entrada)
        except ValueError:
            print("Por favor ingresa un número válido.")
            continue
        if minimo <= valor <= maximo:
            return valor
        else:
            print(f"Valor fuera de rango. Debe estar entre {minimo} y {maximo} cm. Intenta de nuevo.")

def Clasificaplantita():   
    print("\nIngrese las medidas de la flor a clasificar (en cm):")

    # Aquí se usa Pidevalorcito para forzar que el usuario ingrese valores válidos
    sepal_length = Pidevalorcito("Longitud del sépalo", *rangos['sepal_length'])
    sepal_width  = Pidevalorcito("Ancho del sépalo", *rangos['sepal_width'])
    petal_length = Pidevalorcito("Longitud del pétalo", *rangos['petal_length'])
    petal_width  = Pidevalorcito("Ancho del pétalo", *rangos['petal_width'])

    # Se crea el arreglo con los Plantitas ingresados
    medidas = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Se hace la predicción
    pred = model.predict(medidas)
    clase = int(round(pred[0]))
    clase = np.clip(clase, 0, 2)

    # Se muestra el resultado
    print("La especie de la flor es:", encoder.inverse_transform([clase])[0])

    # Se hace la predicción
    pred = model.predict(medidas)
    clase = int(round(pred[0]))
    clase = np.clip(clase, 0, 2)

    # Mostrar también la precisión del modelo
    print("Precisión general del modelo:", accuracy_score(Yprueba, Yprediceclase))
# ====================================================
# 11. Llamada para que el usuario pueda probar
# ====================================================
Clasificaplantita()

# ======================================
# 12. Graficar regresión lineal con dos variables
# ======================================

# Usaremos dos columnas el largo y el ancho del pétalo
Simplex = Plantitas[["PetalLengthCm"]].values  # variable independiente (x)
Simpley = Plantitas["PetalWidthCm"].values     # variable dependiente (y)

# Se crear un modelo de regresión lineal solo para estas dos variables
Modelsimple = LinearRegression()
Modelsimple.fit(Simplex, Simpley)

# Hacer predicciones para la línea
Liniecitax = np.linspace(min(Simplex), max(Simplex), 100).reshape(-1, 1)
Liniecitay = Modelsimple.predict(Liniecitax)

# Grafica de dispersión y línea de regresión
plt.scatter(Simplex, Simpley, color="pink", label="Plantitas reales")  # puntos reales
plt.plot(Liniecitax, Liniecitay, color="black", linewidth=2, label="Regresión lineal")  # línea de regresión
plt.xlabel("Longitud del pétalo (cm)")
plt.ylabel("Ancho del pétalo (cm)")
plt.title("Regresión lineal: Longitud vs Ancho del pétalo")
plt.legend()
plt.show()