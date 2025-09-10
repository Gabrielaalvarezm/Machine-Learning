# ================================================
#   REGRESIÓN LOGÍSTICA PARA DETECTAR SPAM
# ================================================
# Gabriela Alvarez Martinez - Ivan Yesid Camargo Bocachica

# Se importan librerías necesarias para el ejercicio
# - pandas: para manejar los datos en forma de tabla
# - sklearn: para entrenar y evaluar el modelo de regresión logística
# - seaborn y matplotlib: para hacer gráficos
# - numpy: para cálculos matemáticos
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ====================================================
# 1. Cargar dataset con ruta automática
# ====================================================
# se lee el archivo CSV que contiene información de los correos electrónicos
# Cada fila es un correo, y cada columna es una característica (ej: longitud, adjuntos, links, etc.)
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(ruta_base, "Datasetml.csv")

datos = pd.read_csv(ruta_csv)

# ====================================================
# 2. Preprocesamiento
# ====================================================
# La columna "clasificacion" nos dice si el correo es SPAM o HAM (normal).
# Para que el modelo entienda los datos, transformamos:
# HAM 0  (correo normal)
# SPAM 1 (correo basura)
datos["clasificacion"] = datos["clasificacion"].map({"HAM": 0, "SPAM": 1})

X = datos[["longitud_cuerpo", "num_adjuntos", "num_links",
           "remitente_empresa", "contiene_palabras_dinero", "urgencia"]]
y = datos["clasificacion"]

# ====================================================
# 3. División en entrenamiento y prueba
# ====================================================
# Se divide el dataset en 2 partes:
# 80% de los datos para entrenar el modelo (enseñarle cómo son los correos)
# 20% de los datos para probar el modelo (ver qué tan bien predice en casos nuevos)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================================
# 4. Definir pipeline y búsqueda de mejores parámetros
# ====================================================
# La regresión logística tiene parámetros que afectan su desempeño, como:
# C: controla cuánto se ajusta el modelo a los datos
# penalty: tipo de regularización (para evitar que el modelo se complique demasiado)
# solver: el algoritmo matemático que se usa para entrenar
# se usa "GridSearchCV" para probar diferentes combinaciones y elegir la mejor.
pipeline = Pipeline([
    ("escalador", StandardScaler()),  # normaliza las variables
    ("modelo", LogisticRegression(max_iter=5000, random_state=42))
])

parametros = {
    "modelo__C": [0.01, 0.1, 1, 10],
    "modelo__penalty": ["l1", "l2"],
    "modelo__solver": ["liblinear", "saga"]
}

busqueda = GridSearchCV(
    pipeline,
    parametros,
    scoring="f1",
    cv=5,
    n_jobs=-1
)
busqueda.fit(X_entrenamiento, y_entrenamiento)

# ====================================================
# 5. Mejor modelo encontrado
# ====================================================
# Se muestra cuál fue la combinación de parámetros que dio mejores resultados
mejor_modelo = busqueda.best_estimator_
modelo_final = mejor_modelo.named_steps["modelo"]  # solo el modelo

print("========================================")
print("1) Mejores parámetros encontrados:")
print(busqueda.best_params_)
print("========================================")

# ====================================================
# 6. Evaluación del modelo
# ====================================================
# Se pruba el modelo con los datos de prueba para medir su desempeño
y_predicho = mejor_modelo.predict(X_prueba)

print("2) F1 Score en el conjunto de prueba:")
print(f1_score(y_prueba, y_predicho))
print("========================================")

print("3) Reporte de clasificación (Precisión, Recall, F1):")
print(classification_report(y_prueba, y_predicho, target_names=["HAM","SPAM"]))
print("========================================")

# ====================================================
# 7. Matriz de confusión
# ====================================================
# La matriz de confusión muestra cuántos correos se clasificaron bien o mal:
# Arriba izquierda: HAM bien clasificados
# Abajo derecha: SPAM bien clasificados
# Los otros dos valores son errores del modelo
print("4) Matriz de confusión:")
matriz = confusion_matrix(y_prueba, y_predicho)
print(matriz)

# Se dibuja la matriz en forma de gráfico para interpretarla mejor
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
            xticklabels=["HAM","SPAM"], yticklabels=["HAM","SPAM"])
plt.title("Matriz de confusión")
plt.xlabel("Predicción del modelo")
plt.ylabel("Valor real")
plt.show()

# ====================================================
# 8. Función sigmoide
# ====================================================
# La regresión logística se basa en la función sigmoide:
# convierte cualquier número en un valor entre 0 y 1, que se interpreta como probabilidad de SPAM.
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Graficamos cómo se ve la curva sigmoide
z = np.linspace(-10, 10, 100)
p = sigmoide(z)

plt.plot(z, p, color="blue")
plt.title("Función Sigmoide de la Regresión Logística")
plt.xlabel("z = β0 + β1*x1 + ... + βn*xn")
plt.ylabel("Probabilidad de SPAM (y=1)")
plt.grid()
plt.show()

# ====================================================
# 9. Curva ROC y AUC
# ====================================================
# La curva ROC compara:
# Qué tan bien detecta los SPAM (sensibilidad)
# Contra cuántos HAM se confunden como SPAM (falsos positivos)
# El AUC es el área bajo la curva, mide la capacidad general de clasificación.
y_probabilidades = mejor_modelo.predict_proba(X_prueba)[:, 1]  
fpr, tpr, thresholds = roc_curve(y_prueba, y_probabilidades)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title("Curva ROC")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# ====================================================
# 10. Importancia de las variables (coeficientes)
# ====================================================
# La regresión logística aprende un coeficiente (peso) para cada variable.
# Si el coeficiente es positivo aumenta la probabilidad de SPAM.
# Si es negativo disminuye la probabilidad de SPAM.
coeficientes = modelo_final.coef_[0]
caracteristicas = list(X.columns)

tabla_coef = pd.DataFrame({"Variable": caracteristicas, "Coeficiente": coeficientes})
tabla_coef = tabla_coef.sort_values(by="Coeficiente", ascending=False)

sns.barplot(data=tabla_coef, x="Coeficiente", y="Variable", palette="viridis")
plt.title("Importancia de las variables en el modelo")
plt.show()

# ====================================================
# 11. Mostrar la ecuación final del modelo
# ====================================================
# La regresión logística genera una ecuación de la forma:
# P(SPAM) = 1 / (1 + e^-(β0 + β1*x1 + β2*x2 + ... + βn*xn))
# Donde:
# β0 es el intercepto (constante del modelo)
# βi son los coeficientes aprendidos para cada variable
print("5)Intercepto (β0) del modelo:")
print(modelo_final.intercept_)
print("========================================")

print("6) Coeficientes (βi) de cada variable:")
print(modelo_final.coef_)
print("========================================")

print("7) Variables en orden para armar la ecuación:")
print(list(X.columns))
print("========================================")
