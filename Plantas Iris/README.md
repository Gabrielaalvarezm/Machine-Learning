# Informe: Clasificación de Plantas Iris usando Regresión Lineal  
#### Gabriela Alvarez Martinez - Ivan Yesid Camargo Bocachica

## Introducción
Este proyecto implementa un sistema de **clasificación de plantas Iris** utilizando un modelo de **regresión lineal**.  
El dataset empleado es el clásico *Iris Dataset*, que contiene medidas de largo y ancho de sépalos y pétalos de tres especies de flores: **Setosa, Versicolor y Virginica**.

---
## Objetivo
El objetivo principal es entrenar un modelo que pueda **clasificar nuevas flores** a partir de sus medidas, garantizando que las entradas se validen en los rangos reales de los datos.  
Además, se incluye una **gráfica de regresión lineal simple** para visualizar la relación entre el largo y el ancho de los pétalos.

---

## Procedimiento
1. **Carga de datos**  
   - Se utiliza el archivo `Iris.csv` que contiene la información de las flores.  
   - La columna `Id` se elimina porque solo contiene índices y no aporta información útil.

2. **Preparación de los datos**  
   - Se separan las variables independientes (**X**) que son las medidas de sépalos y pétalos.  
   - La variable dependiente (**y**) es la especie de la flor.  
   - Como las especies están en texto, se convierten en números mediante `LabelEncoder`:  
     - *Setosa = 0*  
     - *Versicolor = 1*  
     - *Virginica = 2*

3. **División de los datos**  
   - Se dividen en dos conjuntos:  
     - 70% para **entrenamiento** (el modelo aprende).  
     - 30% para **prueba** (se valida el desempeño del modelo).

4. **Entrenamiento del modelo**  
   - Se utiliza **LinearRegression** de `sklearn` para encontrar la relación matemática entre las medidas y la especie de la flor.  

5. **Evaluación del modelo**  
   - Se calculan las predicciones en el conjunto de prueba.  
   - Como los resultados de regresión son valores decimales, se redondean y se limitan al rango `[0, 2]`.  
   - Se mide la **precisión** con `accuracy_score`.

6. **Clasificación de nuevas flores**  
   - El usuario puede ingresar manualmente las medidas de una flor.  
   - Antes de aceptarlas, el programa valida que estén dentro de los rangos observados en el dataset (ejemplo: longitud del sépalo entre 4.3 y 7.9 cm).  
   - El sistema devuelve la especie correspondiente y también muestra la **precisión global del modelo**.

7. **Visualización con regresión lineal simple**  
   - Para representar gráficamente, se seleccionan dos variables:  
     - Longitud del pétalo (eje X).  
     - Ancho del pétalo (eje Y).  
   - Se genera un modelo de regresión lineal simple con estas dos variables.  
   - Se grafica un **diagrama de dispersión** de los datos y la **línea de regresión** que muestra la tendencia.

---

## Algoritmo Utilizado
El algoritmo aplicado fue la **Regresión Lineal**.  
Este método busca ajustar una línea (o hiperplano, en dimensiones mayores) que explique la relación entre las variables de entrada (**características de la flor**) y la variable de salida (**especie**).  

Aunque normalmente se usaría un algoritmo de clasificación (como *Regresión Logística o KNN*), en este caso se eligió **Regresión Lineal** por fines didácticos.  
Con una estrategia de redondeo, se logró convertir las salidas continuas en **clases enteras (0, 1, 2)** para clasificar las flores.

---

## Gráfica de Regresión Lineal
Se generó una gráfica para representar la relación entre el **largo** y el **ancho del pétalo**:  

- Los puntos en **rosa** muestran los datos reales del dataset.  
- La línea en **negro** representa la regresión lineal ajustada.  

Esta gráfica permite evidenciar cómo el modelo encuentra una tendencia entre las variables seleccionadas.

---

## Ejecución
1. Descargar el archivo `Iris.csv`.  
2. Ejecutar el script en Python.  
3. Ingresar manualmente las medidas de la flor cuando el programa lo solicite.  
4. Observar la clasificación de la especie y la precisión del modelo.  
5. Visualizar la gráfica de regresión lineal.
---

## Conclusiones
- El modelo alcanzó una **precisión cercana al 100%**, lo que demuestra que la regresión lineal puede aproximar de manera aceptable la clasificación de las especies.  
- Validar los valores de entrada garantiza que las predicciones se realicen únicamente dentro de rangos reales, evitando resultados incoherentes.  
- La gráfica de regresión simple refuerza la comprensión de la relación entre variables específicas, aunque el modelo completo considera las 4 medidas para clasificar.  
- Este sistema puede extenderse fácilmente para clasificar nuevas flores con solo ingresar sus medidas.

