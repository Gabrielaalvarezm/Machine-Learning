# Regresión Logistica Clasificación de Correos (Spam vs Ham)
#### Gabriela Alvarez Martinez - Ivan Yesid Camargo Bocachica

## Resumen  
En este proyecto se implementó un modelo de **regresión logística** para clasificar correos electrónicos como **SPAM** (correo no deseado) o **HAM** (correo normal).  
Se utilizó un dataset con características extraídas de los correos (longitud del cuerpo, número de adjuntos, cantidad de links, remitente, palabras clave relacionadas con dinero, entre otras).  

El objetivo fue **entrenar un modelo supervisado**, evaluarlo con métricas adecuadas y analizar la importancia de las variables que permiten distinguir entre correos normales y basura.  

---

## Objetivo  
Desarrollar y evaluar un modelo de **regresión logística** que permita identificar correos electrónicos SPAM con un alto nivel de precisión, analizando los factores más relevantes para la clasificación y presentando los resultados mediante métricas y gráficas explicativas.  

---

## Dataset y Preprocesamiento  

El dataset empleado se compone de registros de correos electrónicos, con las siguientes variables:  

- `longitud_cuerpo`: cantidad de caracteres del mensaje.  
- `num_adjuntos`: número de archivos adjuntos.  
- `num_links`: cantidad de enlaces presentes.  
- `remitente_empresa`: si el correo proviene de un dominio empresarial.  
- `contiene_palabras_dinero`: si el correo incluye palabras como "dinero", "premio", "oferta".  
- `urgencia`: si contiene expresiones como "urgente", "inmediatamente", etc.  
- `clasificacion`: etiqueta objetivo → HAM (0), SPAM (1).  

### Excepciones en los features
Algunos atributos no pudieron convertirse directamente a valores numéricos:  

- **Remitente en texto libre** (ejemplo: direcciones de correo exactas).  
  ➝ No se incluyó porque se requería un procesamiento más avanzado (codificación one-hot o embeddings).  

- **Contenido del mensaje en texto completo**.  
  ➝ No se usó, ya que el preprocesamiento de texto (TF-IDF, bag of words o embeddings) excedía el alcance del ejercicio.  

---

## Metodología  
1. **Preprocesamiento**: transformación de etiquetas (HAM=0, SPAM=1).  
2. **División de datos**: 80% entrenamiento, 20% prueba.  
3. **Optimización de hiperparámetros** con `GridSearchCV` para encontrar la mejor combinación de solver y penalty.  
4. **Entrenamiento del modelo**.  
5. **Evaluación con métricas**: F1-score, precisión, recall, matriz de confusión.  
6. **Visualización de resultados**: gráficas de desempeño, curva sigmoide, curva ROC y análisis de coeficientes.  
### Variables descartadas

Durante el preprocesamiento se identificaron algunas variables que no pudieron ser transformadas a valores numéricos de manera directa, por lo que no se incluyeron en el modelo:

- **Nombre del remitente:** es un campo demasiado variable, difícil de codificar sin técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP).
- **Texto del asunto:** al contener lenguaje natural libre, requeriría vectorización (Bag of Words, TF-IDF, embeddings) para convertirlo en valores numéricos.
- **Contenido completo del correo:** por su extensión y complejidad, no se trabajó en esta versión del proyecto.

Estas exclusiones no afectan el objetivo principal del ejercicio, que es ilustrar el uso de la regresión logística con variables numéricas más directas (longitud del cuerpo, número de adjuntos, urgencia, etc.).


---

## Resultados y Análisis  

### 1. Matriz de Confusión  
La matriz de confusión muestra los aciertos y errores del modelo.  

**Gráfico:**  
![Matriz de Confusión](matriz_confusion.png) 

---

### 2. Función Sigmoide  
La regresión logística se basa en la función sigmoide, que transforma valores numéricos en probabilidades entre 0 y 1.  

**Gráfico:**  
![Curva Sigmoide](curva_sigmoide.png) 

---

### 3. Curva ROC y AUC  
- La curva ROC representa la relación entre **verdaderos positivos (SPAM detectados correctamente)** y **falsos positivos (HAM clasificados como SPAM)**.  
- El AUC mide la capacidad general de clasificación.  

**Gráfico:**  
![Curva ROC](curva_ROC.png)

---

### 4. Importancia de las Variables  
El modelo asigna un peso (coeficiente) a cada variable:  
- Positivo ➝ aumenta la probabilidad de SPAM.  
- Negativo ➝ disminuye la probabilidad de SPAM.  

**Gráfico:**  
![Importancia de Variables](importancia_variables.png)

---

## Conclusiones  
- El modelo logró un buen desempeño en la detección de SPAM, mostrando un **F1-score equilibrado** entre precisión y recall.  
- Las variables más relevantes fueron:  
  - Presencia de **palabras relacionadas con dinero**.  
  - Número de **links** en el correo.  
  - Términos de **urgencia**.  
- Los atributos textuales complejos (remitente en texto libre, contenido completo del mensaje) no se incluyeron porque requerían técnicas de NLP más avanzadas.  

- La **regresión logística es un modelo simple pero eficaz** para este problema, y podría mejorar aún más con procesamiento de texto avanzado.  
