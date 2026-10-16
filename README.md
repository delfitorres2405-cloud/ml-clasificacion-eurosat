# Clasificación de suelo con imágenes satelitales (EuroSAT) 🛰️

Proyecto de **Machine Learning** en el que se clasifican imágenes satelitales de 64×64 píxeles en 10 categorías de uso de suelo. La idea consiste en comparar tres enfoques distintos y ver cuál funciona mejor: **redes neuronales convolucionales (CNN)**, **Transfer Learning con MobileNetV2** y **modelos basados en árboles**.

## Motivación

Mi motivación para llevar a cabo este proyecto surge a partir del aprendizaje de diversos conceptos de Machine Learning en una materia de la facultad. Los temas captaron mi interés profundamente y quería poner en práctica los conocimientos adquiridos mediante un proyecto propio, implementando distintos modelos para comprender mejor su funcionamiento y sacar mis propias conclusiones a partir de los resultados obtenidos.

Para ello, decidí implementar modelos de redes neuronales y árboles de decisión. Si bien, en teoría, los árboles de decisión pueden utilizarse para resolver problemas de clasificación de imágenes, a medida que avancé con el proyecto me di cuenta de que quizás no sean la opción más adecuada para este tipo de tareas. De esta forma, reafirmé que la experiencia práctica es fundamental para poner a prueba los conocimientos y continuar aprendiendo y profundizándolos. Los motivos se detallarán más adelante.

## Dataset

Se ha utilizado [EuroSAT RGB](https://github.com/phelber/EuroSAT), que tiene 27.000 imágenes del satélite Sentinel-2 de 64×64 píxeles y 3 canales (R, G, B). 

Imagen del satélite en cuestión:
<img width="1415" height="1061" alt="sat" src="https://github.com/user-attachments/assets/2cf88a84-a0d3-4bf4-80cb-9dd3e340da23" />

Las imágenes están etiquetadas en 10 clases:

| Clase | Descripción |
|---|---|
| AnnualCrop | Terrenos para cultivos anuales |
| Forest | Bosque |
| HerbaceousVegetation | Vegetación herbácea |
| Highway | Autopistas y carreteras |
| Industrial | Zonas industriales |
| Pasture | Terrenos de pastoreo |
| PermanentCrop | Terrenos para cultivos permanentes |
| Residential | Zonas residenciales |
| River | Ríos |
| SeaLake | Mares y lagos |

Las clases están levemente desbalanceadas (casi todas tienen entre 2.500 y 3.000 imágenes, aunque `Pasture` tiene solo 2.000). No es un desbalance grave, pero además de analizar el **Accuracy** de los modelos, se ha tenido en cuenta el **F1 por clase** y **F1 micro**.

El dataset se descarga solo con `torchvision` la primera vez que se corre el notebook.

## Herramientas utilizadas

- **Python 3.12**
- **TensorFlow / Keras** para armar y entrenar redes neuronales (CNN y MobileNetV2)
- **scikit-learn** para árboles, Random Forest, división de datos y métricas
- **torchvision** para descargar el dataset
- **NumPy y pandas** para manejar datos y armar tablas comparativas
- **Matplotlib** para curvas de entrenamiento y matriz de confusión
- **Jupyter Notebook** para desarrollar todo el proyecto

## División de datos

Se ha dividido el dataset en tres conjuntos usando `stratify`, para que cada uno mantenga la misma proporción de clases que el original:

| Conjunto | Proporción | Imágenes | Propósito |
|---|---|---|---|
| Entrenamiento | 60 % | 16.200 | Que el modelo aprenda |
| Validación | 20 % | 5.400 | Comparar configuraciones y frenar el entrenamiento (early stopping) |
| Testeo | 20 % | 5.400 | Evaluación final con datos que el modelo nunca vio |

Los píxeles quedan en su escala original (0-255). En las redes neuronales, la normalización se hace dentro del propio modelo con una capa `Rescaling`, y los árboles no lo necesitan.

## Modelos implementados

### 1. Redes neuronales convolucionales (CNN) 🧠

La arquitectura de las redes neuronales convolucionales (CNN) es particularmente útil en aplicaciones de procesamiento de imágenes. Las primeras capas de la red identifican características básicas, como bordes y texturas, mientras que las capas posteriores combinan estas características para reconocer patrones y atributos más complejos.

- **CNN base:** 3 bloques convolucionales (`Conv2D` + `MaxPooling2D`, con 32, 64 y 128 filtros), una capa densa de 128 neuronas y una salida de 10 neuronas. Se ha empleado el optimizador Adam, `ModelCheckpoint` para guardar el mejor modelo y `EarlyStopping` sobre la accuracy de validación.
- **CNN mejorada:** la misma arquitectura pero agregando **Dropout** (0.3 y 0.5) y **Data Augmentation** (giros horizontales y verticales y rotaciones), para intentar reducir el overfitting de la CNN base.

Luego, se implementaron redes neuronales utilizando Transfer Learning, con el objetivo de aprovechar los conocimientos adquiridos por una red preentrenada y evaluar si esto permite mejorar la exactitud de la clasificación.

### 2. Transfer Learning con MobileNetV2 🤓

Se ha utilizado MobileNetV2 preentrenada con ImageNet, sin su clasificador original (`include_top=False`) y con los pesos congelados (`trainable = False`). Arriba se ha agregado una capa `GlobalAveragePooling2D`, un `Dropout(0.3)` y una capa densa de 10 neuronas, y se ha entrenado con giros aleatorios como aumento de datos.

- **Versión 1 (64×64):** con las imágenes en su tamaño original.
- **Versión 2 (128×128):** se agregó una capa `Resizing` dentro del modelo para aumentar la resolución de entrada, ya que MobileNetV2 fue preentrenada con imágenes de 224×224 píxeles y, al utilizar un tamaño de 64×64, la accuracy de validación se estancaba.

### 3. Modelos basados en árboles 🌳

Los árboles no trabajan con imágenes sino con tablas, así que se ha aplanado cada imagen en un vector de 12.288 valores (un feature por cada píxel y canal). Al hacer eso se pierde la estructura espacial, por lo que estos modelos no pueden detectar bordes ni texturas.

- **Árbol de decisión base** con índice de Gini.
- **Árbol con poda:** poda por costo-complejidad con `ccp_alpha`. Probé 8 valores de alpha y me quedé con el que dio mayor accuracy en validación.
- **Random Forest** con 100 árboles.

## Resultados

Todos los modelos evaluados sobre el conjunto de testeo:

| Modelo | Accuracy | F1 macro |
|---|---|---|
| CNN base | 0.789 | 0.778 |
| CNN mejorada (Dropout + Data Augmentation) | 0.911 | 0.909 |
| Transfer Learning 64×64 | 0.872 | 0.868 |
| Transfer Learning 128×128 | 0.930 | 0.927 |
| Árbol Gini | 0.452 | 0.441 |
| Árbol con poda | 0.481 | 0.473 |
| Random Forest | 0.658 | 0.641 |

> Como los entrenamientos de las redes tienen componentes aleatorios (inicialización de pesos, orden de los lotes, Dropout y Data Augmentation), los valores pueden variar un poco entre ejecuciones.

## Conclusiones

- Las CNN le ganan por mucho a los modelos de árboles. Los árboles trabajan con los píxeles aplanados y pierden la estructura espacial de la imagen, mientras que las CNN pueden detectar bordes, texturas y patrones.
- Dropout y Data Augmentation mejoraron considerablemente a la CNN base.
- El mejor modelo fue **Transfer Learning con MobileNetV2 y las imágenes ampliadas a 128×128**, con un accuracy cercano al 93% en testeo. Se nota el beneficio de aprovechar lo que ya aprendió una red entrenada con un dataset mucho más grande.
- Agrandar las imágenes de 64×64 a 128×128 mejoró notablemente el Transfer Learning, probablemente porque queda más cerca del tamaño con el que se entrenó MobileNetV2.
- Entre los árboles, la poda y el Random Forest mejoran al árbol base, pero igual quedan muy lejos de las redes neuronales.
- En el mejor modelo, las clases con menor F1 fueron `Highway`, `Pasture`, `PermanentCrop` y `River`, aunque las diferencias con el resto son chicas.

### Limitaciones y qué mejoraría

- Entrené cada modelo una sola vez, sin repetir con distintas semillas, por lo que no tengo conocimiento de cuánta variación hay entre pasadas.
- EuroSAT se divide al azar, así que es una posibilidad que parches muy parecidos de una misma zona hayan quedado en entrenamiento y en testeo. En ese caso el modelo no estaría generalizando sino reconociendo algo que ya vio, y el accuracy en testeo no reflejaría del todo cómo respondería con imágenes de zonas nuevas.
- Hay otras arquitecturas preentrenadas y técnicas de ajuste que podrían dar mejores resultados, pero todavía no las conozco lo suficiente como para aplicarlas. Quedan como próximo paso para seguir aprendiendo.

## Cómo ejecutarlo

1. Clonar el repositorio:
```
   git clone https://github.com/TU_USUARIO/ml-clasificacion-eurosat.git
   cd ml-clasificacion-eurosat
```
2. Instalar las dependencias:
```
   pip install -r requirements.txt
```
3. Abrir el notebook:
```
   jupyter notebook clasificacion_eurosat.ipynb
```

El dataset se descarga solo en la carpeta `data/` y los modelos entrenados se guardan como archivos `.keras`. Los dos están en el `.gitignore`, así que no se suben al repo. Entrené todo en CPU, así que puede tardar un buen rato (el Transfer Learning a 128×128 es el más lento).

## Estructura del repositorio

```
├── clasificacion_eurosat.ipynb   # Notebook con todo el proyecto
├── requirements.txt              # Dependencias
├── README.md
├── LICENSE
└── .gitignore
```

## Licencia

Proyecto bajo licencia MIT. Ver el archivo `LICENSE`.
