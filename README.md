# Explicabilidad Aplicada a la Detección de Objetos en la Industria
Este repositorio contiene el código desarrollado como parte de un Trabajo de Fin de Máster (TFM). El proyecto aborda el desarrollo, entrenamiento y evaluación de modelos de detección de objetos y personas, junto con la aplicación y comparación sistemática de técnicas de Inteligencia Artificial Explicable (XAI) en escenarios industriales.

El objetivo de este trabajo es analizar el impacto del uso de un nuevo detector de objetos en la calidad de las explicaciones generadas por las mismas capas de explicabilidad utilizadas en estudios previos, así como evaluar la existencia de métodos de explicabilidad que permitan reducir el tiempo computacional manteniendo la precisión de los modelos analizados.

## 📋 Tabla de contenidos
- Descripción general
- Estructura del repositorio
- Escenarios de detección
- Arquitecturas de detección
- Técnicas de explicabilidad
- Métricas de evaluación
- Requisitos técnicos
- Instalación
- Uso
- Licencia

## 📌 Descripción general
El proyecto se estructura en torno a dos ejes:

1. Dos escenarios de detección: detección de personas en secuencias de video real y detección de objetos industriales en imágenes sintéticas.
2. Dos arquitecturas de detección: Faster R-CNN (basado en ResNet50 + FPN) y YOLOv8s.
Sobre cada combinación de escenario y modelo se aplican cinco técnicas de explicabilidad (RISE, D-RISE, GradCAM-E, LIME y D-CLOSE), cuyos resultados se evaluan mediante métricas cuantitativas de fidelidad y localización.

## 📂 Estructura del repositorio
```
.
├── deteccion_de_humanos_en_video/
│   ├── Faster/
│   │   ├── faster_entrenamiento.py          # Entrenamiento de Faster R-CNN
│   │   ├── deteccion_humano_sort_*.py       # Inferencia con/sin tracker SORT
│   │   ├── sort.py                          # Tracker SORT
│   │   ├── ground_truth_sinMujerAtras.json  # Anotaciones ground truth (COCO)
│   │   ├── cam_ta1_ws2/                     # Frames de cámara real
│   │   ├── videos/                          # Secuencias de video por acción
│   │   └── METRICAS/                        # Módulos de explicabilidad
│   │       ├── base.py                      # Generacion de mascaras
│   │       ├── rise.py                      # RISE
│   │       ├── drise.py                     # D-RISE
│   │       ├── gcame.py                     # GradCAM-E
│   │       ├── DRISE_main.py                # Evaluación D-RISE
│   │       ├── GCAME_main.py                # Evaluación GradCAM-E
│   │       ├── LIME_main.py                 # Evaluación LIME
│   │       └── DCLOSE_main.py               # Evaluación D-CLOSE
│   └── yolo/
│       ├── entrenar_yolo.py                 # Entrenamiento YOLOv8
│       ├── obtener_best_yolo.py             # Selección del mejor checkpoint
│       ├── human.yaml                       # Configuración del dataset (1 clase)
│       ├── best.pt                          # Pesos del mejor modelo
│       └── metricas/                        # Módulos de explicabilidad (YOLO)
│
├── deteccion_de_objetos_en_imagenes/
│   ├── Faster/
│   │   ├── fasterRCNN.py                    # Entrenamiento Faster R-CNN
│   │   ├── validacion_faster.py             # Validación del modelo
│   │   ├── metricas/                        # Módulos de explicabilidad
│   │   └── _out_sdrec_01/ ... _out_sdrec_15/  # Datos sintéticos (15 carpetas)
│   └── yolo/
│       ├── yolo-sinteticos.py               # Entrenamiento con grid search
│       ├── yolo-val.py                      # Validación
│       ├── editar_datos.py                  # Preprocesado de datos
│       ├── dataset.yaml                     # Configuración del dataset (11 clases)
│       ├── best.pt                          # Pesos del mejor modelo
│       └── metricas/                        # Módulos de explicabilidad (YOLO)
│
├── LICENSE                                  # AGPL-3.0
└── README.md
```

## 🎯 Escenarios de detección

### 👥 Detección de personas en vídeo

Se trabaja con frames extraidos de cámaras reales en un entorno industrial. El dataset incluye secuencias de vídeo con distintas posturas y acciones humanas (de pie, agachado, trabajando, hablando por telefono, etc.). Las anotaciones siguen el formato COCO y el modelo se entrena para una única clase: **human**.

Se emplea el tracker **SORT** (Simple Online and Realtime Tracking) para mantener la identidad de las personas a lo largo de los frames.

### 🏭 Detección de objetos industriales en imágenes sintéticas

Se utilizan imágenes renderizadas sintéticamente (15 escenas), probablemente generadas con un motor de simulacion 3D. El dataset contiene 11 clases de objetos relacionados con un escenario de ensamblaje robótico:
```
ID	Clase
0	background
1	table_top_skin
2	battery_holder
3	t_connector
4	factory_peg_8mm
5	forge_round_peg_8mm
6	battery_individual
7	box
8	klt_bin
9	bms_b
10	bms_a
```
## 🤖 Arquitecturas de detección

### 🔬 Faster R-CNN
- **Backbone**: ResNet50 + Feature Pyramid Network (FPN).
- **Variante humanos**: pesos preentrenados en COCO (backbone congelado, transfer learning). 10 épocas, SGD (lr=0.001, momentum=0.9), StepLR.
- **Variante objetos industriales**: pesos preentrenados en ImageNet (fine-tuning completo). 10 épocas, SGD (lr=0.005, momentum=0.9), StepLR.
- **Aumento de datos**: ColorJitter, MotionBlur, GaussianBlur, PixelDropout (humanos); HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Blur, RandomGamma (objetos industriales, via albumentations).
- **Descargar los modelos**:
   - detección humano: https://drive.google.com/file/d/1UtmQOsVdYUflc7NE8Pfg6KP5zJmhdXzM/view?usp=sharing
   - detección objetos industriales: https://drive.google.com/file/d/1ul2Z_d70KQu_dYim55c9LW0qAoSqIiwN/view?usp=sharing

### 🚀 YOLO
- **Modelo base**: YOLOv8s preentrenado (Ultralytics).
- Entrenamiento mediante **grid search** sobre batch size (8, 16), learning rate (1e-3, 5e-4) y tamaño de imagen (640, 512).
- 50 épocas por configuración.
- Optimizador por defecto de Ultralytics (AdamW con cosine annealing).

## 🧠 Técnicas de explicabilidad

Se implementan cinco técnicas XAI, adaptadas para funcionar con detectores de objetos:
### RISE (Randomized Input Sampling for Explanation)
Método de perturbación de caja negra. Genera N máscaras aleatorias binarias sobre la imagen de entrada y observa como varía la puntuación del modelo. El mapa de saliencia se calcula como la suma ponderada de las máscaras, donde el peso es la confianza del modelo para cada versión enmascarada.

### 🧩 D-RISE (Detection RISE)
Extensión de RISE adaptada a la detección de objetos. A diferencia de RISE, el peso de cada mascara incorpora tanto la confianza de la detección como el IoU entre la caja predicha y la caja objetivo, lo que produce mapas de saliencia especificos para cada detección concreta.

- Configuración típica: N=5000 mascaras, p=0.25, batch GPU=8.

### ⚡ GradCAM-E (Gradient-weighted Class Activation Mapping for Explainability)
Técnica de caja blanca basada en gradientes. Utiliza hooks de PyTorch para capturar las activaciones y los gradientes en la capa backbone.body.layer4 de Faster R-CNN. El mapa de activación se pondera por los gradientes promediados espacialmente y se recorta a la región de interés (bounding box).

### 🧩 LIME (Local Interpretable Model-agnostic Explanations)
Método de perturbación agnostico al modelo. Segmenta la imagen en superpíxeles (SLIC, n=200) y aprende un modelo lineal local que explica la predicción en función de la presencia o ausencia de cada superpíxel. Se adapta a detección mediante un wrapper que convierte las puntuaciones de detección en pseudo-probabilidades.

- Configuración: 1000 muestras perturbadas por imagen.

### ⚡ D-CLOSE (Detection Closed-form Local Occlusion Saliency Explanation)
Método determinista basado en oclusión. Para cada superpíxel (en dos niveles de granularidad, 100 y 200 segmentos), se enmascara la región a negro y se mide la caída de confianza del detector. La contribución de cada superpíxel es proporcional a la diferencia entre la puntuación original y la puntuación con la oclusión.

## 📊 Métricas de evaluación
Todas las técnicas XAI se evalúan con las mismas cinco métricas:
| Metrica                | Tipo           | Descripcion |
|------------------------|---------------|-------------|
| Deletion Correlation   | Fidelidad     | Los superpíxeles se eliminan de más a menos saliente. Se mide la correlación de Pearson entre la saliencia del superpíxel y la caída de confianza al eliminarlo. |
| Insertion Correlation  | Fidelidad     | Partiendo de una imagen difuminada, se revelan los superpíxeles de mas a menos saliente. Se mide la correlación entre la saliencia y la ganancia de confianza. |
| Pointing Game          | Localizacion  | Comprueba si el píxel de maxima activacion del mapa de saliencia cae dentro de la caja ground truth (acierto/fallo binario). |
| EBPG                   | Localizacion  | Fracción de la energia total del mapa de saliencia que se concentra dentro de la caja ground truth (valor continuo entre 0 y 1). |
| Sparsity               | Concentracion | Mide lo concentrado (no difuso) que es el mapa de saliencia. Valores altos indican explicaciones más focalizadas. |

## 💻 Requisitos técnicos

### Hardware
- **GPU con CUDA**: necesaria para el entrenamiento y recomendada para la inferencia. Los scripts utilizan torch.cuda de forma generalizada.
- Se recomienda un mínimo de **8-16 GB de VRAM** para la generación de mapas D-RISE con batches de GPU.

### Software
- **Python** 3.9 o superior.
- Principales dependencias:

| Categoria                | Bibliotecas |
|--------------------------|-------------|
| Deep learning            | torch, torchvision |
| Deteccion YOLO           | ultralytics |
| Explicabilidad           | lime, shapely |
| Vision por computador     | opencv-python, Pillow, scikit-image |
| Aumento de datos         | albumentations |
| Computacion cientifica    | numpy, scipy |
| Evaluacion               | scikit-learn |
| Visualizacion             | matplotlib, seaborn, pandas |
| Utilidades               | tqdm |
| Tracking                 | SORT (incluido en el repositorio) |


## ⚙️ Instalacion
1. Clonar el repositorio:
```
git clone https://github.com/hodeiazur/Explicabilidad-Aplicada-a-la-Detecci-n-de-Objetos-en-la-Industria.git
cd Explicabilidad-Aplicada-a-la-Detecci-n-de-Objetos-en-la-Industria
``` 
  **Nota**: el repositorio utiliza Git LFS para almacenar imágenes, vídeos y pesos de modelos. Asegúrate de tener Git LFS instalado antes de clonar.

2. Crear un entorno virtual e instalar las dependencias:

```
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install lime shapely albumentations
pip install opencv-python Pillow scikit-image scikit-learn
pip install numpy scipy matplotlib seaborn pandas tqdm
```
  Ajusta la version de CUDA (cu118, cu121, etc.) segun tu configuracion de hardware.

## ▶️ Uso
### 🏋️ Entrenamiento
**Faster R-CNN (detección de personas)**:
```
python deteccion_de_humanos_en_video/Faster/faster_entrenamiento.py
```
**Faster R-CNN (objetos industriales)**:
```
python deteccion_de_objetos_en_imagenes/Faster/fasterRCNN.py
```
**YOLO (detección de personas)**:
```
python deteccion_de_humanos_en_video/yolo/entrenar_yolo.py
```
**YOLO (objetos industriales)**:
```
python deteccion_de_objetos_en_imagenes/yolo/yolo-sinteticos.py
```
### 🔍 Inferéncia con tracking
```
python deteccion_de_humanos_en_video/Faster/deteccion_humano_sort_labelHuman_conReal.py
```
### 🧾 Evaluación de explicabilidad
Los scripts de métricas se encuentran en las carpetas METRICAS/ o metricas/ de cada módulo. Ejemplo para D-RISE sobre Faster R-CNN (humanos):
```
python deteccion_de_humanos_en_video/Faster/METRICAS/DRISE_main.py
```
Análogamente para las demas tecnicas:
```
python deteccion_de_humanos_en_video/Faster/METRICAS/GCAME_main.py
python deteccion_de_humanos_en_video/Faster/METRICAS/LIME_main.py
python deteccion_de_humanos_en_video/Faster/METRICAS/DCLOSE_main.py
```  
  **Nota**: algunos scripts contienen rutas absolutas a datasets que deberán adaptarse a tu entorno local.

## 📜 Licencia
Este proyecto se distribuye bajo la licencia GNU Affero General Public License v3.0 (AGPL-3.0). Consulta el archivo LICENSE para más detalles.


## 👤 Autor
Hodei Azurmendi Hormaetxe
