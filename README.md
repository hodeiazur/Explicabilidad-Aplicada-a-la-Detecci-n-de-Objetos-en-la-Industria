# Explicabilidad Aplicada a la Detección de Objetos en la Industria
Este repositorio contiene el codigo desarrollado como parte de un Trabajo de Fin de Master (TFM). El proyecto aborda el desarrollo, entrenamiento y evaluacion de modelos de deteccion de objetos y personas, junto con la aplicacion y comparacion sistematica de tecnicas de Inteligencia Artificial Explicable (XAI) en escenarios industriales.

El objetivo principal es evaluar la calidad de las explicaciones que generan distintos metodos de explicabilidad cuando se aplican a modelos de deteccion de objetos, comparando tanto las arquitecturas de deteccion como las propias tecnicas XAI.

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

1. Dos escenarios de deteccion: deteccion de personas en secuencias de video real y deteccion de objetos industriales en imagenes sinteticas.
2. Dos arquitecturas de deteccion: Faster R-CNN (basado en ResNet50 + FPN) y YOLOv8s.
Sobre cada combinacion de escenario y modelo se aplican cinco tecnicas de explicabilidad (RISE, D-RISE, GradCAM-E, LIME y D-CLOSE), cuyos resultados se evaluan mediante metricas cuantitativas de fidelidad y localizacion.

## 📂 Estructura del repositorio
```
.
├── deteccion_de_humanos_en_video/
│   ├── Faster/
│   │   ├── faster_entrenamiento.py          # Entrenamiento de Faster R-CNN
│   │   ├── deteccion_humano_sort_*.py       # Inferencia con/sin tracker SORT
│   │   ├── sort.py                          # Tracker SORT
│   │   ├── ground_truth_sinMujerAtras.json  # Anotaciones ground truth (COCO)
│   │   ├── cam_ta1_ws2/                     # Frames de camara real
│   │   ├── videos/                          # Secuencias de video por accion
│   │   └── METRICAS/                        # Modulos de explicabilidad
│   │       ├── base.py                      # Generacion de mascaras
│   │       ├── rise.py                      # RISE
│   │       ├── drise.py                     # D-RISE
│   │       ├── gcame.py                     # GradCAM-E
│   │       ├── DRISE_main.py                # Evaluacion D-RISE
│   │       ├── GCAME_main.py                # Evaluacion GradCAM-E
│   │       ├── LIME_main.py                 # Evaluacion LIME
│   │       └── DCLOSE_main.py               # Evaluacion D-CLOSE
│   └── yolo/
│       ├── entrenar_yolo.py                 # Entrenamiento YOLOv8
│       ├── obtener_best_yolo.py             # Seleccion del mejor checkpoint
│       ├── human.yaml                       # Configuracion del dataset (1 clase)
│       ├── best.pt                          # Pesos del mejor modelo
│       └── metricas/                        # Modulos de explicabilidad (YOLO)
│
├── deteccion_de_objetos_en_imagenes/
│   ├── Faster/
│   │   ├── fasterRCNN.py                    # Entrenamiento Faster R-CNN
│   │   ├── validacion_faster.py             # Validacion del modelo
│   │   ├── metricas/                        # Modulos de explicabilidad
│   │   └── _out_sdrec_01/ ... _out_sdrec_15/  # Datos sinteticos (15 carpetas)
│   └── yolo/
│       ├── yolo-sinteticos.py               # Entrenamiento con grid search
│       ├── yolo-val.py                      # Validacion
│       ├── editar_datos.py                  # Preprocesado de datos
│       ├── dataset.yaml                     # Configuracion del dataset (11 clases)
│       ├── best.pt                          # Pesos del mejor modelo
│       └── metricas/                        # Modulos de explicabilidad (YOLO)
│
├── LICENSE                                  # AGPL-3.0
└── README.md
```

## 🎯 Escenarios de detección

### 👥 Detección de personas en vídeo

Se trabaja con frames extraidos de camaras reales en un entorno industrial. El dataset incluye secuencias de video con distintas posturas y acciones humanas (de pie, agachado, trabajando, hablando por telefono, etc.). Las anotaciones siguen el formato COCO y el modelo se entrena para una unica clase: **human**.

Se emplea el tracker **SORT** (Simple Online and Realtime Tracking) para mantener la identidad de las personas a lo largo de los frames.

### 🏭 Detección de objetos industriales en imágenes sintéticas

Se utilizan imagenes renderizadas sinteticamente (15 escenas), probablemente generadas con un motor de simulacion 3D. El dataset contiene 11 clases de objetos relacionados con un escenario de ensamblaje robotico:
```
**ID	Clase**
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
- **Variante humanos**: pesos preentrenados en COCO (backbone congelado, transfer learning). 10 epocas, SGD (lr=0.001, momentum=0.9), StepLR.
- **Variante objetos industriales**: pesos preentrenados en ImageNet (fine-tuning completo). 10 epocas, SGD (lr=0.005, momentum=0.9), StepLR.
- **Aumento de datos**: ColorJitter, MotionBlur, GaussianBlur, PixelDropout (humanos); HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Blur, RandomGamma (objetos industriales, via albumentations).
- **Descargar los modelos**:
   - detección humano:
   - detección objetos industriales:

### 🚀 YOLO
- **Modelo base**: YOLOv8s preentrenado (Ultralytics).
- Entrenamiento mediante **grid search** sobre batch size (8, 16), learning rate (1e-3, 5e-4) y tamano de imagen (640, 512).
- 50 epocas por configuracion.
- Optimizador por defecto de Ultralytics (AdamW con cosine annealing).

## 🧠 Técnicas de explicabilidad

Se implementan cinco tecnicas XAI, adaptadas para funcionar con detectores de objetos:
### RISE (Randomized Input Sampling for Explanation)
Metodo de perturbacion de caja negra. Genera N mascaras aleatorias binarias sobre la imagen de entrada y observa como varia la puntuacion del modelo. El mapa de saliencia se calcula como la suma ponderada de las mascaras, donde el peso es la confianza del modelo para cada version enmascarada.

### 🧩 D-RISE (Detection RISE)
Extension de RISE adaptada a la deteccion de objetos. A diferencia de RISE, el peso de cada mascara incorpora tanto la confianza de la deteccion como el IoU entre la caja predicha y la caja objetivo, lo que produce mapas de saliencia especificos para cada deteccion concreta.

- Configuracion tipica: N=5000 mascaras, p=0.25, batch GPU=8.

### ⚡ GradCAM-E (Gradient-weighted Class Activation Mapping for Explainability)
Tecnica de caja blanca basada en gradientes. Utiliza hooks de PyTorch para capturar las activaciones y los gradientes en la capa backbone.body.layer4 de Faster R-CNN. El mapa de activacion se pondera por los gradientes promediados espacialmente y se recorta a la region de interes (bounding box).

### 🧩 LIME (Local Interpretable Model-agnostic Explanations)
Metodo de perturbacion agnostico al modelo. Segmenta la imagen en superpixeles (SLIC, n=200) y aprende un modelo lineal local que explica la prediccion en funcion de la presencia o ausencia de cada superpixel. Se adapta a deteccion mediante un wrapper que convierte las puntuaciones de deteccion en pseudo-probabilidades.

- Configuracion: 1000 muestras perturbadas por imagen.

### ⚡ D-CLOSE (Detection Closed-form Local Occlusion Saliency Explanation)
Metodo determinista basado en oclusion. Para cada superpixel (en dos niveles de granularidad, 100 y 200 segmentos), se enmascara la region a negro y se mide la caida de confianza del detector. La contribucion de cada superpixel es proporcional a la diferencia entre la puntuacion original y la puntuacion con la oclusion.

## 📊 Metricas de evaluacion
Todas las tecnicas XAI se evaluan con las mismas cinco metricas:
| Metrica                | Tipo           | Descripcion |
|------------------------|---------------|-------------|
| Deletion Correlation   | Fidelidad     | Los superpixeles se eliminan de mas a menos saliente. Se mide la correlacion de Pearson entre la saliencia del superpixel y la caida de confianza al eliminarlo. |
| Insertion Correlation  | Fidelidad     | Partiendo de una imagen difuminada, se revelan los superpixeles de mas a menos saliente. Se mide la correlacion entre la saliencia y la ganancia de confianza. |
| Pointing Game          | Localizacion  | Comprueba si el pixel de maxima activacion del mapa de saliencia cae dentro de la caja ground truth (acierto/fallo binario). |
| EBPG                   | Localizacion  | Fraccion de la energia total del mapa de saliencia que se concentra dentro de la caja ground truth (valor continuo entre 0 y 1). |
| Sparsity               | Concentracion | Mide lo concentrado (no difuso) que es el mapa de saliencia. Valores altos indican explicaciones mas focalizadas. |

## 💻 Requisitos tecnicos

### Hardware
- **GPU con CUDA**: necesaria para el entrenamiento y recomendada para la inferencia. Los scripts utilizan torch.cuda de forma generalizada.
- Se recomienda un minimo de **8-16 GB de VRAM** para la generacion de mapas D-RISE con batches de GPU.

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
  **Nota**: el repositorio utiliza Git LFS para almacenar imagenes, videos y pesos de modelos. Asegurate de tener Git LFS instalado antes de clonar.

2. Crear un entorno virtual e instalar las dependencias:
3.
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
**Faster R-CNN (deteccion de personas)**:
```
python deteccion_de_humanos_en_video/Faster/faster_entrenamiento.py
```
**Faster R-CNN (objetos industriales)**:
```
python deteccion_de_objetos_en_imagenes/Faster/fasterRCNN.py
```
**YOLO (deteccion de personas)**:
```
python deteccion_de_humanos_en_video/yolo/entrenar_yolo.py
```
**YOLO (objetos industriales)**:
```
python deteccion_de_objetos_en_imagenes/yolo/yolo-sinteticos.py
```
### 🔍 Inferencia con tracking
```
python deteccion_de_humanos_en_video/Faster/deteccion_humano_sort_labelHuman_conReal.py
```
### 🧾 Evaluacion de explicabilidad
Los scripts de metricas se encuentran en las carpetas METRICAS/ o metricas/ de cada modulo. Ejemplo para D-RISE sobre Faster R-CNN (humanos):
```
python deteccion_de_humanos_en_video/Faster/METRICAS/DRISE_main.py
```
Analogamente para las demas tecnicas:
```
python deteccion_de_humanos_en_video/Faster/METRICAS/GCAME_main.py
python deteccion_de_humanos_en_video/Faster/METRICAS/LIME_main.py
python deteccion_de_humanos_en_video/Faster/METRICAS/DCLOSE_main.py
```  
  **Nota**: algunos scripts contienen rutas absolutas a datasets que deberan adaptarse a tu entorno local.

## 📜 Licencia
Este proyecto se distribuye bajo la licencia GNU Affero General Public License v3.0 (AGPL-3.0). Consulta el archivo LICENSE para mas detalles.


## 👤 Autor
Hodei Azurmendi Hormaetxe
