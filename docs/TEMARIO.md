# Temario de Computer Vision II — resumen por capítulos

Resumen de los apuntes de teoría de la asignatura *Computer Vision II*, capítulo a
capítulo. Para cada capítulo se indica: **(a)** secciones, **(b)** conceptos,
**(c)** fórmulas clave, **(d)** métodos y arquitecturas y **(e)** qué se puede
programar en PyTorch (y dónde está en este repositorio).

> Los capítulos 1–5 tienen formato de libro con ejercicios; los capítulos 6–9 son
> notas. Al final hay un apartado de *posibles erratas* detectadas en los apuntes.

---

## CAPÍTULO 1 — Introduction

**(a) Secciones:** 1.1 Python Beyond the Basics (1.1.1 Docstrings, 1.1.2 Linting, 1.1.3 Testing) · 1.2 Virtual Environments and Containers · 1.3 Git (Version Control, How it Works, GitHub Actions) · 1.4 Deep Learning in Production (Models in Production, Types of Deployment) · 1.5 Important Vision Architectures · 1.6 Transfer Learning (Transfering Knowledge, Fine-tuning) · 1.7 Exercises.

**(b) Conceptos**
- **Docstrings:** tres formatos, Sphinx (`:param:`, `:type:`, `:return:`, `:rtype:`), NumPy (`Parameters / ----------`) y Google (`Args: / Returns:`). Lo importante es usar siempre el mismo en todo el proyecto.
- **Linting** (análisis estático):
  - Typing: `list[str | int]`, `Literal["easy","normal","hard"]`, `torch.nn.Module`; el chequeo se hace con Mypy.
  - Formato con Black e isort. isort ordena los imports en stdlib → terceros → propios, y hay que configurarlo compatible con Black.
  - Fichero LICENSE y Makefile (`.PHONY`, `install`, `format`, `lint`…).
  - Complejidad cognitiva con complexipy (umbral típico 15). Sube con el anidamiento, la recursión, los break/continue y los cambios de contexto.
  - Otras herramientas: Flake8, Pylint (PEP 8) y Ruff (escrito en Rust).
- **Testing:** tests unitarios con PyTest.
- **Entornos virtuales:** ventajas (aislamiento, reproducibilidad, sistema limpio, probar varias versiones de Python). La herramienta es `uv`, de Astral.
- **Contenedores vs. máquinas virtuales:**
  - VM: hipervisor, aislamiento fuerte, pesada, arranque lento.
  - Contenedor (Docker): comparte el kernel, es ligero, arranca al instante, es portable y aísla menos.
- **Git:** working dir → `add` → staging → `commit` → repo local → `push` → remoto. También `fetch` vs `pull` y `reset`. GitHub Actions para CI/CD con YAML en `.github/workflows/`.
- **Producción:** modelo candidato vs. modelo campeón. Tres tipos de despliegue:
  - **Shadow:** ambos predicen y solo se devuelve la del campeón. Es el más seguro, pero hace 2 forward por petición.
  - **A/B testing:** típicamente 50/50 con test de significancia. Menos seguro, 1 forward.
  - **Canary:** se empieza con poco tráfico al candidato y se va subiendo. Seguridad intermedia, 1 forward; es lento si las etiquetas tardan en llegar.
- **Arquitecturas:**
  - **AlexNet (2012):** 5 conv + 3 FC, ReLU, dropout, data augmentation.
  - **VGGNet (2014):** filtros 3×3, hasta 19 capas, diseño homogéneo.
  - **GoogLeNet (2014):** módulo Inception (versiones naive y con reducción), conv 1×1 para reducir dimensión, clasificadores auxiliares, GAP en lugar de FC.
  - **ResNet (2015):** conexiones residuales/skip contra el problema de degradación; más de 100 capas.
  - **DenseNet (2017):** cada capa recibe todas las anteriores; reutiliza features y usa menos parámetros.
- **Transfer learning / fine-tuning** (Tabla 1.1):

  | | Dataset similar | Dataset diferente |
  |---|---|---|
  | Muchos datos | fine-tune de algunas o todas las capas | fine-tune de todo o entrenar desde cero |
  | Pocos datos | clasificador lineal en la última capa o backbone como extractor | caso malo: features intermedias, u otra estrategia/modelo (p. ej. few-shot) |

**(c) Fórmulas:** no hay fórmulas relevantes.

**(d) Métodos y arquitecturas:** AlexNet, VGG-16/19, GoogLeNet/Inception, ResNet, DenseNet. Shadow, A/B y Canary deployment.

**(e) Ejercicios PyTorch:**
- Cargar un modelo preentrenado (`torchvision.models.resnet18(weights=...)`), congelar el backbone (`requires_grad=False`), sustituir `model.fc` y hacer fine-tuning en MNIST (es el Ej. 11 del libro). Variante: descongelado progresivo.
- Implementar un bloque residual, un módulo Inception (con 1×1) y un dense block desde cero, comprobando las shapes.
- Tests PyTest de funciones de tensores.

---

## CAPÍTULO 2 — Object Detection

**(a) Secciones:** 2.1 Introduction (Definition of the Problem, Detecting a Single Object, Naive Approach) · 2.2 Previous Concepts (IoU, NMS, mAP, Selective Search, Anchor Boxes) · 2.3 R-CNNs · 2.4 Fast R-CNNs · 2.5 Faster R-CNNs · 2.6 Single Stage Object Detection (2.6.1 YOLO) · 2.7 Exercises.

**(b) Conceptos**
- **Representación de una bounding box:** (x1,y1,x2,y2), esquinas superior izquierda e inferior derecha, o bien (cx, cy, w, h).
- **Detectar un solo objeto:** CNN seguida de dos MLP (clase + coordenadas). Si solo hay una clase (p. ej. seguir el balón en un partido) basta la regresión.
- **Enfoque naive con crops:** hace falta añadir una clase "background". El número de cajas posibles explota (fórmula abajo); incluso tomando c cajas por píxel salen unas cHW.
- **IoU:** índice de Jaccard entre conjuntos de píxeles, en el rango [0,1].
- **NMS:**
  1. Se ordenan por confianza las cajas que no son background.
  2. Se toma la de mayor confianza y se eliminan las que tengan IoU > ε con ella (opcionalmente, solo si son de la misma clase).
  3. Se repite. Al final se pueden filtrar las de probabilidad < 0.5.
- **mAP:**
  1. Por clase, se ordenan las predicciones por confianza.
  2. Se calcula precision/recall barriendo umbrales (p. ej. linspace 0→1 con paso 0.05). Una predicción es TP si su IoU supera el umbral.
  3. AP = área bajo la curva PR (regla del trapecio).
  4. mAP = media de los AP; puede ponderarse por número de instancias.
- **Selective Search:** segmentación inicial de Felzenszwalb, después fusión greedy de las regiones más similares, y de ahí salen los candidatos.
- **Anchor boxes:** escala s∈(0,1], ratio r>0. Solo se usan las combinaciones que contienen s1 o r1, lo que da n+m−1 anchors por píxel.
- **R-CNN:**
  1. Selective search.
  2. CNN preentrenada truncada, con cada región redimensionada.
  3. Una SVM por clase.
  4. Regresión lineal para la caja.
  5. Se traducen las coordenadas a la imagen original.
  6. Umbral + NMS.
  
  Es lento: miles de forward por imagen.
- **Fast R-CNN:** un solo forward de la CNN (entrenable) sobre toda la imagen. Las propuestas se proyectan sobre el feature map, pasan por **RoI pooling** (todas a c×h2×w2, dando un tensor n×c×h2×w2) y luego FC → n×d con d=c·h2·w2, prediciendo clase y caja.
- **RoI pooling:** se divide la región H×W en celdas de ⌈H/h⌉×⌈W/w⌉ y se agrega con max o mean. En los bordes las celdas salen más pequeñas.
- **Faster R-CNN:** sustituye selective search por una **Region Proposal Network**:
  - Conv sobre el feature map y anchors en cada píxel.
  - Predicción binaria objeto/fondo + caja por anchor, luego NMS, y de ahí las propuestas que van al RoI pooling.
  - Se entrena conjuntamente (end-to-end), con 4 pérdidas.
- **Tabla comparativa (VOC 2007):**

  | | Tiempo/imagen | Speed-up | mAP |
  |---|---|---|---|
  | R-CNN | 50 s | 1x | 66.0% |
  | Fast R-CNN | 2 s | 25x | 66.9% |
  | Faster R-CNN | 0.2 s | 250x | 66.9% |

  Todos usan CE/BCE para la clase y L1/L2 para la caja.
- **Una etapa vs. dos etapas:** los de dos etapas son más precisos y más lentos; YOLO y SSD hacen una sola pasada sobre una rejilla.
- **YOLO:**
  - Rejilla S×S; la celda que contiene el centro del objeto es la responsable.
  - B cajas por celda (p, x, y, w, h) y **una sola clase por celda** (vector C).
  - Salida S×S×(5B+C). En el paper: S=7, B=2, C=20, es decir 7×7×30.
  - Red de 24 conv + 2 FC, inspirada en GoogLeNet (reducción 1×1 + conv 3×3). Leaky ReLU en todas las capas y activación lineal al final. Unos 45 FPS.
  - Entrenamiento con error cuadrático; λcoord=5 y λnoobj=0.5; se predice √w y √h.
  - La caja "responsable" es la de mayor IoU con el ground truth, lo que lleva a que los predictores se especialicen.
  - Limitaciones: falla con varias clases en una misma celda; como mucho B objetos por celda (mal con objetos pequeños agrupados, p. ej. bandadas de pájaros); trata casi igual los errores en cajas grandes y pequeñas.

**(c) Fórmulas**
- Pérdida para un solo objeto:
  - L = w·L_CE + (1−w)·L_L2
  - L_CE = Σ_i y_i·log( exp(ŷ_i) / Σ_j exp(ŷ_j) ). Al PDF le falta el signo menos.
  - L_L2 = ¼[(x1−x̂1)² + (y1−ŷ1)² + (x2−x̂2)² + (y2−ŷ2)²]
- Número de cajas en el enfoque naive: [H(H+1)/2]·[W(W+1)/2]. Para una imagen 3×3 salen 36.
- IoU = J(A,B) = |A∩B| / |A∪B|
- precision = TP/(TP+FP); recall = TP/(TP+FN); mAP = (1/N) Σ AP_i
- Anchors: w_a = w·s·√r, h_a = h·s/√r. Total: wh(n+m−1) anchors.
- Pérdida de YOLO, L = L_BB + L_O + L_NO + L_C:
  - L_BB = λcoord Σ_i^{S²} Σ_j^B 1^obj_ij [(x−x̂)² + (y−ŷ)² + (√w−√ŵ)² + (√h−√ĥ)²]
  - L_O = Σ Σ 1^obj_ij (C_i − Ĉ_i)²
  - L_NO = λnoobj Σ Σ (1 − 1^obj_i)(C_i − Ĉ_i)²
  - L_C = Σ_i 1^obj_i Σ_c (p_i(c) − p̂_i(c))². Se suma por celdas, no por cajas.

**(d) Métodos:** IoU, NMS, mAP/AP, Selective Search (Felzenszwalb), anchor boxes, R-CNN (con SVM + regresión lineal), Fast R-CNN (RoI pooling), Faster R-CNN (RPN), YOLO (v1), SSD (solo mencionado).

**(e) Ejercicios PyTorch** (muy típicos de examen; coinciden con los Ej. 2-8 del libro):
- `iou(box1, box2, fmt="xyxy"|"cxcywh")`, vectorizado sobre tensores N×4.
- `nms(boxes, scores, eps)` desde cero y comparación con `torchvision.ops.nms`.
- `mean_average_precision(pred, gt, iou_thr)`.
- `generate_anchors(h, w, scales, ratios)` con n+m−1 anchors por píxel.
- `roi_pool(feat C×H×W, out=(h,w))` con max/mean, comparado con `torchvision.ops.roi_pool`.
- Red de detección de un solo objeto con dos cabezas y pérdida ponderada CE + L2.
- Cabeza YOLO (salida reshape a S×S×(5B+C)) y **pérdida YOLO completa** con máscaras 1^obj.
- Codificar y decodificar el ground truth al formato de rejilla.

---

## CAPÍTULO 3 — Semantic Segmentation and Style Transfer

**(a) Secciones:** 3.1 Semantic Segmentation (Naive Approach, Transposed Convolution, Fully Convolutional Networks, U-Net) · 3.2 Style Transfer (Method, Extracting Features, Defining the Loss Function) · 3.3 Other tasks (Image Segmentation, Instance Segmentation, Panoptic Segmentation, Keypoint Estimation) · 3.4 Exercises.

**(b) Conceptos**
- **Segmentación semántica:** clasificación a nivel de píxel. El enfoque naive con ventana deslizante (clasificar el píxel central de cada parche) es ineficiente y no comparte features.
- **Softmax por canales:** se hace que la salida tenga la misma resolución que la entrada con un canal por clase, y se aplica softmax en la dimensión de canales.
- **Convolución transpuesta (upsampling):**
  - Cada elemento de la entrada multiplica al kernel, se coloca en su posición en un tensor intermedio y se suman todos los intermedios.
  - El padding se aplica a la **salida**: elimina p filas/columnas de cada lado.
  - El stride se aplica a los intermedios/salida.
  - Con varios canales funciona como la conv normal: un kernel c_i×kh×kw por cada canal de salida.
  - Ejemplo del libro: entrada [[0,1],[2,3]] y kernel [[0,1],[2,3]] dan [[0,0,1],[0,4,6],[4,12,9]]. Con padding 1 queda [4].
- **Inicializar la conv transpuesta con métodos clásicos de upsampling:**
  - Bed of Nails: rellena con ceros; kernel [[1,0],[0,0]] con stride 2.
  - Nearest Neighbor: kernel [[1,1],[1,1]] con stride 2.
  - Bilinear interpolation.
  - Max Unpooling: necesita guardar los índices del max pooling.
- **FCN:** CNN que reduce la resolución → conv 1×1 con #canales = #clases → conv transpuesta hasta la resolución original. Pérdida cross-entropy. En inferencia, cada canal se mapea al color de su clase.
- **U-Net:** encoder (conv + max pooling) y decoder (conv transpuesta), con **skip connections por concatenación** a la misma resolución. Nació para biomedicina y funciona con pocos datos.
- **Style Transfer** (Gatys):
  - Lo que se optimiza son los **píxeles de la imagen sintetizada**, que se inicializa con la imagen de contenido. La CNN preentrenada está congelada.
  - Las capas cercanas a la entrada dan detalle y las cercanas a la salida información global. El contenido se toma de una capa profunda; el estilo, de varias capas.
  - Las features de contenido y estilo se precalculan, y también la Gram del estilo.
- **Otras tareas:**
  - Image segmentation: no supervisada, por correlación de píxeles; no garantiza semántica.
  - Instance segmentation: distingue perro 1 de perro 2.
  - Panoptic: instancias para los "things" contables y clase para el "stuff" (cielo, hierba).
  - Keypoint estimation: con OpenPose, HRNet o MediaPipe.

**(c) Fórmulas**
- Conv transpuesta (stride 1): cada intermedio mide (nh+kh−1)×(nw+kw−1). Con stride s: (nh+kh+s−1)×(nw+kw+s−1), tal como lo escribe el libro. Padding p elimina 2p filas y 2p columnas.
- Content loss: L_C = (1/n_C) Σ_r 1/(C_r H_r W_r) Σ_{c,i,j} (f_r(x)_{ijc} − f_r(x_C)_{ijc})²
- Gram: X_r ∈ R^{C_r × H_rW_r}, G = X_r X_rᵀ ∈ R^{C_r×C_r}, con G_ij = x_i·x_j
- Style loss: L_S = (1/n_S) Σ_r 1/(C_r H_r W_r) Σ (X_r X_rᵀ − X^S_r X^S_rᵀ)²
- TV loss: L_V = 1/(2·3·H·W) Σ_c Σ_{i<H} Σ_{j<W} (|x_{i,j+1,c} − x_{i,j,c}| + |x_{i+1,j,c} − x_{i,j,c}|)
- Total: L = αL_C + βL_S + γL_V, opcionalmente con α+β+γ=1.

**(d) Métodos:** Transposed Conv, Bed of Nails, Nearest Neighbor, Bilinear, Max Unpooling, FCN, U-Net, Neural Style Transfer (Gram matrix, TV loss), Mask/instance/panoptic (solo conceptual), OpenPose, HRNet, MediaPipe.

**(e) Ejercicios PyTorch:**
- Conv transpuesta 2D desde cero (stride 1 y padding 0; luego stride y padding generales) y verificación contra `nn.ConvTranspose2d` / `F.conv_transpose2d`.
- Demostrar en código que los kernels [[1,0],[0,0]] y [[1,1],[1,1]] con stride 2 equivalen a bed of nails y nearest neighbor.
- Upsampling manual (bed of nails, NN, bilinear, max unpooling con índices, comparando con `nn.MaxUnpool2d`).
- FCN y U-Net con comprobación de shapes y concatenación en las skips (hay que cuidar el crop o padding).
- `gram_matrix(feat)` normalizada por C·H·W; content, style y TV loss; bucle de optimización sobre `x.requires_grad_()` con VGG congelada y hooks o `features[i]`.

---

## CAPÍTULO 4 — 3D and Video

**(a) Secciones:** 4.1 3D Representations (Depth Map, Surface Normals, Implicit Functions, Voxels, Point Cloud) · 4.2 3D Convolutions (Model, Max/Avg Pooling 3D) · 4.3 Predicting Representations (Depth Maps, Surface Normals, Simulating Implicit Functions, Voxel Classes, Point Clouds) · 4.4 Predicting Shapes (Metrics, 3D Datasets, Generating Point Clouds) · 4.5 Video (Introduction, Action Recognition, Comparation of the Models, Long-Term Temporal Structure, Other Problems in Video) · 4.6 Exercises.

**(b) Conceptos**
- **Representaciones 3D:**
  - **Depth map:** H×W, un canal; valor pequeño = cerca.
  - **Normal map:** vector 3D por píxel, visualizado como RGB = XYZ.
  - **Función implícita:** f(x,y,z)=0 en la superficie, >0 fuera y <0 dentro (función de distancia). Es continua, pero cara de evaluar en denso.
  - **Vóxeles:** V×V×V de ocupación, o V×V×V×k si guardan atributos. Regulares, pero ineficientes en memoria.
  - **Point cloud:** conjunto disperso de (x,y,z) más atributos. Ligera, pero no estructurada y difícil para CNNs.
- **Conv 3D:** entrada (D,C,H,W), donde D puede ser profundidad o tiempo; kernel (K_D,K_H,K_W). Es muy costosa.
- **Pooling 3D:** max y avg. El max equivale a un kernel con un 1 en la posición del máximo; el avg, a un kernel con 1/(K_D K_H K_W) en todas las posiciones.
- **Predecir depth:** FCN con 1 canal de salida y MSE. Problema: ambigüedad de escala/profundidad. Solución: scale-invariant loss; con λ=1 es totalmente invariante a multiplicar la predicción por c>0, y esto depende del log.
- **Predecir normales:** FCN con 3 canales y distancia coseno, en [0,2] (0 = misma dirección, 1 = ortogonales, 2 = opuestos).
- **Simular una función implícita:** se muestrean puntos y se usa su distancia a la superficie como etiqueta. Regresión de 3 entradas a 1 salida.
- **Clasificar vóxeles:** conv 3D (+ pooling 3D) → flatten → MLP con CE/BCE.
- **Clasificar point clouds** (estilo PointNet, aunque el libro no lo nombra): MLP compartido por punto → agregación **invariante a permutación** (max/mean) → MLP.
- **Métricas de forma:**
  - IoU: solo sirve en vóxeles; no capta estructuras finas y los valores bajos significan poco.
  - **Chamfer:** sin correspondencias; vale 0 si P=Q, es simétrica, diferenciable y aplicable a cualquier representación, pero sensible a outliers.
  - **F1@δ:** con umbral δ (p. ej. 1% del tamaño del objeto); según el libro, la mejor métrica de uso común, aunque sensible a δ.
- **Datasets:** ShapeNet (más de 50k mallas, 55 categorías; variantes ShapeNetCore y ShapeNetSem) y Pix3D (más de 10k imágenes reales con modelo 3D alineado, pose, máscaras y keypoints).
- **Generar point clouds desde una imagen:**
  - CNN y dos ramas: una FC que da P1 puntos globales, y una conv que da P2 puntos por píxel (reshape de 3P2×H″×W″ a H″W″P2×3).
  - Se concatenan en (P1 + H″W″P2)×3 y la pérdida es Chamfer.
- **Vídeo:** tensor T×C×H×W. Es muy pesado (unos 10 GB/min en HD a 30 fps), así que se reduce resolución y fps.
- **Modelos de action recognition:**
  - **Single-frame CNN:** promedio en la "fusion layer"; baseline; admite longitud arbitraria.
  - **Late fusion:** con FC (flatten + concat + MLP) o con pooling (media en T, H y W + MLP). Le cuesta el movimiento de bajo nivel.
  - **Early fusion:** reshape de T×3×H×W a 3T×H×W y CNN 2D; la primera conv colapsa el tiempo. No es invariante a desplazamientos temporales.
  - **3D CNN:** **C3D**, "la VGG de las 3D CNN", con 39.5 GFLOP (frente a 0.7 de AlexNet y 13.6 de VGG-16). Solo tolera pequeños desplazamientos temporales.
  - Early, late y 3D necesitan un T fijo.
- **Two-Stream:**
  - Rama espacial RGB y rama temporal de flujo óptico (TV-L1, RAFT, cv2).
  - De T frames salen T−1 flujos; la entrada temporal es 2(T−1)×H×W.
  - Fusión por concatenación + MLP o promediando predicciones.
  - Pros: necesita menos datos que C3D. Contras: el flujo es caro, no diferenciable y duplica la memoria.
- **I3D (Inflated):** se inflan los kernels 2D a 3D repitiendo los pesos en el tiempo (con /T) para aprovechar pesos preentrenados.
- **Datasets y comparación de modelos:** Sports-1M (unos 1.1M vídeos, 487 clases, etiquetas ruidosas automáticas, multi-label, top-5 accuracy) y Kinetics-400 (400 clases, al menos 400 clips de ~10 s cada una, más de 300k vídeos, top-1).
- **Estructura temporal larga:**
  - **CNN + LSTM:** la CNN suele estar congelada; salida por frame o una sola.
  - **Recurrent CNN (RCNN) multicapa:** h(L,t) depende de h(L,t−1) y h(L−1,t).
  - **SlowFast** (FAIR): vía lenta con pocos frames y rica en semántica; vía rápida con α veces más frames (α=8) y menos canales. El texto da "β=1/6 y 1/8", mezclando la notación. Hay fusión periódica entre ambas.
- **Otras tareas:** Temporal Action Localization (intervalos [inicio, fin] + etiqueta) y Spatio-Temporal Action Detection (una caja por frame + etiqueta).

**(c) Fórmulas**
- Conv 3D: O_{d,h,w} = Σ_i Σ_j Σ_k I_{d+i,h+j,w+k}·K_{i,j,k}
- D_out = ⌊(D + 2P_D − D_D(K_D−1) − 1)/S_D + 1⌋, y análogamente H_out y W_out.
- Scale-invariant: L = (1/HW) Σ (log y − log ŷ)² − λ/(HW)² (Σ (log y − log ŷ))². En el PDF el segundo término aparece con "+", pero para que sea invariante a escala con λ=1 tiene que ser "−" (Eigen et al.). Conviene comprobarlo en clase.
- Normales: L = (1/HW) Σ (1 − y·ŷ / (‖y‖‖ŷ‖))
- Chamfer: d_C(P,Q) = (1/|P|) Σ_p min_q ‖p−q‖² + (1/|Q|) Σ_q min_p ‖q−p‖²
- F1 = 2PR/(P+R), con TP definido como distancia ≤ δ.
- I3D: W3D[t,:,:] = (1/T)·W2D

**(d) Métodos:** Conv3d, MaxPool3d/AvgPool3d, FCN para depth/normales, PointNet-like, Chamfer, F1@δ, ShapeNet, Pix3D, Single-frame, Late/Early fusion, C3D, Two-Stream, optical flow (TV-L1, RAFT), I3D, CNN+LSTM, RCNN, SlowFast, Sports-1M, Kinetics-400, TAL.

**(e) Ejercicios PyTorch:**
- Conv 3D manual (1 canal, stride 1, padding 0; luego general; luego una capa multicanal) y verificación contra `nn.Conv3d`.
- Scale-invariant loss, con un test de que L(y, cŷ) = L(y, ŷ).
- Pérdida de distancia coseno para normales (`F.cosine_similarity`).
- **Chamfer distance** vectorizada (`torch.cdist`) y F1@δ.
- Clasificador de point clouds con max-pool, con un test de invariancia a permutación (`x[:, torch.randperm(N)]`).
- Clasificador de vóxeles con Conv3d.
- Modelos de vídeo con su manejo de shapes: single-frame (`view(B*T, ...)`), late fusion, early fusion (`reshape(B, 3T, H, W)`), 3D CNN (`permute` a B×C×T×H×W), CNN+LSTM, SlowFast simplificado.
- Inflar pesos 2D→3D (I3D).

---

## CAPÍTULO 5 — Explainability

**(a) Secciones:** 5.1 Introduction (Why Explainability?, Interpretability vs Explainability) · 5.2 Visualization of Internal Features (Features Along the Network, Maximally Activating Patches, Gradient Ascent, Feature Inversion, DeepDream) · 5.3 Perturbation-Based Methods (SHAP, Occlusion, RISE) · 5.4 Gradient-Based Methods (Saliency Maps, Input x Gradient, Guided Backpropagation, DeconvNet, SmoothGrad, Integrated Gradients) · 5.5 Ad-Hoc Methods (CAM, Grad-CAM, Guided Grad-CAM, Attention as Explainability) · 5.6 Evaluation (Common Problems, Loyalty and Inverse Loyalty, ROAR, Sanity Checks) · 5.7 Exercises.

**(b) Conceptos**
- **Motivación:** medicina, finanzas, derecho; el GDPR obliga a poder explicar decisiones automáticas.
- **Interpretabilidad vs. explicabilidad:** interpretabilidad = modelos white-box (árbol, regresión lineal); explicabilidad = explicaciones post-hoc de modelos black-box.
- **Jerarquía de features:** bordes y colores → partes → objetos o clases.
- **Maximally activating patches:**
  1. Pasar muchas imágenes y registrar la activación de un canal o capa.
  2. Reescalar a la resolución de la imagen.
  3. Mostrar los parches de activación máxima.
- **Gradient ascent:** se genera una imagen sintética que maximiza la activación. La agregación es mean, max o norma, a veces sobre |·|. Se regulariza con L2 y clipping.
- **Feature inversion:** reconstruye la imagen a partir de sus activaciones. En capas bajas sale casi la original; en profundas, algo más semántico y borroso.
- **DeepDream:** gradient ascent partiendo de la imagen real, que amplifica lo que la red ya ve (texturas en capas bajas, objetos en capas altas).
- **Métodos de perturbación:** son model-agnostic, pero caros. Trade-off del tamaño de parche (superpixel): parches grandes, menos forward pero menos precisión.
  - **SHAP:** valores de Shapley sobre regiones. En la práctica se aproxima con Kernel SHAP: se muestrean subconjuntos, se ponderan con π(S) y se ajusta una regresión lineal sobre vectores binarios de presencia; los coeficientes son los φ_i.
  - **Occlusion:** parche de tamaño S deslizante (como una conv), acumulando |f(x) − f′(x)| y contadores de "seen pixels"; el resultado es el cociente.
  - **RISE:** N máscaras aleatorias (Bernoulli S×S con p=0.5 → upsample bilineal → shift aleatorio); se acumula f′(x) en los píxeles **no** enmascarados y se divide por el número de veces vistos. Valores típicos: N=4000, s=7 o 14, p=0.5.
- **Métodos de gradiente:** son model-specific (solo redes neuronales) y baratos (1 forward + 1 backward).
  - **Saliency map** (Simonyan): max_k |∂f_c/∂x|. Ruidoso y sin signo.
  - **Input × Gradient:** penaliza los píxeles de valor bajo.
  - **Guided Backprop:** en la ReLU solo pasa el gradiente si la activación forward > 0 **y** el gradiente > 0. Es más limpio, pero no fiel a la causalidad del modelo.
  - **DeconvNet:** solo filtra por gradiente > 0, ignore la máscara forward.
  - **SmoothGrad:** promedia N mapas sobre x + N(0,σ²), con N=10-100 y σ = 10-20% del rango.
  - **Integrated Gradients:** integral desde un baseline x′ (imagen negra), con M=50-300 pasos.
- **Métodos ad-hoc:**
  - **CAM:** requiere última conv → GAP → lineal.
  - **Grad-CAM:** vale para cualquier CNN, pero no para redes que no trabajan con imágenes.
  - **Guided Grad-CAM:** producto elemento a elemento de Grad-CAM (dónde) y Guided Backprop (qué píxeles).
  - **Atención como explicación:** "attention is not explanation": no es causal y depende del contexto; es como confundir correlación con causalidad.
- **Problemas comunes y qué los mitiga:**

  | Problema | Lo mitiga |
  |---|---|
  | Gradientes inestables | SmoothGrad |
  | Cambios rápidos de signo | Usar valor absoluto |
  | Información irrelevante | Guided BP (no es class-discriminative), Grad-CAM |
  | Gradientes saturados | IG |
  | Gradientes que se desvanecen | Grad-CAM (usa capas profundas) |
  | Falta de contexto | IG, SmoothGrad, Grad-CAM |

- **Evaluación:**
  - **Loyalty:** quitar primero los píxeles más importantes debería hacer caer la salida rápido; AUC cerca de 1 es bueno.
  - **Inverse loyalty:** quitar primero los menos importantes debería hacerla caer despacio; AUC cerca de 0 es bueno.
  - **ROAR:** quitar los píxeles importantes y **reentrenar**; si la accuracy baja, la explicación es buena. Es caro.
  - **Sanity checks** (Adebayo): randomización de parámetros (de la salida hacia la entrada; el mapa debe cambiar) y randomización de etiquetas. En el ejemplo de MNIST, Grad-CAM sale el mejor e IG / Gradient×Input los peores.
- **Paper 4** (Simonyan, Vedaldi, Zisserman 2014, *Deep Inside Convolutional Networks*):
  - Visualización de clase: argmax_I S_c(I) − λ‖I‖², sobre el score **sin softmax**, porque maximizar la posterior puede conseguirse bajando las otras clases. Se inicializa con imagen cero y se suma la media.
  - Saliency por Taylor de primer orden: w = ∂S_c/∂I en I0, con M_ij = max_c |w_h(i,j,c)|.
  - Localización débilmente supervisada con GraphCut: GMMs con foreground > cuantil 95% y background < cuantil 30%, y se toma la mayor componente conexa. Logra 46.4% de error top-5 en ILSVRC-2013.
  - Equivalencia con DeconvNet salvo en la ReLU: el gradiente usa 1(X_n > 0) y DeconvNet usa 1(R_{n+1} > 0). El switch del max-pooling corresponde al argmax.

**(c) Fórmulas**
- Gradient ascent: argmax_x { agg(f(x)) − λ‖x‖² }, iterando x_{t+1} = x_t + α∇F(x_t)
- Feature inversion: min F(x) = ‖f(x) − f(x0)‖² + λ‖x‖²
- Shapley: φ_i = Σ_{S⊆N∖{i}} |S|!(M−|S|−1)!/M! · [f(S∪{i}) − f(S)]. El PDF pone el valor absoluto de la diferencia, que no es lo estándar. Ejemplo con regiones A, B, C: pesos 1/3, 1/6, 1/6, 1/3.
- Kernel SHAP: π(S) = (M−1) / [C(M,|S|)·|S|(M−|S|)]; modelo lineal g = φ0 + Σ φ_i x_i
- Saliency: S_ij = max_k |∂f_c/∂x_ijk|
- Input × Gradient: S_ij = max_k |x_ijk · ∂f_c/∂x_ijk|
- SmoothGrad: S = (1/N) Σ S(x + ε_i), con ε_i ~ N(0,σ²)
- IG: (x−x′) ∫_0^1 ∂f(x′+α(x−x′))/∂x dα ≈ (x−x′)·(1/M) Σ_k ∂f(x′ + (k/M)(x−x′))/∂x
- CAM: F_k = (1/hw) Σ A_ijk; f_c = Σ_k w^c_k F_k; M_c(i,j) = Σ_k w^c_k A_ijk
- Grad-CAM: α^c_k = (1/hw) Σ_ij ∂f_c/∂A_ijk; M_c = ReLU(Σ_k α^c_k A_k), y se interpola a la resolución de la imagen.
- Loyalty: 1 − |(f_c(x) − f_c(x′)) / f_c(x)| frente al % de píxeles eliminados; se resume con el AUC.

**(d) Métodos:** Maximally Activating Patches, Gradient Ascent / Class Model Visualisation, Feature Inversion, DeepDream, SHAP / Kernel SHAP, Occlusion, RISE, Saliency Maps, Input×Gradient, Guided Backprop, DeconvNet, SmoothGrad, Integrated Gradients, CAM, Grad-CAM, Guided Grad-CAM, atención de ViT, Loyalty / Inverse Loyalty, ROAR, Sanity Checks, GraphCut (paper).

**(e) Ejercicios PyTorch** (es el capítulo más "codificable"; el libro trae 22 ejercicios con ResNet preentrenada):
- Saliency (`x.requires_grad_()`, `out[0,c].backward()`, `x.grad.abs().amax(dim=1)`), Input×Grad, SmoothGrad e IG (bucle de interpolación).
- **Grad-CAM** con `register_forward_hook` y `register_full_backward_hook` en `layer4`, y CAM con los pesos de `fc`.
- Guided Backprop y DeconvNet modificando el backward de la ReLU con hooks o `torch.autograd.Function` (clamp del grad_out y máscara forward).
- Guided Grad-CAM.
- Occlusion y RISE con las acumulaciones "seen/values"; Kernel SHAP con regresión lineal (`torch.linalg.lstsq`).
- Gradient ascent, feature inversion y DeepDream optimizando la imagen.
- Curvas de loyalty con AUC (`torch.trapz`); test de randomización de parámetros (reinicializar capas en cascada y medir correlación).

---

## CAPÍTULO 6 — Adversarial Attacks

**(a) Secciones:** 1.1 Introduction – glossary and concepts (1.1.1 Example, 1.1.2 Fundamental terms and Threat Framework, 1.1.3 Types of attacks, 1.1.4 Hypothesis of why do they occur, 1.1.5 Transferability) · 1.2 Specific Attacks Algorithms (FGSM, PGD, Carlini–Wagner, Boundary Attacks, One Pixel Attack, Universal Adversarial Perturbations) · 1.3 Adversarial training (Ensemble AT, Label Leaking, Virtual AT, Optimisation side) · 1.4 Robustness (Data transfer & augmentation, Smooth activations, Defensive distillation, Unforeseen Adversaries).

**(b) Conceptos**
- **Ejemplo de juguete** con regresión logística: x y w de dimensión 10, wᵀx = −3 y σ(−3) = 0.047. Con δ = 0.5·sign(w) se llega a wᵀx_adv = 2, σ = 0.88.
- **Glosario:** original example, adversarial/perturbed example, perturbation, adversarial attack, transferability.
- **Tres condiciones de un ejemplo adversarial:** el modelo acierta en el original, falla en el perturbado, y un humano acierta en ambos.
- **Taxonomía:**
  - Por objetivo: untargeted (misclassification, confidence reduction) vs. targeted (targeted misclassification, source/target misclassification).
  - Por conocimiento: white-box, grey-box, black-box (oráculo o pares entrada/salida).
  - Por momento: poisoning (en entrenamiento, troyanos) vs. evasion (en test).
  - Por norma ℓp del presupuesto ε: p alto penaliza errores grandes en píxeles individuales; p bajo da perturbaciones dispersas.
- **Por qué ocurren** (Goodfellow 2015): no por sobreajuste ni no-linealidad, sino por la **linealidad en alta dimensión**. El efecto w·δ crece con n. La misma perturbación engaña a arquitecturas distintas.
- **Transferibilidad:** se puede atacar con un modelo sustituto (Szegedy, *Intriguing properties*).
- **Ataques:**
  - **FGSM:** un paso, ℓ∞, débil frente a defensas.
  - **PGD / I-FGSM:** iterativo con proyección a la bola; puede llevar init aleatoria. Es de los más fuertes.
  - **C&W:** targeted, white-box, ℓ2; optimización explícita con Adam y cambio de variable tanh; κ es el parámetro de confianza ("es la loss de una SVM").
  - **Boundary attack:** black-box puro, solo necesita la etiqueta. Empieza desde una imagen adversarial y camina hacia la original por rejection sampling. Lento; la variante es HopSkipJump (escrito "JopSkipJump" en el PDF).
  - **One Pixel:** black-box con Differential Evolution (población, mutación, recombinación); se adapta fácilmente a targeted o untargeted.
  - **UAP** (Moosavi-Dezfooli): una perturbación v para todas las imágenes, con DeepFool como subrutina (linealiza la frontera localmente).
- **Adversarial training:** generar x_adv por minibatch y minimizar la pérdida sobre ellos. Puede memorizar la perturbación y baja la accuracy limpia (unas 0.1, según el texto).
  - **Ensemble AT** (Tramèr): los adversariales se generan con un ensemble de modelos; fronteras más suaves.
  - **Label leaking:** con FGSM y etiqueta verdadera, el modelo acierta más en los adversariales que en los limpios, porque la perturbación filtra información de la etiqueta.
  - **VAT:** semisupervisado; Local Distributional Smoothness con KL entre p(y|x) y p(y|x+δ).
  - **Visión min-max** (Madry).
- **Robustez:**
  - Más datos: preentrenar adversarialmente en ImageNet y luego CIFAR-100; CutMix.
  - Activaciones suaves (GELU mejor que ReLU para AT).
  - Defensive distillation: el estudiante aprende de soft labels; C&W la sigue rompiendo.
  - Adversarios no vistos: los ataques fog o JPEG siguen afectando a modelos entrenados en ℓ∞, así que se entrena con todos los ataques conocidos.

**(c) Fórmulas**
- Problema general: argmin ‖δ_X‖ s.t. f(X+δ_X) = Y*, con X* = X + δ_X
- ‖v‖_p = (Σ|v_i|^p)^{1/p}
- Linealidad: f(x+δ) = w·x + w·δ + b; en el peor caso w·δ = (1/255)Σ|w_i| = n·m/255
- FGSM: x_adv = x + ε·sign(∇_x L(x,y;θ)), con ‖x_adv − x‖_∞ = ε
- PGD: x′ ← P_ε( x′ + α·sign(∇_{x′} L(k̂(x′), y)) ), T iteraciones.
- C&W: min ‖x − x′‖²_2 + c·f(x′), con f(x′) = max( max_{i≠t} Z(x′)_i − Z(x′)_t, −κ ) y x′ = ½(tanh(w) + 1)
- UAP: mientras Err(X_v) ≤ 1−δ, para cada x_i aún no engañado: Δv_i = argmin_r ‖r‖_p s.t. k̂(x_i+v+r) ≠ k̂(x_i), y v ← P_{p,ε}(v + Δv_i)
- AT: (1/n) Σ L(x_i^adv, y_i; θ)
- VAT:
  - δ = argmax_{‖δ‖≤ε} D_KL[p(y|x,θ) ‖ p(y|x+δ,θ)]
  - L = L_CE(etiquetados) + α·D_KL
- Min-max: min_θ E_{(x,y)}[ max_{‖δ‖_p≤ε} L(f_θ(x+δ), y) ]

**(d) Métodos:** FGSM, PGD / I-FGSM, C&W (L2), Boundary Attack, HopSkipJump, One-Pixel (Differential Evolution), UAP, DeepFool, Adversarial Training, Ensemble AT, VAT, Defensive Distillation, CutMix, GELU.

**(e) Ejercicios PyTorch** (muy típicos de examen):
- **FGSM** (con `torch.clamp(x_adv, 0, 1)`).
- **PGD ℓ∞**: proyección `x + (x_adv − x).clamp(−ε, ε)`, init aleatoria; también una versión ℓ2.
- Variantes targeted (descender la pérdida hacia la clase objetivo).
- C&W con variable w y Adam.
- Bucle de adversarial training (mezcla limpio/adversarial) y medición de clean vs. robust accuracy.
- Pérdida VAT con KL (`F.kl_div` con `log_softmax`).
- Reproducir numéricamente el ejemplo de juguete de la regresión logística.
- Transferibilidad: generar en un modelo A y evaluar en un modelo B.
- One-pixel con búsqueda aleatoria.

---

## CAPÍTULO 7 — Generative Models (incluye Bayesian Neural Networks)

**(a) Secciones:**
- **1. Generative Models:**
  - 1.1 Generative vs Discriminative.
  - 1.2 How do we model the real distribution of data (Taxonomy, MLE).
  - 1.3 AutoRegressive Models (PixelRNN, PixelCNN, Overview).
  - 1.4 AutoEncoder y VAE (auto-encoders; VAE con ELBO, Train & Inference, Reparametrization trick, Beta-VAE, VQ-VAE, Overview).
  - 1.5 GANs (Instability y Mode Collapse, Wasserstein GANs, Overview).
  - 1.6 Conditional Generation (CGAN, CVAE).
  - 1.7 Controlled Generation: Beyond Labels (Latent Space, Disentangled Factors).
  - 1.8 More Generative Models (Flow-Based, Diffusion).
- **2. Bayesian Neural Networks:** Bayesian Linear Model, NN vs BNN, Prediction (Bayesian model averaging), Key Differences, Why BNNs, MC Dropout.

**(b) Conceptos**
- **Tres marcos:**
  - Discriminativo p(y|x): confía de más en entradas fuera de distribución (el problema de la "rubbish input").
  - Generativo p(x): permite muestrear, evaluar densidad y detectar anomalías u OOD.
  - Condicional p(x|y).
- **Historia:** GMM, HMM, Hopfield, Boltzmann Machines.
- **Por qué no vale la salida de una red como densidad:** hay que usar p_θ(x) = e^{f_θ(x)}/Z_θ, y Z es intratable.
- **Taxonomía:**
  - Densidad explícita tratable: autorregresivos (regla de la cadena) y flows (invertibles).
  - Densidad aproximada: VAE (variacional), EBM (MCMC, annealing) y score matching.
  - Implícita: GAN, que muestrea pero no evalúa p(x).
- **MLE:** minimizar la log-verosimilitud negativa.
- **PixelRNN:**
  - Factorización píxel a píxel con un softmax de 256 valores (discreto mejor que continuo).
  - Variantes Row LSTM (raster) y Diagonal BiLSTM.
  - Canales R → G → B condicionados.
  - Mask A en la primera capa (excluye el canal actual) y Mask B en las siguientes.
  - Entrenamiento en paralelo, generación secuencial.
- **PixelCNN:** convoluciones enmascaradas, sin pooling, con campo receptivo acotado y conexiones residuales. Más rápida en training/eval, igual de lenta generando.
- **Resumen de los AR:** likelihood exacta, pero muestreo lento y sin código latente.
- **Autoencoders** (no son generativos):
  - Undercomplete vs. overcomplete. Con decoder lineal y MSE equivale a PCA.
  - **Denoising AE** (2008, antes que dropout; idea precursora de difusión).
  - **Sparse AE** (k-sparse).
  - **Contractive AE** (penaliza la norma del Jacobiano).
  - **Stochastic AE.**
- **Problemas del espacio latente de un AE:** no es continuo ni completo; muestrear z aleatorio da basura.
- **VAE:**
  - Prior p(z), likelihood p(x|z) (decoder), posterior p(z|x) intratable, aproximada por q_φ(z|x) (encoder).
  - Se maximiza el ELBO.
  - Prior gaussiano diagonal: dimensiones independientes, favorece el disentanglement.
  - Reparametrization trick: Normal, Gumbel-Softmax, Log-Normal.
- **β-VAE:** contra el colapso del posterior (q(z|x) ≈ p(z)); β con o sin annealing.
- **VQ-VAE:**
  - Codebook e ∈ R^{K×D} y cuantización por vecino más cercano.
  - Straight-through gradient (se copia el gradiente del decoder al encoder).
  - Tres pérdidas: reconstrucción, VQ y commitment (con stop-gradient). El codebook puede actualizarse por EMA.
  - VQ-VAE-2: los apuntes dicen explícitamente que no entra en examen.
- **Resumen del VAE:** espacio latente estructurado y muestreo rápido, pero borroso, con KL collapse.
- **GAN:**
  - Two-sample test; generador vs. discriminador (analogía artista/crítico). En el equilibrio D(·) ≈ 0.5.
  - Teóricamente minimiza la divergencia JS.
  - Generador: BCE con target 1 (forma no saturante). Discriminador: BCE con 1 para reales y 0 para falsos.
- **Inestabilidad:**
  - Entrenar el discriminador k:1 lleva a un discriminador "superior" y a gradientes que se desvanecen.
  - **Mode collapse:** el generador siempre produce lo mismo.
- **WGAN:**
  - Distancia Earth Mover / Wasserstein; el "crítico" no está acotado.
  - Hace falta 1-Lipschitz: con weight clipping (c pequeño → gradientes que se desvanecen; c grande → restricción débil) o con gradient penalty sobre interpolaciones x̃.
  - Por qué mejora: KL da ∞ con soportes disjuntos y JS tiene un salto; W es suave.
  - W-VAE: el término KL del ELBO se sustituye por W.
- **Resumen del GAN:** muestras nítidas, pero inestable, con mode collapse y sin likelihood. Familia: GAN, WGAN, StyleGAN, GauGAN, CycleGAN.
- **CGAN:** se concatena el one-hot al ruido; en el discriminador, mapas one-hot h×w como canales extra.
- **CVAE:** q(z|x,y). Ejemplo: RGB 32×32 + 10 clases da un tensor 13×32×32.
- **Generación controlada:**
  - Propiedades del espacio latente: continuidad, completitud, alineación semántica, aritmética.
  - Disentanglement (∂g/∂z_i solo afecta a la feature i). El entanglement viene de correlaciones en los datos, falta de dimensión o mala especificación.
  - β alto favorece el disentanglement pero empeora la reconstrucción.
  - **Classifier gradients:** z′ = z + α∇_z h_f(g(z)), sin reentrenar el generador.
  - **InfoGAN:** regularización por información mutua.
- **Normalizing flows:** cambio de variable, transformaciones invertibles con Jacobiano fácil; entrada y salida de igual tamaño. NICE, RealNVP, MAF, IAF. Likelihood exacta.
- **Diffusion / score-based:**
  - Score s_θ(x) = ∇_x log p(x); Z desaparece al derivar.
  - SDE forward y reverse; DDPM, Latent Diffusion / Stable Diffusion, U-Net.
  - Calidad SOTA, pero inferencia lenta. Los apuntes dicen que basta con entender el concepto, sin memorizar las ecuaciones.
- **BNN:**
  - Pesos ~ N(μ,σ²) con prior N(0,I); pérdida = loss + KL (Bayes by Backprop, Blundell).
  - Muestreo con reparametrización; a veces se parametriza con log-varianza ρ.
  - Predicción por Monte Carlo (Bayesian model averaging) con incertidumbre.
  - **MC Dropout:** dropout activo en inferencia; calibración con temperature scaling.

**(c) Fórmulas**
- p_θ(x) = e^{f_θ(x)}/Z_θ, con Z_θ = ∫e^{f_θ(x)}dx
- AR: p(x) = Π_t p(x_t | x_1..x_{t−1}); W* = argmax Σ log p(x^(i))
- Canales: p(x_{i,R}|x_{<i}) · p(x_{i,G}|x_{<i}, x_{i,R}) · p(x_{i,B}|x_{<i}, x_{i,R}, x_{i,G})
- AE: L = (1/n) Σ (x_i − f_θ(g_φ(x_i)))²
- Bayes: p(z|x) = p(x|z)p(z)/p(x); p(x) = ∫p(x|z)p(z)dz
- log p(x) = E_q[log p(x|z)] − KL(q(z|x) ‖ p(z)) + KL(q(z|x) ‖ p(z|x))
- ELBO = E_q[log p(x|z)] − KL(q‖p(z)) ≤ log p(x)
- Reparametrización: z = μ + σ⊙ε con ε ~ N(0,I); Gumbel-Softmax z = softmax((log α + g)/τ) con g = −log(−log ε)
- KL cerrada con prior N(0,I): −½Σ(1 + log σ² − μ² − σ²). No aparece en el PDF, pero es necesaria para programarlo.
- β-VAE: L = −E_q[log p(x|z)] + β·KL(q(z|x) ‖ p(z))
- VQ-VAE: L = ‖x − D(e_k)‖ + ‖sg[E(x)] − e_k‖ + β‖E(x) − sg[e_k]‖
- GAN: min_G max_D E_x[log D(x)] + E_z[log(1 − D(G(z)))]; el generador maximiza log D(G(z)).
- WGAN: min_g max_c E[c(x)] − E[c(g(z))], con |f(u) − f(w)| ≤ L‖u − w‖
- WGAN-GP: + λ·E[(‖∇c(x̃)‖₂ − 1)²], con x̃ = εx + (1−ε)g(z)
- W-VAE: −E_q[log p(x|z)] + λ·W(q(z|x), p(z))
- Classifier gradients: z′ = z + α∇_z h_f(g(z)); InfoGAN: L = L_ADV − λ·I(c; G(z,c))
- Flows:
  - p(x) = π(f⁻¹(x))·|det(df⁻¹/dx)|
  - log p_i(z_i) = log p_{i−1}(z_{i−1}) − log|det(df_i/dz_{i−1})|
  - log p(x) = log π_0(z_0) − Σ_i log|det J_i|
- Score: s_θ(x) = ∇_x log p_θ(x) = −∇_x f_θ(x)
- SDE forward: dx = f(x,t)dt + g(t)dw. Reverse: dx = [f − g²∇_x log p_t(x)]dt + g dw̄
- BNN:
  - μ*, Σ* = argmin Σ L(F_θ(x_i), y_i) + KL[q(θ) ‖ p(θ_0)]
  - Predicción: ŷ = (1/K) Σ_k F_{θ_k}(x̂), con θ_k ~ N(μ*, Σ*)

**(d) Métodos:** PixelRNN (Row LSTM, Diagonal BiLSTM), PixelCNN (masked conv A/B), AE / DAE / Sparse / k-Sparse / Contractive / Stochastic AE, VAE, β-VAE, VQ-VAE (y VQ-VAE-2), CVAE, W-VAE, GAN, CGAN, WGAN (clipping), WGAN-GP, InfoGAN, StyleGAN, GauGAN, CycleGAN, NICE / RealNVP / MAF / IAF, DDPM, Score models, Latent / Stable Diffusion, BNN (Bayes by Backprop), MC Dropout, EBM (MCMC).

**(e) Ejercicios PyTorch:**
- **VAE** completo: encoder → (μ, logvar), reparametrize, decoder, pérdida BCE/MSE + KL cerrada. Variantes β-VAE y CVAE (concatenar one-hot / label map de 13×32×32).
- Muestrear e interpolar en el espacio latente.
- **GAN** en MNIST: bucle alternando D (BCE real=1, fake=0, con `.detach()`) y G (BCE con target 1).
- WGAN con weight clipping (`p.data.clamp_`) y **gradient penalty** (`torch.autograd.grad(..., create_graph=True)` sobre x̃).
- CGAN.
- `MaskedConv2d` de tipo A/B para PixelCNN; NLL con CE de 256 clases.
- VQ: cuantizador con codebook, `torch.cdist` + argmin, straight-through `z_q = z + (z_q − z).detach()` y las 3 pérdidas.
- Gumbel-Softmax (`F.gumbel_softmax`).
- Capa de acoplamiento tipo RealNVP con log-det.
- Classifier gradients sobre z.
- Capa bayesiana `BayesLinear` (μ, ρ, reparametrización, KL) y predicción por Monte Carlo.
- **MC Dropout** (`model.train()` en inferencia; media y desviación de K pasadas).

---

## CAPÍTULO 8 — Attention & Transformers in Vision

**(a) Secciones:**
1. Introduction and Motivation: 1.1 Foundational Intro (QKV Framework, Scaled Dot-Product, Multi-Head, Transformer Architecture, Positional Encoding).
2. Vision Transformers: Architecture Overview, Patch Embedding, [CLS] Token, Positional Encoding, Classification, Training Challenges, Extensions DeiT y Swin.
3. Knowledge Distillation: Core Concepts, Types of Knowledge, DeiT Distillation Strategy, Teacher–Student Regimes, Cross-Modal, Adversarial GAN-Based.
4. ANNEX (precursores en NLP): Selective Focus, Database Analogy, Formal Definition, Attention Pooling (Nadaraya-Watson, Kernels, Adaptive), Score Functions (derivación del scaled dot-product, versión batched, alternativas), Multi-Head.
5. Transformer Architecture: Encoder, Decoder.
6. Positional Encoding.

**(b) Conceptos**
- **Motivación:** el cuello de botella del vector fijo en seq2seq con RNN. Bahdanau (2014) introduce la atención; permite paralelismo y conexiones directas entre posiciones.
- **QKV como búsqueda blanda en una base de datos:** los pesos son no negativos, suman 1 (convexos) y son diferenciables (argmax requeriría RL).
- **Nadaraya-Watson:** query = punto de predicción, keys = x_i, values = y_i. No requiere entrenamiento.
  - Kernels: Gaussiano, Boxcar, Epanechnikov. Propiedades: simetría, normalización, invariancia a traslación y rotación.
  - Ancho σ aprendible: σ pequeño = atención afilada; σ grande = atención difusa.
  - Nota del libro: "kernel" de regresión no es lo mismo que kernel de CNN.
- **Derivación del producto escalar:** se expande el gaussiano; el término ‖q‖² desaparece por invariancia del softmax y ‖k‖² se vuelve aproximadamente constante con LayerNorm; queda qᵀk. Se escala por 1/√d porque la varianza crece con d y satura el softmax.
- **Otros scores:** aditivo de Bahdanau, location-based, content-based (coseno).
- **Multi-head:** subespacios distintos con proyección W_O.
- **Transformer:**
  - Encoder: N capas de MHSA + FFN, cada una con residual + LayerNorm (Post-LN; las variantes modernas usan Pre-LN).
  - Decoder: masked MHSA (máscara causal) + cross-attention (Q del decoder, K y V del encoder) + FFN.
  - La FFN proyecta a 4·d_model.
- **Positional encoding:** sinusoidal en el paper original; aprendible 1D por patch en ViT. Hace falta porque la atención es invariante a permutación.
- **ViT** (*An Image is Worth 16x16 Words*):
  - Patches p×p, m = hw/p², cada uno de c·p² valores, proyectados a d. Se implementa como `nn.Conv2d(3, 768, kernel_size=16, stride=16)`.
  - Ejemplo: imagen 256×256 con parches de 16 da 256 tokens de 768 → con [CLS], 257 tokens.
  - [CLS] = `nn.Parameter(torch.randn(1,1,768))`; pos-emb aprendible; N = 12 o 24 capas; cabeza lineal sobre Z^(N)_0.
- **Retos de ViT:**
  - Poco sesgo inductivo (sin localidad ni invariancia a traslación), así que necesita ImageNet-21k (14M) o JFT-300M, o mucha augmentation.
  - Coste cuadrático: 256² = 65,536.
  - Inestabilidad en el entrenamiento.
- **DeiT:** distillation token, profesor CNN (ResNet), RandAugment, Mixup y CutMix.
- **Swin:** ventanas locales desplazadas en capas alternas y estructura jerárquica tipo pirámide; complejidad O(n²) → O(n).
- **Knowledge distillation:**
  - Por respuesta: soft (KL con temperatura τ; τ>1 suaviza, τ<1 endurece) y hard-label.
  - Por features: p. ej. ResNet-50 → ResNet-18, con L2 sobre feature maps y conv 1×1 para igualar canales.
  - Por relaciones: matrices de similitud o mapas de atención (KDP en continual learning).
  - Regímenes: offline (profesor congelado), online (simultáneo, ensemble o EMA), self-distillation.
  - Cross-modal (visión-lenguaje, LiDAR+cámara → solo cámara).
  - Basada en GAN: data-free con generador, entre componentes de la GAN, o conjunta generador-estudiante.

**(c) Fórmulas**
- Attention(q, D) = Σ_i α(q, k_i)·v_i, con α = softmax(a(q, k_i)) = exp(a_i)/Σ_j exp(a_j)
- a(q,k) = qᵀk/√d; Attention(Q,K,V) = softmax(QKᵀ/√d)·V, con Q ∈ R^{n×d}, K ∈ R^{m×d}, V ∈ R^{m×v}
- MultiHead = [head_1;…;head_h]·W_O, con head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
- Nadaraya-Watson: f(x) = Σ K(x−x_i)·y_i / Σ K(x−x_i)
  - Gaussiano: K(u) = (1/√2π)·exp(−u²/2)
  - Boxcar: K(u) = 1 si |u|<1
  - Epanechnikov: max(1−|u|, 0)
  - Adaptativo: α = exp(−‖q−k‖²/(2σ²))
- Derivación: −½‖q−k‖² = qᵀk − ½‖k‖² − ½‖q‖²
- Aditiva: a(q,k) = w_vᵀ·tanh(W_q q + W_k k)
- FFN(x) = max(0, xW1+b1)W2 + b2; LayerNorm(x + Sublayer(x))
- Sinusoidal: PE(pos, 2i) = sin(pos/10000^{2i/d}); PE(pos, 2i+1) = cos(pos/10000^{2i/d})
- ViT: x_i^emb = W_patch·x_i + b; z_i = x_i^emb + e_i^pos; y = softmax(W_class·c_final + b)
- Máscara: `scores.masked_fill(mask==0, -1e9)` antes del softmax.
- Soft KD: L = (1−λ)·L_CE(ψ(Z_s), y) + λτ²·KL(ψ(Z_s/τ), ψ(Z_t/τ))
- Hard KD: L = ½·L_CE(ψ(Z_s), y) + ½·L_CE(ψ(Z_s), y_t), con y_t = argmax Z_t

**(d) Métodos:** Bahdanau attention, Nadaraya-Watson, scaled dot-product, MHA, Transformer encoder-decoder, sinusoidal y learnable PE, ViT, [CLS] token, DeiT (distillation token), Swin (shifted windows), BERT (mencionado), Knowledge Distillation (soft, hard, feature, relation, offline/online/self, cross-modal, GAN-based), RandAugment, Mixup, CutMix.

**(e) Ejercicios PyTorch:**
- `scaled_dot_product_attention(Q,K,V,mask)` con `torch.bmm` y máscara causal (`torch.tril`), verificada contra `F.scaled_dot_product_attention`.
- `MultiHeadAttention` desde cero (split y merge de heads con `view`/`transpose`).
- Bloque encoder Pre/Post-LN con FFN de 4·d.
- PE sinusoidal.
- **ViT mínimo:** PatchEmbed con Conv2d, `flatten(2).transpose(1,2)`, concatenar CLS, sumar pos-emb, `nn.TransformerEncoder`, cabeza sobre el token 0. Comprobar shapes (B, 257, 768).
- Nadaraya-Watson como atención con kernels.
- **Pérdida de distillation** soft con τ² y `F.kl_div(log_softmax(zs/τ), softmax(zt/τ), reduction='batchmean')`, y la versión hard.
- Feature distillation con conv 1×1.
- Visualizar la atención del CLS (relacionado con el Cap. 5).

---

## CAPÍTULO 9 — Audio

**(a) Secciones:**
1. Introduction: 1.1 Sound Fundamentals (Pitch Perception, Intensity and Loudness, Sampling and Quantization, Frames), 1.2 Sound Features.
2. Signal Domain Features: Time Domain; Frequency Domain (STFT, Spectrogram).
3. Why Mel Spectrograms Perform Better: Perceptual Motivation, Mel Scale Construction, Decibel Scale (TorchAudio, Core Components, Mel Filterbank), Performance Advantages.
4. Feature Optimization and Augmentation: MFCC, SpecAugment, Time Shift, Pitch Shift, Time Stretch, Noise Injection.
5. ASR: CNN+RNN con CTC, Seq2Seq, Alignment Problem, CTC, WER, Applications Summary.
6. Annex: Standard Processing Pipeline, el parámetro n_fft, hiperparámetros del Mel.

**(b) Conceptos**
- **Física:** onda mecánica con regiones de compresión y rarefacción. Propiedades: frecuencia (Hz, tono), amplitud (volumen), fase y timbre (armónicos y envolvente).
- **Percepción:** el tono se percibe de forma logarítmica (octava = ×2; La4 = 440 Hz). El volumen, en dB; según el texto, +3 dB = doble intensidad. Umbral de audición 0 dB (20 μPa) y de dolor 120-140 dB.
- **Digitalización:**
  - Muestreo; frecuencia de Nyquist = fs/2 (CD: 44.1 kHz → 22.05 kHz).
  - Cuantización por bit depth (16 bits CD, 24 bits hi-res).
  - Frames de 2^k muestras (256-8192).
- **Features:** bajo nivel (envolvente, ZCR), nivel medio (pitch), alto nivel (género).
  - Dominio temporal: Amplitude Envelope (sensible a outliers), RMS energy, Zero-Crossing Rate (voz sonora vs. sorda).
  - Dominio frecuencial: DFT (los complejos codifican amplitud y fase), Band Energy Ratio, Spectral Centroid, Spectral Flux.
- **STFT:** ventanas solapadas (Hann, Hamming, gaussiana) + DFT por frame, apiladas en el tiempo. Trade-off tiempo-frecuencia.
- **Espectrograma:** magnitud de la STFT, normalmente en log/dB.
- **Mel:** somos más sensibles en bajas frecuencias.
  - Construcción del banco de filtros: mel(fmin), mel(fmax) → puntos equiespaciados en mel → de vuelta a Hz → filtros triangulares → se aplican al espectrograma de potencia.
  - Ventajas: perceptual, generaliza mejor, menos bins (128), en GPU y diferenciable.
- **MFCC:** DCT sobre el mel; estándar clásico en ASR.
- **Augmentation:** SpecAugment (máscaras de frecuencia y de tiempo), time shift (`torch.roll`), pitch shift, time stretch (sobre el espectrograma complejo), ruido gaussiano.
- **ASR:** acoustic model + language model + decoding.
  - **CNN+RNN+CTC** (Deep Speech de Baidu).
  - **Seq2Seq con atención** (Listen, Attend and Spell de Google).
  - **Problema de alineamiento:** p. ej. 1600 frames frente a 5 caracteres.
  - **CTC:** carácter blank "-"; se colapsan repeticiones y se eliminan blanks. En entrenamiento se suma sobre todos los alineamientos; en decodificación, greedy o beam search (top-K).
  - **WER.**
- **Aplicaciones:** clasificación (UrbanSound8K), detección de eventos (corazón, fallos industriales), tagging musical multi-label, speaker / emotion recognition, ASR y TTS.
- **n_fft:** pequeño da mejor resolución temporal; grande, mejor resolución frecuencial. Griffin-Lim para invertir. Hiperparámetros: fmin, fmax, n_mels (altura), n_fft, hop_length (anchura = muestras / hop).

**(c) Fórmulas**
- DFT: X_k = Σ_{n=0}^{N−1} x_n·e^{−2πikn/N}; |X_k| es la magnitud; x(t) = A·sin(ωt + φ)
- Mel: m = 2595·log10(1 + f/700)
- dB: L_dB = 20·log10(P/P0), con P0 = 20 μPa
- Nyquist: f_max = fs/2
- WER = (S + D + I)/N. Ejemplo: 2 sustituciones / 5 palabras = 40%.
- Anchura del espectrograma ≈ N_muestras / hop_length

**(d) Métodos:** STFT, (Mel-)Spectrogram, filtros triangulares, AmplitudeToDB, MFCC, SpecAugment, Time / Pitch shift, TimeStretch, Noise injection, Deep Speech (CNN+RNN+CTC), LAS, CTC loss, greedy y beam search, WER, Griffin-Lim, TorchAudio (`MelSpectrogram`, `MelScale`, `Spectrogram`, `MFCC`, `Resample`, `TimeMasking`, `FrequencyMasking`, `TimeStretch`).

**(e) Ejercicios PyTorch:**
- Pipeline `nn.Sequential(Resample, MelSpectrogram, AmplitudeToDB)` + SpecAugment solo en entrenamiento.
- Funciones a mano: ZCR, RMS, envolvente por frames (`unfold`), conversión Hz↔mel, banco de filtros mel triangular (comparar con `torchaudio.functional.melscale_fbanks`), DFT manual frente a `torch.fft.fft` / `torch.stft`, spectral centroid.
- Máscaras de tiempo y frecuencia manuales.
- Clasificador CNN sobre mel (estilo UrbanSound8K).
- Modelo CNN+GRU con `nn.CTCLoss` (atención a las shapes T×B×C con `log_softmax` y a input/target lengths).
- **Decodificación greedy de CTC** (colapsar y quitar blanks) y beam search sencillo.
- **WER** con distancia de Levenshtein.

---

## PAPERS (complementarios)

- **Paper 4 — *Deep Inside Convolutional Networks*** (Simonyan, Vedaldi, Zisserman, 2014). Ya está integrado en el Cap. 5.
  - Red usada: conv64-conv256×4-full4096×2-full1000, con 39.7% / 17.7% de error top-1 / top-5 en ILSVRC-2013.
  - Implementable: class model visualisation sobre el logit (no el softmax) y saliency con max sobre canales.
- **Paper 10 — I-JEPA** (*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*).
  - Aprendizaje auto-supervisado que predice en el **espacio de embeddings**, no en píxeles. No usa augmentations manuales.
  - Un bloque de contexto predice las representaciones de varios bloques objetivo. Las representaciones objetivo salen de un encoder objetivo actualizado por **EMA**, lo que evita el colapso.
  - Enmascarado multibloque: bloques objetivo grandes y contexto con amplia cobertura espacial. Encoder y predictor son ViT.
  - Supera a MAE y data2vec en linear probing sobre ImageNet y se acerca a los métodos de invariancia. Va bien con pocas etiquetas y en tareas locales (conteo, profundidad). Necesita menos cómputo.
  - Implementable: muestreo de máscaras multibloque sobre la rejilla de patches, actualización EMA `θ_t ← mθ_t + (1−m)θ_s`, pérdida L2 en el espacio latente.

---

## Ideas para ejercicios de examen (las de mayor prioridad)

1. **Cap. 2:** IoU, NMS, anchors, RoI pooling y pérdida YOLO.
2. **Cap. 3:** conv transpuesta manual, U-Net (shapes y skips) y Gram / style / TV loss.
3. **Cap. 4:** Conv3d manual, Chamfer y F1@δ, scale-invariant loss, clasificador de point clouds invariante a permutación, early/late fusion.
4. **Cap. 5:** Saliency, SmoothGrad, IG, Grad-CAM con hooks, Guided Backprop con autograd.Function, Occlusion / RISE, loyalty AUC.
5. **Cap. 6:** FGSM, PGD (proyección ℓ∞), C&W, bucle de adversarial training, VAT.
6. **Cap. 7:** VAE (reparametrización + KL cerrada), β-VAE, CVAE, bucle GAN, WGAN-GP, MaskedConv PixelCNN, cuantizador VQ con straight-through, BNN / MC Dropout.
7. **Cap. 8:** atención escalada con máscara, MHA, ViT (patch embedding con Conv2d + CLS + pos-emb), pérdida KD con τ².
8. **Cap. 9:** pipeline mel con TorchAudio, banco de filtros mel, CTC greedy decode, WER.

Posibles erratas o inconsistencias de los PDFs que conviene confirmar con el profesor:
- La CE del Cap. 2 no lleva signo menos.
- El signo del segundo término de la scale-invariant loss (Cap. 4).
- El valor absoluto en la fórmula de Shapley (Cap. 5).
- La notación de β en SlowFast (Cap. 4).
- La GAN escrita como "1 − log d(g(z))" (Cap. 7).
- La fórmula del Kernel SHAP, mal formateada en la extracción.
- "JopSkipJump" debería ser HopSkipJump (Cap. 6).
