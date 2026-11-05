# Soluciones de los ejercicios teóricos de los apuntes

Respuestas de los ejercicios de explicar, demostrar o calcular a mano, numerados como en
los apuntes (versión del 12/10/2026). Los ejercicios de código están en
[EJERCICIOS_APUNTES.md](EJERCICIOS_APUNTES.md). Estas respuestas las he redactado yo: el
profesor no las ha revisado.

> ⚠️ Dudas que conviene confirmar con el profesor: [T1 Ej. 5](#tema-1--fundamentos-de-cnn)
> y [T3 Ej. 11](#tema-3--segmentación-y-style-transfer).

---

## Tema 1 · Fundamentos de CNN

**Ej. 1. Tabla de arquitecturas**

| Arquitectura | Año | Innovación clave | Estructura |
|---|---|---|---|
| AlexNet | 2012 | ReLU, dropout y *data augmentation* a gran escala | 5 conv + 3 FC |
| VGGNet | 2014 | Apilar filtros 3×3: mismo campo receptivo con menos parámetros | Bloques homogéneos, 16-19 capas |
| GoogLeNet / Inception v1 | 2014 | Módulo Inception con conv 1×1 para reducir canales | Ramas paralelas, GAP, clasificadores auxiliares |
| ResNet | 2015 | Conexiones residuales contra la degradación | Bloques `F(x) + x`, más de 100 capas |
| DenseNet | 2017 | Cada capa recibe todas las anteriores | Bloques densos concatenados, reutiliza características |

**Ej. 3. Aritmética.** `floor((224 + 6 − 6 − 1) / 2) + 1 = floor(223/2) + 1 = 112`. La salida es **(64, 112, 112)**.

**Ej. 4. Gradiente en ResNet.** Con `y = F(x) + x`:

`∂L/∂x = ∂L/∂y · (∂F/∂x + I)`

El término `+1` (la identidad `I`) da un camino directo para el gradiente: aunque `∂F/∂x ≈ 0`, sigue llegando `∂L/∂y`. En una red sin atajos, el gradiente es un producto de muchas jacobianas que puede tender a 0. Además, a un bloque le basta con `F ≈ 0` para aprender la identidad, lo que facilita la optimización.

**Ej. 5. Campo receptivo.** Con `RF₀ = 1` (la entrada) y `RF_L = RF_{L−1} + 2`:

| Capa | 0 (entrada) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| RF | 1 | 3 | 5 | 7 | 9 | **11** |

⚠️ El ejemplo de los apuntes llama «Layer 1» a la imagen de entrada (`RF₁ = 1`). Con esa numeración, «Layer 5» daría 9. El enunciado habla de cinco capas convolucionales y de la fórmula con `RF₀ = 1`, por lo que aquí la respuesta es 11.

**Ej. 6. Transferencia a rayos X.** Hay pocos datos (400) y un dominio distinto de ImageNet: es el caso difícil de la guía. Estrategia:

1. No reentrenar todo: 400 imágenes llevarían al sobreajuste.
2. Usar el backbone como extractor, tomando características de capas **intermedias o tempranas** (las finales son específicas de ImageNet), y entrenar una cabeza pequeña.
3. Con buena validación, descongelar solo los últimos bloques con un *learning rate* bajo.
4. Aplicar *data augmentation* adecuada al dominio y regularización. Si no basta, valorar *few-shot*.

---

## Tema 2 · Detección de objetos

**Ej. 2. Ineficiencia.** Con `c` cajas por píxel hay `c·H·W` propuestas. Con `c = 10` en 224×224 son 501 760 pasadas por el clasificador, casi todas redundantes o de fondo.

**Ej. 6. Número de anchors.** Las combinaciones con `s₁` o `r₁` son `n + m − 1` por píxel, luego el total es `(n + m − 1)·H·W`.

**Ej. 8. Cuello de botella.** R-CNN hace `K` pasadas por la CNN sin compartir cálculos. Fast R-CNN hace una sola pasada y aplica RoI pooling. El código está en `apuntes.py` (`RCNN`, `FastRCNN`).

**Ej. 9. Variantes.** Las características tras el RoI pooling pueden alimentar un *Random Forest* (regresión de cajas) o un *Gradient Boosting* (clasificación).

- **Backpropagation:** estos modelos no son diferenciables, así que no se propaga gradiente al backbone. El entrenamiento pasa a ser por etapas, como en la R-CNN original con SVM.
- **Tiempo:** se añade una fase de extracción y ajuste, y se pierde el entrenamiento conjunto de extremo a extremo. A cambio, pueden funcionar bien con pocos datos.

**Ej. 10. Raíz en YOLO.** Compara el mismo error `Δ` en una caja pequeña y en una grande. Aproximando: `√(w + Δ) − √w ≈ Δ / (2√w)`, de modo que el error al cuadrado es `≈ Δ² / (4w)`.

La penalización es inversamente proporcional a `w`: 5 píxeles en una caja de 10 pesan mucho más que 5 en una de 200. Sin la raíz, ambos pesarían igual, aunque en la caja pequeña el error relativo es mucho mayor.

**Ej. 11. Una etapa frente a dos**

| | Una etapa (YOLO) | Dos etapas (Faster R-CNN) |
|---|---|---|
| Velocidad | Alta, apta para tiempo real | Menor |
| Precisión | Históricamente menor, sobre todo en objetos pequeños | Mayor en la localización |
| Diseño | Red única, entrenamiento sencillo | RPN + cabeza, más complejo |
| Uso preferente | Conducción, vídeo, dispositivos *edge* | Imagen médica, inspección de calidad, análisis fuera de línea |

---

## Tema 3 · Segmentación y *style transfer*

**Ej. 1. Complejidad.** Son `H·W` *forwards* y cada uno procesa `P²` píxeles: `O(H·W·P²)`, sin compartir cálculos entre parches que se solapan casi por completo. Por eso es intratable en tiempo real.

**Ej. 2. Forma de salida.** `H_out = (H_in − 1)·s + k_h − 2p`. Para el caso pedido: `6·2 + 4 − 2 = 14`, es decir, **14×14**.

**Ej. 5. Bed of Nails.** Salida de `(2−1)·2 + 2 = 4` por lado. Cada entrada coloca `valor·K` a partir de `(2i, 2j)`:

```
a·K en (0,0)   b·K en (0,2)   c·K en (2,0)   d·K en (2,2)

a 0 b 0
0 0 0 0      ← los bloques no se solapan y K solo tiene un 1 arriba a la izquierda
c 0 d 0
0 0 0 0
```

Es exactamente Bed of Nails con factor 2. ∎

**Ej. 6. Nearest Neighbor.** Igual, pero con `K` de unos: cada entrada rellena su bloque 2×2.

```
a a b b
a a b b
c c d d
c c d d
```

Es exactamente Nearest Neighbor con factor 2. ∎

**Ej. 9.3. Memoria de las *skip connections*.** Los mapas del *encoder* deben mantenerse hasta que los usa el *decoder*, lo que eleva el pico de memoria en el *forward*. La concatenación duplica además los canales de entrada del *decoder*, que necesita más parámetros y más activaciones guardadas para el *backward*. Sin *skips*, esos mapas también se guardan si hacen falta para el gradiente, pero el *decoder* es más ligero.

**Ej. 11. Gram y coseno.** Sea `x_i` el canal `i` aplanado. Entonces `G_ij = x_iᵀx_j / (C·H·W)`. Como `cos(x_i, x_j) = x_iᵀx_j / (‖x_i‖‖x_j‖)`, si `‖x_i‖ = ‖x_j‖ = 1` se cumple `x_iᵀx_j = cos(x_i, x_j)`. ∎

⚠️ Con la normalización `1/(C·H·W)` de los apuntes, la entrada es el coseno **dividido por esa constante**. Solo es exactamente el coseno si se omite el denominador.

**Ej. 14. Efectos esperados** (el enunciado pide experimentar: comprueba estas tendencias con tu código).

- **α/β grande:** domina el contenido. **α/β pequeño:** domina el estilo y se deforma la estructura.
- **Inicio con ruido:** converge más lento y conserva peor el contenido que iniciar con `x_C`.
- **γ = 0:** más artefactos de alta frecuencia. **γ = 10⁻³:** imagen espacialmente más suave.

**Ej. 15. Comparativa de tareas**

| Tarea | Entrada | Salida | Pérdida típica | Aplicaciones |
|---|---|---|---|---|
| Semántica | Imagen | Clase por píxel, sin distinguir instancias | CE por píxel | Conducción, medicina |
| Instancias | Imagen | Caja, clase y máscara por objeto | Detección + BCE de máscara | Robótica, conteo |
| Panóptica | Imagen | Clase e identificador por píxel (*things* y *stuff*) | Combinación de las dos anteriores | Percepción completa de escena |
| *Keypoints* | Imagen | Coordenadas o *heatmaps* de puntos | MSE sobre *heatmaps* | Pose humana, deporte |

---

## Tema 4 · 3D y vídeo

**Ej. 1. Conv3D.** `T: (16−3)/1 + 1 = 14`. `H, W: floor((112−7)/2) + 1 = 53`. Con el orden del enunciado: **(14, 64, 53, 53)**.

**Ej. 2. Conv3D a mano.** Entrada de `(T=3, C=1, H=5, W=5)` y kernel 2×2×2, con `k[0] = [[1,0],[0,0]]` y `k[1] = [[0,0],[0,−1]]`. La salida es `(2, 1, 4, 4)` y la primera posición vale:

`y[0,0,0] = x[0,0,0] − x[1,1,1]`

Cada salida combina un cubo de 2×2×2 multiplicando elemento a elemento y sumando. `conv3d_manual` en `geometry.py` permite comprobarlo.

**Ej. 3. Inflado.** `W₃D[t] = W₂D / T` para todo `t ∈ {1..5}`. En un vídeo estático con todos los fotogramas iguales a `I`:

`Σ_t (W₂D / T) * I = T · (W₂D * I) / T = W₂D * I`

La respuesta coincide con la del filtro 2D. ∎

**Ej. 4. Pérdida invariante a escala.** Sea `d_ij = log y_ij − log ŷ_ij` y `n = HW`. Con `ŷ' = cŷ`, `d' = d − a` siendo `a = log c`. Para `λ = 1`, `L = (1/n)Σd² − (1/n²)(Σd)²`, que es la **varianza** de `d`, y restar una constante no cambia la varianza. Desarrollando:

- `(1/n)Σ(d−a)² = (1/n)Σd² − 2a·d̄ + a²`
- `(1/n²)(Σ(d−a))² = (d̄ − a)² = d̄² − 2a·d̄ + a²`

Al restar se cancelan `−2a·d̄` y `a²`, y queda `L(y, ŷ)`. ∎

**Ej. 5. Distancia coseno** (`1 − y·ŷ`). Alineados: **0**. Ortogonales: **1**. Opuestos: **2**.

**Ej. 6. Memoria.**

- Vóxeles: `(100/0.1)³ = 10⁹`.
- Puntos: `4π·10²·100 ≈ 125 664`.

Los vóxeles gastan memoria en espacio vacío (crecen con `resolución³`), pero son una rejilla regular en la que se puede convolucionar. La nube es unas 8000 veces más pequeña, aunque no tiene orden ni conectividad.

**Ej. 7. Chamfer.** `d_C(P,Q) = Σ_{p∈P} min_{q∈Q} ‖p−q‖² + Σ_{q∈Q} min_{p∈P} ‖p−q‖²`.

1. Es **simétrica**: intercambiar `P` y `Q` solo intercambia los dos sumandos.
2. Si `P = Q`, cada punto tiene a sí mismo como vecino a distancia 0, así que `d_C = 0`.
3. **Valores atípicos:** el mínimo no se puede «repartir». Un punto muy lejano aporta su distancia al cuadrado completa, aunque todos los demás encajen perfectamente.

**Ej. 9. Nubes de puntos.**

1. Un MLP sobre `(x₁,y₁,z₁,x₂,…)` da un peso distinto a cada posición. Intercambiar dos puntos intercambia sus coordenadas y, en general, `w₁x₁ + w₂x₂ ≠ w₁x₂ + w₂x₁` si `w₁ ≠ w₂`. No es invariante a permutaciones.
2. Con un MLP compartido, `g(P) = max_i h(p_i)`. `h` no depende de la posición y el máximo es conmutativo, de modo que cualquier permutación produce el mismo resultado. ∎

**Ej. 10. Comparativa de vídeo.** Código en `video.py`.

| Modelo | Parámetros | Movimiento de bajo nivel |
|---|---|---|
| *Late fusion* con *pooling* | Los de la CNN 2D, sin coste temporal | Nulo: promedia sin ver el orden |
| *Early fusion* | Solo cambia la primera capa (canales `T·C`) | Limitado: colapsa el tiempo en la primera capa |
| CNN 3D | Más por capa: kernels con dimensión temporal | Alto: extrae patrones espaciotemporales capa a capa |

---

## Tema 5 · Explicabilidad

**Ej. 1. Importancia.** Sirve para generar confianza y detectar sesgos o correlaciones espurias. Ejemplos: en medicina, comprobar que el modelo mira la lesión y no una marca del hospital; en justicia, auditar sesgos y justificar decisiones; en conducción autónoma, analizar fallos de seguridad.

**Ej. 2. Grad-CAM generaliza CAM.** Con `f_c = b_c + Σ_k w_k^c F_k` y `F_k = (1/Z) Σ_ij A_ij^k`:

`∂f_c/∂A_ij^k = w_k^c / Z`

Por tanto, `α_k^c = (1/Z) Σ_ij w_k^c / Z = (1/Z) · Z · w_k^c / Z = w_k^c / Z`. Es decir, `α_k^c ∝ w_k^c` con constante `1/Z`. ∎

**Ej. 3. Saturación del gradiente.** `f(15) = 15 − 5 = 10`. Para `x > 10`, `f'(x) = 1 − 1 = 0`: la *saliency* dice que `x` no importa, aunque la salida es 10.

Con Integrated Gradients y `x' = 0`: `IG = (15 − 0) ∫₀¹ f'(15α) dα`. Como `f'(15α) = 1` si `15α < 10`, es decir, `α < 2/3`, y 0 en otro caso:

`IG = 15 · 2/3 = 10 = f(15) − f(0)`

Atribuye toda la salida, tal como exige la propiedad de completitud.

**Ej. 4. Complejidad.**

- **Shapley exacto:** cada píxel promedia sobre `2^{N−1}` coaliciones, lo que da `O(N·2^N)` en total (o `O(2^N)` evaluaciones si se reutilizan).
- **Occlusion:** `O(N)` evaluaciones con stride 1 (menos con parches y stride mayores).

Como `2^N` es inviable, SHAP debe aproximar con **Monte Carlo** y usar superpíxeles en lugar de píxeles.

**Ej. 22. El problema OOD.** Una imagen con píxeles a cero no pertenece a la distribución de entrenamiento: la salida puede caer por el cambio de distribución y no porque esos píxeles fueran importantes. Además, se asume independencia entre las características. Es una de las motivaciones de ROAR.

---

## Tema 6 · Ataques adversarios

**Ej. 1. Hipótesis lineal.** En alta dimensión, muchos cambios pequeños se acumulan aunque el modelo sea localmente lineal. Explicación complementaria: las características «no robustas» que generalizan pero son imperceptibles para las personas.

Para `f(x) = wᵀx + b`, el cambio máximo con `‖η‖∞ ≤ ε` se da en `η = ε·sign(w)` y vale `ε‖w‖₁`. Crece linealmente con `n` si la magnitud media de los pesos, `m = (1/n)Σ|w_i|`, se mantiene constante, porque entonces el cambio es `ε·m·n`.

**Ej. 2. Modelos de amenaza.**

- **Caja blanca:** se conoce todo, gradientes incluidos.
- **Caja gris:** conocimiento parcial, por ejemplo la arquitectura.
- **Caja negra:** solo consultas.

La **transferibilidad** significa que un adversario creado sobre un modelo sustituto suele engañar a otro, porque los modelos aprenden fronteras parecidas. Por eso se puede atacar un modelo propietario sin acceder a él.

**Ej. 3. FGSM y PGD.** FGSM es un único paso de ascenso linealizado, de tamaño máximo `ε`. PGD itera pasos pequeños y reevalúa el gradiente, de modo que sigue mejor la superficie no lineal de la pérdida, y de ahí que sea más fuerte. `P_ε` proyecta de vuelta a la bola tras cada paso. En ℓ∞ es un recorte por coordenada: `x + clamp(x' − x, −ε, ε)`.

**Ej. 4. Cambio de variable en C&W.** Como `tanh ∈ (−1, 1)`, `½(tanh(w) + 1) ∈ (0, 1)` para cualquier `w`. La restricción de caja se cumple por construcción y se puede optimizar `w` sin restricciones con Adam, sin recortes que distorsionen el gradiente.

---

## Tema 7 · IA generativa

**Ej. 1. Factorización 2×2.** `p(x) = p(x₁)·p(x₂|x₁)·p(x₃|x₁,x₂)·p(x₄|x₁,x₂,x₃)`. Cada píxel usa como contexto todos los anteriores en orden de lectura.

**Ej. 2. Máscaras 5×5.** Orden de lectura; el centro es la fila 3, columna 3.

```
Máscara A     Máscara B
1 1 1 1 1     1 1 1 1 1
1 1 1 1 1     1 1 1 1 1
1 1 0 0 0     1 1 1 0 0
0 0 0 0 0     0 0 0 0 0
0 0 0 0 0     0 0 0 0 0
```

La A se usa en la primera capa: si incluyera el centro, el píxel se vería a sí mismo y la predicción sería trivial. La B se usa después: el centro ya no contiene el píxel, sino características que solo dependen de píxeles anteriores, gracias a la A.

**Ej. 3. Entrenamiento y generación.** En entrenamiento se conoce toda la imagen y las máscaras garantizan la causalidad, por lo que todas las posiciones se predicen en un solo *forward*. Al generar, cada píxel necesita los anteriores, que aún no existen: hay que muestrear de uno en uno, con `H·W·C` pasos.

**Ej. 4. RGB.** `p(R,G,B|ctx) = p(R|ctx)·p(G|ctx,R)·p(B|ctx,R,G)`. Rojo: ningún canal del píxel actual. Verde: el rojo. Azul: rojo y verde.

**Ej. 5. Autoencoder.** El *encoder* comprime la entrada, el latente es la representación compacta y el *decoder* reconstruye. El objetivo de la reconstrucción es **la propia entrada** `x` (la imagen limpia, en la variante *denoising*).

**Ej. 6. Generación.** El latente de un AE no tiene estructura: un vector arbitrario puede caer en una zona «vacía» y decodificarse como ruido. El término KL del VAE acerca el posterior a `N(0, I)`, lo que da un espacio continuo. Para generar: `z ~ N(0, I)` y `x = decoder(z)`.

**Ej. 7. Variantes.**

| Variante | Idea | Caso de uso |
|---|---|---|
| *Denoising* AE | Reconstruye la imagen limpia a partir de una ruidosa | Limpieza de imágenes |
| *Sparse* AE | Penaliza activaciones para que pocas se activen | Características interpretables |
| β-VAE | Aumenta el peso de la KL (`β > 1`) | Factores latentes desenredados |
| VQ-VAE | Latente discreto mediante un *codebook* | Compresión, prior autorregresivo |

**Ej. 8. GAN.** G genera imágenes para engañar a D, y D distingue reales de falsas. Se actualizan alternadamente porque el objetivo es un juego *minimax* con objetivos opuestos: cada red se entrena frente a la otra congelada. Objetivos de D: **1** para las reales y **0** para las generadas.

**Ej. 9. Colapso de modos.** G genera solo unos pocos modos. Ejemplo: en MNIST, solo produce muy bien el dígito 1. Cada imagen es realista, por lo que evaluarlas por separado no lo detecta: hay que medir la **diversidad** del conjunto.

**Ej. 10. Inferencia.** `z ~ p(z)` y `x = G(z)`; solo hace falta el **generador**. Una cGAN añade la condición, `x = G(z, c)`, para elegir, por ejemplo, la clase.

**Ej. 11. Condicional.** `p(x)` modela todas las imágenes y `p(x|c)` las que cumplen `c`. Para una misma condición se obtienen imágenes distintas variando el ruido o el latente `z`. Otra condición útil: un texto, un boceto o un mapa de segmentación.

---

## Tema 8 · Atención y Transformers

**Ej. 1. Por qué `√d`.**

1. `E[q·k] = Σ E[q_i]E[k_i] = 0`, por independencia. Además, `Var(q_i k_i) = E[q_i²]E[k_i²] − 0 = 1` y los términos son independientes entre sí, de modo que `Var(q·k) = d`. ∎
2. `Var(q·k/√d) = d/d = 1`. Sin escalar, la magnitud crece como `√d`, el softmax se satura (casi *one-hot*) y los gradientes se vuelven muy pequeños.

**Ej. 2. Codificación posicional.** Con `X' = PX`, siendo `P` una permutación, `Q, K, V` se permutan igual y la salida es `P·Attn(X)`: es equivariante. Sin posiciones, un ViT apenas distingue una imagen de la misma con los parches barajados (con *CLS* o GAP, el resultado es prácticamente igual).

**Ej. 3. CNN y ViT.** Las CNN tienen sesgos fuertes (localidad y equivarianza a la traslación), por lo que necesitan menos datos. Los ViT tienen pocos sesgos y atención global desde el inicio, lo que los hace muy flexibles pero dependientes de muchos datos. Una buena receta (DeiT: *augmentation*, regularización, destilación) reduce mucho esa desventaja.

**Ej. 4. CLS y GAP.** GAP promedia todos los parches con el mismo peso. CLS es un *token* aprendido: en cada capa actúa como *query* y atiende a los parches, con pesos que dependen de la imagen, de modo que agrega información global de forma dinámica.

**Ej. 9. DeiT.** La **receta** de entrenamiento (*augmentation* y regularización fuertes) permite entrenar ViT sin datos masivos. La **variante destilada** es opcional: añade el *token* `x_dist`, supervisado por el profesor. Una CNN como profesor aporta sus sesgos inductivos (localidad), que son justo los que le faltan al ViT.

**Ej. 11. SW-MSA.** La atención se calcula dentro de ventanas sin solape, con un coste lineal en el número de parches. Pero así las ventanas no se comunican. Por eso los bloques alternan W-MSA y SW-MSA: se desplazan las ventanas `⌊M/2⌋` parches, de modo que las nuevas abarcan los bordes de las anteriores. El desplazamiento cíclico con máscara mantiene el número de ventanas y hace eficiente el cálculo.

---

## Tema 9 · Audio

**Ej. 1.** `L = 3·16000 = 48000`, luego la forma es **(16, 2, 48000)**. Nyquist: **8 kHz**.

**Ej. 2.** 25 ms = **400** muestras y 10 ms = **160**. La ventana controla la resolución en frecuencia frente al tiempo; el salto, la resolución temporal y el número de fotogramas.

**Ej. 3.** **(1, 80, 98)**. Mel imita la percepción humana, con más resolución en frecuencias bajas y menos dimensiones. El logaritmo comprime el rango dinámico, en línea con la percepción de la sonoridad.

**Ej. 4.**

| Tarea | Pitch shift | Time stretch |
|---|---|---|
| Sonidos ambientales | Normalmente conserva (moderado) | Normalmente conserva (moderado) |
| Estimación de tono | **No**: cambia la etiqueta | Conserva |
| Verificación de hablante | Riesgo: altera la identidad | Moderado |
| Eventos localizados | Conserva la clase | **Hay que ajustar** los tiempos de las etiquetas |

**Ej. 5.** Una CNN 2D funciona porque el espectrograma tiene patrones locales. Pero los ejes no son equivalentes: desplazar en frecuencia cambia el sonido (no es invariante como una traslación espacial), el tiempo tiene orden causal y un mismo sonido no «ocluye» a otro, sino que sus energías se suman.

**Ej. 6.** Clasificar un clip: se puede hacer *pooling* a un solo vector. Transcribir: hay que conservar el orden temporal, porque la salida es una secuencia.

**Ej. 7.** `c,c,∅,a,a,∅,t,t → c,∅,a,∅,t → cat`. Otro camino: `c,∅,∅,a,∅,t,t,t`.

**Ej. 8.** CTC marginaliza sobre todas las alineaciones monótonas y predice por fotograma, de modo que no necesita alineación explícita. La atención aprende los pesos sobre los fotogramas para cada *token* de salida. Diferencia en inferencia: CTC es paralelizable y permite decodificación sencilla por fotograma; el *decoder* de atención genera *token* a *token*, condicionado a los anteriores.

**Ej. 9.** *Greedy* elige lo mejor en cada paso y puede descartar un prefijo que después conduce a una secuencia más probable. *Beam search* mantiene `K` hipótesis. El coste de aumentar `K` es más tiempo y memoria, aproximadamente lineal en `K`.

**Ej. 10.** `N = 4`. Hay dos alineaciones mínimas de 2 ediciones: **WER = 2/4 = 0.5**.

- Borrar `small` e insertar `fast`: `a [small] red [fast] car` (1 borrado + 1 inserción).
- Sustituir `small → red` y `red → fast` (2 sustituciones).

---

## Apéndice A · Ingeniería de software

**Ej. 1 y 2. Docstrings y tipos**

```python
def scale(values: list[float], factor: float | int = 1) -> list[float]:
    """Escala una lista (Google).

    Args:
        values: valores.
        factor: factor de escala.

    Returns:
        Valores escalados.
    """
    # Sphinx:  :param values: valores.  :type values: list[float]  :return: ...  :rtype: list[float]
    # NumPy:   Parameters\n----------\nvalues : list[float]\n    Valores.\n\nReturns\n-------\nlist[float]
    return [value * factor for value in values]
```

**Ej. 3 y 4. Herramientas**

```bash
uv add --dev black isort mypy flake8 ruff pylint complexipy bandit
black . && isort --profile black . && mypy src
flake8 src && ruff check src && pylint src && complexipy src && bandit -r src
```

**Ej. 5. Complejidad cognitiva.** Sube con los saltos del flujo y el anidamiento: una función sin ramas puntúa 0; un `if` con un `for` anidado puntúa alrededor de 3; varios bucles y condiciones anidados con `break` superan fácilmente el umbral típico de 15. Compara tu estimación con `complexipy`.

**Ej. 6. Tests.** Los tests de este repositorio son ejemplos: `tests/fundamentals/test_f10_cnn.py` combina valores a mano, casos límite y comparación con PyTorch sobre entradas aleatorias.

**Ej. 7. Entorno**

```bash
uv init proyecto && cd proyecto
uv venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv add torch torchvision
```

**Ej. 8. Makefile**

```makefile
.PHONY: install format lint test clean all

install:
	uv sync
format:
	uv run isort --profile black src tests
	uv run black src tests
lint:
	uv run isort --profile black --check src tests
	uv run black --check src tests
	uv run mypy src
	uv run flake8 src
	uv run ruff check src
	uv run pylint src
	uv run complexipy src
	uv run bandit -r src
test:
	uv run pytest
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache

all: install format lint test clean
	@echo "All tasks completed!"
```
