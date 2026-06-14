# Estado del arte — Guerra Electrónica Cognitiva (guía detallada por modelo)

**Proyecto:** Cognitive Electronic Warfare System (TFM)
**Fecha de la revisión:** 2026-06-14
**Objetivo de publicación:** revista Q1 (IEEE TNNLS, Knowledge-Based Systems, Information Sciences, Expert Systems with Applications)

> **Método.** Esta guía se basa en lectura en profundidad de artículos de acceso abierto (PubMed Central, arXiv) y en abstracts/snippets verificados de fuentes tras paywall (Springer, ScienceDirect, MDPI). Cada caso de estudio indica su nivel de detalle (lectura completa vs. abstract) y, cuando una fuente devolvió contenido posiblemente inferido, se señala explícitamente. **Antes de citar formalmente en el TFM, re-verificar cada referencia en su fuente original** (las cifras provienen de la extracción automática del texto).

---

## 0. Marco conceptual: ¿qué es la GE cognitiva?

La guerra electrónica cognitiva (Cognitive EW) sustituye el paradigma clásico de **librería de amenazas** por un bucle cerrado **sentir → orientar → decidir → actuar → aprender** ejecutado a velocidad de máquina. El sistema observa el entorno electromagnético, identifica amenazas, sintetiza o selecciona contramedidas y, de forma crucial, **incorpora la experiencia acumulada** a decisiones futuras de manera autónoma (EMSOPEDIA, *Cognitive EW*).

El argumento que motiva todo el proyecto está bien establecido en la literatura: los sistemas de ataque electrónico convencionales dependen de una **taxonomía de formas de onda definida explícitamente** y de una **librería de identificación** preparada en planificación de misión, que selecciona contramedidas por *lookup* contra amenazas conocidas. Estos sistemas (i) no comprenden la *intención* de una señal novedosa, (ii) fallan ante amenazas "zero-day", y (iii) son insuficientes contra radares multifunción modernos con gran agilidad de RF y sin periodicidad de pulso estable. La GE cognitiva embebe ML en el núcleo del bucle para clasificar señales, detectar comportamiento anómalo y **sintetizar jamming optimizado dinámicamente** en lugar de seleccionarlo de una tabla fija (EMSOPEDIA, *Cognitive EW*).

Una tensión recurrente y directamente relevante para el TFM: la GE "reacciona en tiempo real o cuasi-tiempo real" y debe decidir "en fracciones diminutas de segundo", pero la literatura **rara vez reporta latencias de inferencia medidas**. Esto convierte la latencia documentada (<5 ms jamming, <1 ms ELINT, con media + p99 en hardware fijo) en una contribución diferenciadora.

---

## 1. Deep RL para jamming adaptativo

### 1.1 Planteamiento del campo

El problema se formula casi universalmente como un **proceso de decisión de Markov (MDP)** o, cuando el estado interno del adversario no es observable, un **POMDP**. El agente (jammer o radar cognitivo) observa una representación del entorno espectral, elige una acción (forma de onda, banda, potencia, técnica) y recibe una recompensa ligada a una métrica de efectividad (SJNR, tasa de error de símbolo, probabilidad de detección). El reto central es **aprender con pocas interacciones** en un entorno no estacionario y adversarial, donde el oponente también adapta su comportamiento.

Tres familias técnicas dominan:

- **Value-based (DQN y variantes):** Double-DQN y *Dueling* Double-DQN (D3QN) con *experience replay* priorizado y *fixed Q-targets*. Apropiadas para espacios de acción discretos moderados.
- **Maximum-entropy / actor-critic (SAC, PPO):** mejor exploración y estabilidad; necesarias cuando el espacio de acción crece o es continuo.
- **Híbridos teoría de juegos + RL:** tendencia ascendente 2024-25 para manejar incertidumbre de tipo de radar y comportamiento estocástico que el RL puro maneja peor.

Un problema práctico transversal es el **crecimiento del espacio de acción**: combinar modulación × potencia × frecuencia × técnica genera miles de acciones, y los algoritmos value-based dejan de converger. Las soluciones recientes adaptan arquitecturas de *embedding* de acciones (Wolpertinger) para escalar.

### 1.2 Caso de estudio A — Jamming inteligente con Improved-SAC + Wolpertinger
*(lectura completa: Sensors / PMC9601320)*

- **Problema:** un jammer aprende a interrumpir un enlace de comunicaciones cuyas partes adaptan modulación, potencia y frecuencia (estrategia anti-jamming fija: subir potencia → cambiar frecuencia → cambiar modulación). Escenario no cooperativo, pocas interacciones.
- **Formulación MDP:** el estado combina los parámetros de comunicación observados con la acción de jamming previa (`Sₜ = Sₜ* + aₜ`). La acción es un vector discreto **modulación × potencia × frecuencia**; el espacio máximo probado es **1.200 acciones = 4 × 30 × 10** (modulaciones × niveles de potencia × puntos de frecuencia). La recompensa es a trozos: cuando el SER supera el umbral `X` se da recompensa alta con un **factor de alineación de frecuencia `F1`** (1 si la frecuencia de jamming coincide con la de las partes) más un **factor de recompensa** y un **penalizador de potencia** (evita que el agente persiga ciegamente la potencia máxima); por debajo del umbral, se usa la distancia en frecuencia como guía.
- **Algoritmo y arquitectura:** **Soft Actor-Critic** con una **arquitectura Wolpertinger mejorada** que produce una *proto-action* continua, la mapea al espacio discreto vía **K-NN** y expande el conjunto candidato con combinaciones de frecuencia. Red de política con 4 capas ocultas (128→256→512→128) y cuatro Q-networks idénticas, ReLU.
- **Entrenamiento:** replay de 10⁶ muestras; LR actor/critic = 0,0015; γ = 0,1 (prioriza convergencia inmediata); τ = 0,005; batch 100; coeficiente de entropía dinámico decreciente; canal AWGN a SNR 20 dB. Co-simulación MATLAB↔Python.
- **Resultados (escalabilidad del espacio de acción):**

  | Nº acciones | Improved-SAC (media) | SAC base | Q-learning / DQN |
  |---|---|---|---|
  | 20 | 98,29 % (100 % en ronda 9) | — | inferior |
  | 150 | 93,85 % (90 % en ronda 29) | 83,26 % | inferior |
  | 600 | 91,95 % | 85,34 % | inferior |
  | 1.200 | 87,83 % | 78,83 % | **no convergen** |

  Demuestra que el acoplamiento Wolpertinger permite escalar a espacios de acción grandes donde DQN colapsa.
- **Limitaciones declaradas:** la librería de acciones de jamming es aún pequeña para complejidad real; no modela ocupación de canal; la recompensa basada en SER se reconoce subóptima.

### 1.3 Caso de estudio B — Diseño de forma de onda anti-jamming con D3QN
*(lectura completa: Sensors / PMC9692253)*

Aunque es la **perspectiva del radar** (anti-jamming), ilustra con detalle la formulación RL que el modelo 1 del TFM usará en espejo (jammer vs. radar).

- **Problema:** radar aerotransportado de banda X (9,5 GHz, 100 MHz de ancho) diseña en tiempo real su forma de onda transmitida para maximizar SJNR ante jamming de lóbulo principal, clutter no gaussiano y ruido.
- **Formulación MDP:** estado = distribución de potencia del jammer en 5 sub-bandas × 6 niveles → 6⁵ = 7.776 estados; acción = forma de onda del radar con la misma discretización (7.776 acciones); recompensa ∝ SJNR; γ = 0,9.
- **Algoritmo:** **Dueling Double DQN (D3QN)** con *fixed Q-targets* y *prioritized experience replay* (prioridad pᵢ = |δᵢ| + ε). Síntesis de forma de onda transmisible vía método de transformación iterativa (ITM).
- **Resultados (Ps = 5 W, Pⱼ = 10 W salvo indicado):**

  | Forma de onda | SJNR | Prob. detección |
  |---|---|---|
  | LFM convencional | 12,8 dB | 53 % |
  | RL (policy iteration) | 13,7 dB | 70 % |
  | **D3QN** | **15,8 dB** | **97 %** |

  Mejora cabecera: **+2,08 dB de SJNR y +26,79 % de prob. de detección vs. RL**, y **+3,03 dB / +44,25 % vs. LFM**. A Pⱼ = 30 W, Pd = 99,84 % (D3QN) vs. 92,27 % (RL) vs. 83,95 % (LFM).
- **Limitaciones declaradas:** espacio discreto (se sugiere formulación continua); recompensa monolítica (solo SJNR); modelo de jammer idealizado (observabilidad perfecta, sin retardos); validación solo en simulación con parámetros sintéticos fijos; **no se reporta tiempo de convergencia ni latencia de inferencia**; escenarios multi-jammer no explorados.

### 1.4 Hueco / oportunidad para el TFM (modelo 1)
- **Latencia <5 ms** documentada (media + p99) en hardware fijo — prácticamente ausente en la literatura.
- **Espacio de acción discreto-compuesto** con técnicas EW reales (noise, DRFM, cross-eye, VGPO, RGPO) y adaptación a cambios de waveform/ECCM, frente a los espacios "potencia × frecuencia" abstractos habituales.
- Manejo explícito de **observabilidad parcial** (POMDP con historial), donde muchos trabajos asumen observación perfecta.

---

## 2. Temporal CNN para clasificación ELINT

### 2.1 Planteamiento del campo

La entrada son secuencias de **PDW** (Pulse Descriptor Words): PRI (intervalo de repetición), PW/PD (ancho de pulso), RF (frecuencia), amplitud y periodo de barrido de antena. Tres familias compiten:

- **CNN 1D (con o sin atención):** extractor automático de características sobre parámetros o muestras crudas; baseline robusto y ligero.
- **Recurrentes (RNN/LSTM, CBRNN):** modelan dependencias temporales del tren de pulsos; útiles cuando el patrón inter-pulso es discriminante.
- **Transformer + GCN:** lo más reciente — atención para la dimensión temporal y grafos para las relaciones estructurales entre parámetros.

Un giro metodológico relevante: codificar la secuencia PDW como **imagen** y aplicar **segmentación (U-Net)** para resolver *deinterleaving* (separar pulsos entremezclados de varios emisores) y reconocimiento conjuntamente. La motivación recurrente es que los radares ágiles modernos hacen que "PDW + PRI por sí solos sean insuficientes".

### 2.2 Caso de estudio A — CNN multi-stream para 18 clases de radar
*(lectura completa: Sensors / PMC8707803)*

- **Problema:** identificación de emisor específico entre **18 clases** de radar con rangos de parámetros **fuertemente solapados** (de ahí la necesidad de análisis multi-parámetro).
- **Entrada:** cuatro parámetros (PRI ms, PD µs, RF GHz, SP s) y, alternativamente, formas de onda crudas (hasta 335.544 muestras). Tres estrategias: parámetros post-detección, formas de onda crudas, y combinación multi-stream.
- **Arquitectura:** redes 1D-CNN de 17 capas (bloques Conv1D + BatchNorm + MaxPool con filtros 5→9→6, kernels 5/3/2), Dense + Dropout + Softmax. La variante **multi-stream** concatena tres redes paralelas (PRI, PD, TW). Optimizador Adam, LR = 5×10⁻⁵.
- **Resultados:** parámetro único rinde mal (PD: 5,6–16,7 %; PRI: hasta 78 %; TW: hasta 72 %). La red **multi-stream concatenada alcanza 72,2–100 %** (mejor caso 100 %, típico 90–99 %). Robustez a ruido aditivo alta (99,7–100 % hasta nivel 0,5), pero degrada ante interferencia señal-a-señal (de 99,1 % a 44,6 % entre niveles 0,1 y 0,9).
- **Dato clave para el TFM (latencia):** inferencia reportada de **0,015 s/muestra en una GTX 1060** (≈15 ms) para la red de parámetro único, y 0,046 s para la multi-stream. Es decir, **el objetivo <1 ms del TFM exige diseño/optimización deliberados** (red ligera, batching, hardware fijo) — un objetivo de contribución, no algo gratuito.
- **Limitaciones declaradas:** escalado manual de frecuencias por memoria; entrenamiento de hasta 24 h para señales crudas; sin comparación empírica con baselines de DL modernos; validación solo con señales sintéticas de simulador propio.

### 2.3 Caso de estudio B (deep dive) — LSTM específicas por atributo
*(lectura completa del PDF: arXiv 1911.07683 — artículo 06)*

- **Idea central:** en lugar de procesar conjuntamente todos los atributos del pulso, dedica **una pila de LSTM por atributo** y concatena sus salidas. Motivación: cada parámetro PDW (PRI, RF, PW, amplitud…) tiene dinámica temporal propia y mezclarlos diluye la señal discriminante.
- **Entrada y normalización:** secuencias de pulsos construidas desde PDWs (M atributos por pulso). Propone una **normalización por secuencia** que concatena z-score y min-max → `2·M` canales, resaltando patrones temporales útiles con independencia de la escala absoluta.
- **Arquitectura:** `2·M` capas LSTM apiladas (una rama por atributo), concatenadas a lo largo del *hidden size*, seguidas de capa FC + softmax sobre `C` clases. **Dropout 0,5** entre LSTMs y antes de la FC.
- **Dataset:** **60.910 muestras de entrenamiento / 17.382 de test, 17 clases de emisores** (aéreos/terrestres), secuencias de **longitud variable** y **clases desbalanceadas** → se evalúa con **accuracy macro-promediada** (penaliza ignorar clases minoritarias).
- **Resultados (relativos):** las LSTM específicas por atributo superan al RNN conjunto en **2–14 %**; frente a baselines de DL (MLP de Petrov et al., CNN/ResNet18 para series temporales) la mejora es del **2–19 %**, batiendo a **ResNet18 por 14–15 %**. En **robustez a ruido gaussiano** (hasta 10 % de magnitud ≈ SNR 20 dB, umbral típico de sistemas EW) mantiene ventaja creciente: de +2 % sin ruido a +6 % a SNR 20 dB sobre Petrov et al.
- **Lectura para el TFM:** confirma que **separar el procesamiento por atributo** y normalizar por secuencia es una palanca de precisión barata, y que la métrica correcta con clases desbalanceadas es la **macro-accuracy** (no la global).

### 2.4 Dataset destacado (nuevo)
**The Turing Synthetic Radar Dataset** (Gunn et al., 2026): dataset simulado a gran escala para *pulse deinterleaving* con **métricas estandarizadas (V-measure)** para identificación/clustering de emisores. Es una oportunidad de benchmark reproducible muy poco explotado. (HF: https://hf.co/papers/2602.03856)

### 2.5 Hueco / oportunidad para el TFM (modelo 2)
- Reportar **latencia <1 ms** (media + p99) en hardware fijo — los trabajos publican accuracy pero no tiempos (y el caso 2.2 sugiere ~15 ms de partida).
- **Robustez bajo SNR variable y PDW corrompidos/perdidos** evaluada sistemáticamente.
- Benchmark sobre el dataset Turing para reproducibilidad.

---

## 3. Multi-Agent RL para coordinación en formación

> **Modelo con análisis en profundidad.** Esta sección es el *deep dive* del modelo 3: el caso de estudio A (MA-CJD) se ha leído íntegro (PDF en `../articles/07...pdf`) por ser la plantilla más cercana al objetivo del TFM.

### 3.1 Planteamiento del campo

El problema se modela como **juego de Markov / Dec-POMDP**: varios agentes EW (aeronaves o jammers) con observación local deben coordinar emisiones para un objetivo común (suprimir un IADS, mantener comunicaciones). El esquema dominante es **CTDE** (*Centralized Training, Decentralized Execution*): se entrena con información global pero cada agente ejecuta solo con su observación local.

El algoritmo de referencia es **QMIX**, que aprende una función de valor de acción **centralizada pero factorizable** (mezcla monótona de los valores individuales mediante una *hyper-network* con pesos no negativos que garantiza la condición *Individual-Global-Max*), resolviendo la **asignación de crédito** entre agentes. Dos problemas que QMIX mitiga y conviene conocer: la **no estacionariedad** (cada agente es parte del entorno de los demás) y la **"pereza" de agente** (la recompensa global puede desincentivar a un agente concreto). Líneas activas 2024-25: **MARL jerárquico** contra agilidad/diversidad de frecuencia, **asignación cooperativa de recursos de jamming** y control de enjambres UAV bajo EW.

Una limitación técnica recurrente: cuando la decisión combina **modo discreto + parámetros continuos** (p. ej. técnica de jamming + nivel de potencia), QMIX por sí solo no basta — discretizar la potencia pierde precisión. La solución de referencia es acoplar QMIX con una arquitectura de **acción parametrizada** (MP-DQN).

### 3.2 Caso de estudio A (deep dive) — MA-CJD: jamming cooperativo contra radares en red
*(lectura completa del PDF: Cai et al., Autonomous Intelligent Systems 5:3, 2025, SpringerOpen — artículo 07)*

Es, en la práctica, **un escenario de supresión de IADS**: varios jammers protegen unidades de defensa frente a una red de radares cooperativos. Mapea casi 1:1 con el modelo 3 del TFM (un agente por plataforma coordinando deception + gestión de potencia).

- **Escenario:** **4 jammers vs 4 radares**. Jammers a 50 km del centro en azimuts 45/−45/135/−135°; radares a distancia ~N(400 km, 30 km) y azimut uniforme, avanzando hacia el centro; cuando un radar fija (lock) una unidad o un señuelo, pasa a modo seguimiento. Movimiento en plano 2D. Cuatro tipos de radar con distinta amenaza (Tabla 1: potencia 300/300/180/100 MW; factores anti-jamming de lóbulo principal/lateral distintos).
- **Formulación (juego de Markov `⟨N,S,Aᵢ,p,r,γ⟩`):** los agentes son los jammers; el modelo de radar es el entorno.
  - **Acción parametrizada `uᵢ = (Tᵢ, Pᵢ)`:** `Tᵢ` discreto codifica **objetivo + tipo** (`Tᵢ=0` sin jamming; par → deception, impar → suppression; `⌊(Tᵢ+1)/2⌋` = ID del objetivo), con `K = 2·|R|` (aquí 9 acciones discretas, `2×4+1`). `Pᵢ ∈ [0,1]` es el **nivel de potencia continuo** (mapeado a potencia real entre `P_min` y `P_max`).
  - **Recompensa `r = r_d + r_p + r_j`:** `r_d` penaliza que un radar fije una unidad (entre −1,2 y −0,8 según amenaza del radar: −0,9/−1,1/−1,0/−1,0); `r_p` penaliza el consumo de potencia (lineal, entre −0,1 y −0,01); `r_j` es la **probabilidad de éxito de jamming** (para deception, `1 − ∏(1−p_det,i)` sobre falsos blancos que superan el SNR del blanco real; para suppression, la reducción de probabilidad de detección del eco). Diseño que prioriza evadir detección, equilibrando eficiencia de potencia.
  - **Estado (vector de 48 dim):** por cada radar, atributos inherentes (potencia pico `Pₜ`, ancho de haz `θₘ`, periodo de barrido `Tₛ`, tipo de radar one-hot) + dinámicos (dirección de haz `θₐ`, posición `posᵣ`); más las posiciones de los jammers.
- **Algoritmo (MA-CJD = QMIX + MP-DQN + Double DQN):** cada agente tiene una **Actor network** (3 capas FC, dim 128, ReLU, salida Sigmoid → nivel de potencia por cada acción discreta) y una **Q network** *multi-pass* con **capa oculta GRU de 128** (incorpora historial estado-acción), salida de 9 valores. La **Mixing network** es una hyper-network de 2 capas (ELU, 64 dim) con pesos en valor absoluto (IGM). **Double DQN** desacopla selección (red original) y evaluación (red objetivo) para mitigar sobreestimación del target TD.
- **Entrenamiento e implementación:** ε-greedy de 0,95 → 0,05 decayendo en 100.000 pasos; LR `α=0,005` (Q + Mixing) y `β=0,003` (Actor); Adam; por ciclo, 16 episodios de interacción + 4 iteraciones muestreando 32 episodios; 100.000 pasos totales. Entorno en **C++** (servidor) + cliente **Python 3.12**; redes en **PyTorch 2.4 / CUDA 12.4**; hardware Intel i9-13900 + RTX 4090. *(Nota: mismo stack PyTorch 2.4/CUDA 12.4 que el TFM.)*
- **Resultados (Tabla 2, medias sobre 32 experimentos):**

  | Estrategia | Duración total de lock (s) ↓ | Suma de potencia de jamming ↓ |
  |---|---|---|
  | Random | 84,49 | 69,61 |
  | Rule-based | 70,18 | 67,38 |
  | PER-DDQN | 66,32 | 64,89 |
  | QMix | 65,70 | 60,98 |
  | **MA-CJD** | **37,91** | **35,94** |

  MA-CJD logra la **menor duración de lock (37,91 s, −45,98 % vs. rule-based)** y el **menor consumo (35,94, −46,66 % vs. rule-based)**. Frente a QMix puro, añadir MP-DQN **reduce el consumo de recursos un 41,06 % y el tiempo de detección un 42,30 %**. Durante el entrenamiento el tiempo de lock baja **>60 %**. Es el único método que aprende a **modular la potencia según el estado** (poca potencia lejos, más al acercarse el radar).
- **Limitaciones declaradas:** modelo aún simplificado (2D, sin altitud); se plantea incorporar modelos y parámetros más detallados para aplicabilidad práctica. No reporta **latencia de inferencia**.

### 3.3 Caso de estudio B — QMIX para resiliencia anti-jamming en enjambres
*(lectura del texto completo HTML: arXiv 2512.16813)*

Complementa al anterior desde el lado de **comunicaciones** (no jamming ofensivo), con detalle de recompensa y entorno:

- **Problema:** proteger redes de enjambre frente a un *reactive jammer* markoviano con umbral.
- **Formulación:** acción conjunta por agente `(canal cᵢ, potencia Pᵢ)` con `M ∈ {4,8,10}` canales y `N ∈ {5,10}` agentes. **Observación local:** potencia recibida por canal + acción y recompensa previas. **Estado central (solo entrenamiento):** potencia agregada por canal + estado del jammer. **Recompensa:** throughput individual `log₂(1+SINR)` menos penalización de interferencia co-canal `λ/d_ij^β`; recompensa de equipo = suma. Canal Rayleigh con *block fading* (`T_c=100`); jammer con dos umbrales (`θ_H=0,4`, `θ_L=0,2`).
- **Baselines:** óptimo "genie-aided", UCB local, heurística reactiva sin estado.
- **Resultados:** QMIX **converge a políticas cooperativas que casi igualan la cota genie-aided** (con `M=10`); con `N=10, M=4` el margen sobre los baselines **supera el 50 %** en convergencia, con ejecución totalmente descentralizada. *(El artículo da curvas, no tablas numéricas exactas de throughput.)*
- **Limitaciones declaradas:** un solo jammer, agentes homogéneos, sin modelar latencia ni restricciones de energía/sensado ruidoso (futuro trabajo).

> Nota de confianza: una fuente tipo *survey* (arXiv 2508.11687) devolvió tablas de métricas con apariencia inferida por el extractor; **no se citan esas cifras**.

### 3.4 Hueco / oportunidad para el TFM (modelo 3)

El MA-CJD es la base más sólida y reutilizable. Diferenciadores concretos del TFM frente a él:
1. **Métricas EW operacionales** (J/S ratio, burnthrough range, POI) en lugar de solo "duración de lock" y "suma de potencia" — más interpretables para revisores de dominio.
2. **Escenario NTTR (PBECR/TPECR)** con parámetros SIADS publicados (SA-2 a S-400), frente a 4 tipos de radar sintéticos.
3. **Deception coordinada explícita** entre aeronaves (no solo asignación objetivo+potencia).
4. **Latencia de inferencia documentada** (media + p99) — ausente en MA-CJD y en el resto.
5. **Reutilizar la receta probada**: QMIX + MP-DQN (acción parametrizada modo+potencia) + Double DQN + Q-net con GRU es un punto de partida validado; el TFM puede adoptarla y enfocar la contribución en escenario/realismo/latencia.

---

## 4. GAN para señales sintéticas

### 4.1 Planteamiento del campo

Dos usos: (i) **data augmentation** para clasificación de modulación cuando las muestras reales escasean (especialmente señales **LPI**, escasas por diseño), y (ii) **diseño generativo de formas de onda** con propiedades deseadas (baja probabilidad de detección/interceptación). Los enfoques ganadores son **cGAN** y **Wasserstein DCGAN con atención** para *small-sample*. Tendencia emergente 2025-26: **híbridos físico + generativo** (p. ej. SA-Radar) y **modelos de difusión** desplazando a la GAN pura en algunas tareas radar.

### 4.2 Caso de estudio (deep dive) — cWGAN-GP para formas de onda radar LPD
*(lectura completa del PDF: arXiv 2403.12254, publicado en IEEE Trans. Radar Systems 2025, DOI 10.1109/TRS.2025.3542283 — artículo 04)*

Es especialmente relevante porque **usa el mismo dataset RadioML 2018.01A** de la capa de datos del proyecto, como distribución de fondo RF.

- **Problema:** diseñar formas de onda que sean simultáneamente difíciles de detectar (LPD) y buenas sensando (función de ambigüedad tipo "thumbtack"). Optimización multiobjetivo: minimizar la **divergencia KL** entre la distribución de la forma de onda generada y el fondo RF ambiente, manteniendo el lóbulo principal estrecho y los lóbulos laterales suprimidos.
- **Arquitectura:** **Conditional Wasserstein GAN con Gradient Penalty (cWGAN-GP)**. Generador 1D-ResNeXt (12,2 M parámetros) que toma ruido `z` + vector condicional `y` (muestra instantánea del fondo RF) y produce una **forma de onda IQ compleja de 1024 muestras** (salida `2·tanh(x/2)`). Crítico simétrico (11,7 M parámetros) con *layer normalization* (compatible con gradient penalty). Pérdida Wasserstein-1 + penalización de gradiente (λ) + pérdida de ambigüedad (lóbulo principal y laterales) en *fine-tuning*.
- **Entrenamiento:** fase 1 solo WGAN-GP hasta converger; fase 2, 3.000 iteraciones con `L_W + η·L_ambig`. Adam (gen, LR 1×10⁻⁵), RMSProp (crítico, LR 5×10⁻⁵), batch 512 en 2× V100, 5 iteraciones de crítico por 1 de generador. Fondo RF: **RadioML 2018.01A** (24 modulaciones, –20..+30 dB), con 50 % para entrenar el generador y 50 % el detector (sin solape).
- **Resultados (detección a 1 % de FAR con detector ResNet-18):**

  | Forma de onda | Tasa de detección @1 % FAR |
  |---|---|
  | Formas base (radar SIDLE) | 95–99 % |
  | **Generadas cWGAN-GP** (mismo nº muestras) | **19,5 % ± 5,5 %** |
  | Generadas, detector con 10× datos | ~70 % |
  | Generadas, detector con 100× datos | 80,8 % |

  Es decir, **~90 % menos detectabilidad** con igual entrenamiento; la dificultad persiste aun dando 100× más datos al detector. El detector ciclostacionario (FAM) directamente falla a FAR bajo. El sensado (ancho de lóbulo principal, PSL, ambigüedad Doppler) es **ajustable vía η** con compromiso explícito LPD↔sensado.
- **Limitaciones declaradas:** ancho de pulso fijo; dependencia del fondo (peor sensado cuando se condiciona en AM/FM); degradación ante interferencia a 0 dB; coste de entrenamiento (≈24 M parámetros, 2× V100); latencia de inferencia no reportada.

### 4.3 Hueco / oportunidad para el TFM (modelo 4)
Posicionar la GAN **no como fin sino como augmentation con impacto medible**: ablation con/sin datos sintéticos sobre los modelos 1-2, reportando la mejora en accuracy/robustez. Mencionar **difusión** como alternativa (dirección 2025-26). El precedente cWGAN-GP sobre RadioML da una plantilla de arquitectura y de protocolo de evaluación directamente reutilizable.

---

## 5. Librería EW convencional (baseline)

### 5.1 Rol

No es un frente de ML, sino el **baseline rule-based** contra el que se mide la mejora de los modelos 1-4. La literatura de GE cognitiva confirma y justifica su uso como suelo de rendimiento: los sistemas ECM tradicionales (noise preset, repeater, deception, chaff, decoys, maniobras evasivas) dependen de **librería de amenazas pre-cargada** y **fallan ante amenazas "zero-day"** y radares digitales programables con waveforms flexibles (EMSOPEDIA, *Cognitive EW*; patente US 11808882 sobre ECM sin base de datos de amenazas).

### 5.2 Diseño recomendado para el TFM
- Implementar un selector determinista amenaza→contramedida que reproduzca la doctrina clásica.
- Usarlo como **fila de referencia** en todas las tablas comparativas: el objetivo >92 % de victorias se mide *contra* este baseline, y la mejora de los modelos RL/CNN debe ser estadísticamente significativa.
- Evita afirmaciones cualitativas: cuantifica el delta (win rate, J/S, burnthrough) modelo cognitivo − baseline.

---

## 6. Síntesis transversal y posicionamiento Q1

1. **El diferenciador no es un modelo, es la integración.** La literatura trata casi siempre un modelo aislado (un DRL de jamming, una CNN de clasificación, un QMIX de coordinación). El valor del TFM está en el **sistema completo** (5 modelos) sobre escenario NTTR (PBECR/TPECR) con métricas EW de dominio (J/S, burnthrough, POI, false alarm rate).
2. **La latencia es una contribución medible y casi inédita.** Ninguno de los casos de estudio leídos reporta latencia de inferencia; el caso 2.2 sugiere ~15 ms de partida para una CNN en GTX 1060. Medir **media + p99 en hardware fijo** (L4/T4 + CPU, no en la GPU más rápida) y alcanzar <5 ms / <1 ms es un resultado publicable por sí mismo.
3. **Reproducibilidad como ventaja.** El campo EW es opaco por su naturaleza clasificada; un pipeline reproducible (RadioML 2018.01A + dataset Turing + configs versionadas + seeds) es escaso y valioso para revisores.
4. **Coherencia de datos entre modelos.** RadioML 2018.01A es a la vez base de clasificación (modelo 2), fondo RF para la GAN (modelo 4, como en el caso 4.2) y fuente de modulaciones para el entorno RL — refuerza la narrativa de "capa de datos compartida".
5. **Encaje de revistas:** ESWA y Knowledge-Based Systems ya publican *deinterleaving* y *jamming DRL* (encaje directo); IEEE TNNLS si el ángulo es metodológico (POMDP/MARL); Information Sciences para el componente generativo/datos.

---

## 7. Datasets y benchmarks

| Dataset | Modelo(s) | Uso | Enlace |
|---------|-----------|-----|--------|
| RadioML 2018.01A | 2, 4, (1) | Clasificación de modulación IQ a distintos SNR; fondo RF para GAN | Capa de datos del proyecto |
| The Turing Synthetic Radar Dataset (2026) | 2 | Deinterleaving / identificación de emisores; métrica V-measure | https://hf.co/papers/2602.03856 |
| SIDLE (radar) | 4 | Formas de onda radar base para comparación de detectabilidad | Citado en arXiv 2403.12254 |
| SA-Radar (sim. controlable) | 4 | Generación de datos radar realistas por parámetros | https://hf.co/papers/2506.03134 |

---

## 8. Referencias

**Acceso abierto leído en profundidad:**
1. *Deep Reinforcement Learning Based Decision Making for Complex Jamming Waveforms* — Sensors (PMC9601320). https://pmc.ncbi.nlm.nih.gov/articles/PMC9601320/
2. *Airborne Radar Anti-Jamming Waveform Design Based on Deep Reinforcement Learning* — Sensors (PMC9692253). https://pmc.ncbi.nlm.nih.gov/articles/PMC9692253/
3. *Specific Radar Recognition Based on Characteristics of Emitted Radio Waveforms Using CNNs* — Sensors (PMC8707803). https://pmc.ncbi.nlm.nih.gov/articles/PMC8707803/
4. *Adaptive LPD Radar Waveform Design with Generative Deep Learning* — arXiv 2403.12254 / IEEE Trans. Radar Systems 2025, DOI 10.1109/TRS.2025.3542283. https://arxiv.org/html/2403.12254
5. *Coordinated Anti-Jamming Resilience in Swarm Networks via Multi-Agent RL* — arXiv 2512.16813. https://arxiv.org/abs/2512.16813
7. *A cooperative jamming decision-making method based on MARL (MA-CJD: QMIX + MP-DQN + Double DQN)* — Cai et al., Autonomous Intelligent Systems 5:3, 2025 (SpringerOpen, acceso abierto). https://doi.org/10.1007/s43684-025-00090-4

**Abstract / snippet verificado (re-verificar cuerpo antes de citar cifras):**
6. *Radar Emitter Classification with Attribute-specific RNNs* — arXiv 1911.07683. https://arxiv.org/abs/1911.07683
8. *Improving anti-jamming decision-making for cognitive radar via MARL* — Signal Processing, 2023. https://www.sciencedirect.com/science/article/abs/pii/S1051200423000477
9. *Frequency Diversity Array Radar and Jammer Frequency-Domain Power Countermeasures via MARL* — Remote Sensing 16(11):2127, 2024. https://doi.org/10.3390/rs16122127
10. *Hybrid Game-Theoretic and Reinforcement Learning for Adaptive Radar Jamming Decision-Making* — 2025. https://www.researchgate.net/publication/397230778
11. *Radar Jamming Decision-Making in Cognitive EW: A Review* — 2023. https://www.researchgate.net/publication/370167521
12. *A framework for radar signal deinterleaving and parameter estimation (split-pulse, deep learning)* — Expert Systems with Applications, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0957417425010280
13. *Data augmentation with conditional GAN for automatic modulation classification* — ACM WiseML. https://dl.acm.org/doi/10.1145/3395352.3402622
14. *WDCGAN-GSMR: small-sample radar signal modulation recognition* — Signal Processing, 2026. https://www.sciencedirect.com/science/article/abs/pii/S1051200426000916

**Dominio / framing:**
15. *Cognitive EW* — EMSOPEDIA. https://www.emsopedia.org/entries/cognitive-ew/
16. *Radar electronic countermeasures without a threat database* — patente US 11808882. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11808882
17. *Jammer Versus Radar in a Cognitive Electronic Warfare Setting* — preprint TechRxiv, 2025/26. https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176537934.47026414

> **Aviso de fiabilidad.** Algunas fechas de publicación (2026) y enlaces de ResearchGate/paywall provienen de extracción automática y deben confirmarse en la fuente original antes de la bibliografía final del TFM. Las cifras de **todos** los casos de estudio en profundidad (1.2, 1.3, 2.2, 2.3, 3.2, 3.3, 4.2) provienen de **lectura completa** del artículo (PDFs en `../articles/`).
