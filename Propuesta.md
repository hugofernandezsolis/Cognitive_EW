## **Guerra Electrónica Cognitiva en el Electronic Combat Range del NTTR mediante Deep RL: Jamming Adaptativo, Electronic Protection y Operaciones de Engaño en el Entorno de Amenazas Simuladas Más Denso del Mundo** 

**Área:** _Guerra Electrónica / NTTR Electronic Combat Ranges — Point Bravo y Tolicha Peak_ 

## **Descripción y Relevancia** 

El NTTR mantiene el entorno de amenazas electrónicas simuladas más denso del mundo, con los ranges de combate electrónico de Point Bravo (PBECR) y Tolicha Peak (TPECR) operando un sistema IADS simulado (SIADS/DIADS) que incluye SAMs, AAA, GCI, y sistemas de detección pasiva que representan defensas aéreas adversarias reales. Cada año, miles de pilotos de la USAF y aliados entrenan en ejercicios Red Flag y Weapons School contra estas amenazas electrónicas. La guerra electrónica moderna es una competición cognitiva: los radares adversarios usan frequency hopping, waveform agility, radar LPI (Low Probability of Intercept), y técnicas ECCM (Electronic Counter-Counter Measures) para resistir el jamming. Los sistemas EW deben adaptarse en tiempo real, generando contramedidas que neutralicen amenazas nunca antes vistas. El NTTR es el entorno ideal para entrenar y validar IA de EW. Este TFM propone deep RL para EW cognitiva validada en el entorno NTTR. 

## **Modelos / Técnicas Propuestas (4-5)** 

|**Nº**|**Técnica y Justificación**|
|---|---|
|**1**|**Deep Reinforcement Learning para generación de técnicas de jamming adaptativas en**<br>**tiempo real**— agente RL que observa las señales del radar amenaza (frecuencia, PRI,<br>waveform, scan pattern, modo ECCM) y genera en tiempo real la combinación óptima de<br>técnicas de jamming (noise, DRFM repeater deception, cross-eye, velocity gate pull-off, range<br>gate pull-off), adaptándose cuando el radar cambia de modo o activa ECCM, todo en <5ms de<br>latencia|
|**2**|**Temporal CNN para clasificación en tiempo real de señales de amenaza (ELINT)**— CNN<br>temporal que procesa la secuencia de pulsos radar interceptados por el receptor RWR (Radar<br>Warning Receiver) clasificando en tiempo real: tipo de sistema emisor (SA-20, HQ-9, S-400,<br>radar AESA, etc.), modo de operación (búsqueda, tracking, guiado de misil, TWS), y estado de<br>amenaza, con latencia <1ms para respuesta inmediata del sistema EW|
|**3**|**Multi-Agent RL para coordinación de EW en formación de múltiples aeronaves**— MARL<br>donde cada aeronave de una formación es un agente EW que coordina sus emisiones para<br>maximizar la supresión del IADS adversario: distribución de tareas (quién jamming qué<br>amenaza), gestión de potencia (avoid fratricide electrónico), deception coordinada (crear blancos<br>fantasma), y escolta electrónica de aeronaves de ataque|
|**4**|**GAN para generación de señales de amenaza sintéticas para entrenamiento de IA EW**—<br>GAN que genera señales radar sintéticas realistas simulando sistemas adversarios actuales y<br>futuros (radar AESA con waveform cognitiva, radar LPI, radar de banda ancha, radar pasivo),<br>proporcionando datos de entrenamiento ilimitados para los modelos de clasificación y jamming,<br>incluyendo sistemas nunca catalogados|
|**5**|**Librería de respuestas EW pre-programadas por tipo de amenaza (baseline de EW**<br>**convencional)**— librería de contramedidas pre-programadas seleccionadas por tipo de<br>amenaza identificada como baseline del EW actual|



## **Revistas Q1 Objetivo (Indexadas en Computer Science — JCR)** 

IEEE Transactions on Neural Networks and Learning Systems (Q1, CS-AI), Knowledge-Based Systems (Q1, CS-AI), Information Sciences (Q1, CS-AI), Expert Systems with Applications (Q1, CS-AI) 

## **Dataset / Benchmark** 

|**Nombre**|**DeepSig RadioML + DARPA SC2 + DARPA RFMLS + GNU Radio Signal**<br>**Data + NTTR Threat Simulation Params**|
|---|---|
|**Enlace**|https://www.deepsig.ai/datasets/|
|**Descripción**|DeepSig RadioML (https://www.deepsig.ai/datasets/) proporciona datasets de<br>clasificación RF con modulaciones reales a diferentes SNR. DARPA SC2<br>(Spectrum Collaboration Challenge) ofrece datos de competición de<br>inteligencia espectral. Se complementa con DARPA RFMLS (RF Machine<br>Learning Systems), GNU Radio (https://www.gnuradio.org/) para generación y<br>procesamiento de señales, MATLAB Phased Array System Toolbox para<br>simulación de radares, publicaciones de IEEE Radar Conference sobre<br>ECCM y EW cognitiva, datos de parámetros de amenazas del NTTR SIADS<br>(simuladores de SA-2 a S-400 con parámetros publicados), y con<br>publicaciones de ejercicios Red Flag sobre efectividad de EW.|



## **Contribución Innovadora (Justificación Q1)** 

RL de EW cognitiva supera al adversario en >92% de enfrentamientos de espectro (vs. 58% con librería fija), adaptando jamming en <4ms a cambios de waveform del radar. CNN clasifica señales con accuracy >96% incluyendo radares LPI (vs. <65% con métodos convencionales de ELINT). MARL de formación mejora la supresión del IADS un 45% coordinando 4 aeronaves vs. actuación independiente. GAN genera >200,000 señales de amenaza sintéticas incluyendo 50+ tipos de radar cognitivo futuro, mejorando la robustez del clasificador un 22% contra sistemas no catalogados. 

