# Fase 1 - EDA

## Resumen del Dataset

- **Número total de archivos candidatos:** 26,179
- **Número total de imágenes válidas:** 26,179
- **Número de clases:** 10
- **Nombres de clases:** cane, cavallo, elefante, farfalla, gallina, gatto, mucca, pecora, ragno, scoiattolo
- **Formatos detectados:** JPEG, PNG

## Distribución

| clase | cantidad | porcentaje |
|---|---:|---:|
| cane | 4863 | 18.58% |
| cavallo | 2623 | 10.02% |
| elefante | 1446 | 5.52% |
| farfalla | 2112 | 8.07% |
| gallina | 3098 | 11.83% |
| gatto | 1668 | 6.37% |
| mucca | 1866 | 7.13% |
| pecora | 1820 | 6.95% |
| ragno | 4821 | 18.42% |
| scoiattolo | 1862 | 7.11% |

La razón entre la clase mayoritaria y la minoritaria es **3.36:1**, lo que representa un desbalance **alto**.

## Hallazgos

- El dataset contiene 26,179 imágenes válidas distribuidas en 10 clases.
- Existe desbalance alto: razón mayoritaria/minoritaria de 3.36:1.
- La resolución mediana es 300×225 px y las dimensiones son heterogéneas.
- Se detectaron 2 formatos de imagen: JPEG, PNG.
- La auditoría encontró 0 archivos corruptos/vacíos, 5 imágenes muy pequeñas y 1970 muy grandes según IQR.
- Las distribuciones RGB cambian entre clases; el color puede aportar señal, pero también codificar fondos/contextos.

## Riesgos detectados

- Desbalance de clases (3.36:1) entre cane y elefante.
- 0 archivos corruptos, vacíos o no legibles.
- 5 imágenes con lado menor a 64 px y 1970 outliers grandes por área.
- Heterogeneidad de resolución, relación de aspecto y modos de color (2 no RGB/RGBA).

## Recomendaciones para Fase 2

- **Resolución objetivo sugerida:** 224×224 px, preservando relación de aspecto mediante resize + padding o crop controlado. Confirmar según el modelo que se seleccione posteriormente.
- **División sugerida:** train/validation/test estratificada (por ejemplo 70/15/15), con semilla fija. Antes de dividir, investigar duplicados para evitar fuga de información. Esta división aún no se ha ejecutado.
- **Data augmentation:** sí, recomendada, aplicada solo al conjunto de entrenamiento; priorizar transformaciones geométricas y fotométricas moderadas.
- **Balanceo:** sí, considerar pesos de clase, muestreo o augmentation focalizada, y evaluar métricas macro además de accuracy.
- Convertir de forma consistente a RGB y normalizar valores de píxel en la fase de preprocesamiento.
- Revisar manualmente los outliers reportados antes de excluir cualquier archivo.

> Archivo generado automáticamente por `animal10_project.ipynb` a partir del contenido actual de `data/`.

# Fase 2 - Preprocessing

## Resolución seleccionada

**224×224 px, RGB (3 canales)**. La verificación completa confirmó que las 26,179 imágenes producen arreglos con forma `(224, 224, 3)`.

## Duplicados encontrados

- Pares duplicados exactos (MD5): **0**.
- Pares potenciales (pHash y aHash): **1,870**.
- Pares relacionados entre etiquetas diferentes: **1**.
- Lista para revisión: `duplicate_review.csv`.
- Decisión: no eliminar automáticamente; mantener cada componente conectado dentro de un único split.

## Estrategia de resize

Se seleccionó **resize preservando relación de aspecto + padding** mediante `resize_with_padding()`. El resize directo se descartó como estrategia principal porque deforma objetos no cuadrados.

## Estrategia de augmentation

En entrenamiento se recomiendan flip horizontal, rotación aleatoria en ±15°, zoom moderado y ajustes moderados de brillo y contraste. No aplicar augmentation a validation o test. En esta fase solo se visualizaron candidatos.

## Split generado

- Train: **n = 18,310**
- Validation: **n = 3,933**
- Test: **n = 3,936**

La división es estratificada, reproducible (`random_state=42`) y respeta grupos de duplicados.

## Class weights

| class_id | class_name | class_weight |
|---:|---|---:|
| 0 | cane | 0.5370 |
| 1 | cavallo | 0.9973 |
| 2 | elefante | 1.8237 |
| 3 | farfalla | 1.2397 |
| 4 | gallina | 0.8457 |
| 5 | gatto | 1.5703 |
| 6 | mucca | 1.4106 |
| 7 | pecora | 1.4451 |
| 8 | ragno | 0.5420 |
| 9 | scoiattolo | 1.3988 |

Los pesos se calcularon con `sklearn.utils.class_weight` sobre train y no se han aplicado todavía.

## Leakage Check

**PASS** — rutas compartidas, MD5 compartidos, grupos duplicados y pares duplicados entre splits: 0.

## Decisiones tomadas

- Conservar imágenes pequeñas y grandes, marcadas para revisión; no hubo eliminación automática.
- Convertir a RGB y redimensionar bajo demanda, sin crear copias procesadas permanentes.
- Mantener duplicados exactos y potenciales en el mismo split.
- Guardar manifiestos con rutas relativas y etiquetas numéricas reproducibles.
- No aplicar balanceo ni augmentation todavía; solo calcular pesos y definir candidatos.

## Recomendaciones para Fase 3

- **Representación principal:** embeddings de una CNN preentrenada / Transfer Learning; conserva estructura espacial y es la opción recomendada.
- **PCA:** no aplicar a los píxeles si se usa CNN. Para un baseline clásico con imágenes flatten, usar PCA antes del clasificador para reducir dimensionalidad.
- **Modelos candidatos:** baseline clásico PCA + regresión logística o SVM lineal; posteriormente CNN pequeña y Transfer Learning.
- **Baseline recomendado:** embeddings congelados de una red preentrenada + clasificador lineal, comparado con PCA + regresión logística.

# Fase 3

## Resultados

Benchmark estratificado reproducible: 500 train, 200 validation y 200 test; 224×224 RGB para los tres pipelines. Test no se utilizó para seleccionar hiperparámetros.

| pipeline | best_C | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---:|---:|---:|---:|---:|---:|
| C - MobileNetV2 embeddings | 10 | 0.9300 | 0.9120 | 0.9343 | 0.9120 | 0.9200 |
| A - PCA + LR | 0.1 | 0.2600 | 0.2380 | 0.2396 | 0.2380 | 0.2337 |
| B - PCA + SVM | 10 | 0.2100 | 0.2100 | 0.2082 | 0.2100 | 0.2015 |

PCA de píxeles conservó 95.0219% de varianza con 247 componentes.

## Mejor representación

**C - MobileNetV2 embeddings**, con Macro F1 **0.9200** y accuracy **0.9300** en el test del benchmark. La comparación favorece representaciones visuales que preservan estructura espacial y semántica frente a píxeles aplanados.

## Errores frecuentes

- **cavallo → mucca:** 2 casos
- **gallina → cane:** 2 casos
- **scoiattolo → ragno:** 1 casos
- **mucca → pecora:** 1 casos
- **gallina → ragno:** 1 casos

Existe además un par perceptualmente idéntico con etiquetas `cavallo` y `mucca`, detectado en Fase 2 y mantenido dentro de un único split; debe revisarse manualmente como posible conflicto de etiqueta.

## Decisión para Fase 4

Continuar con **C) MobileNetV2 Embeddings** si corresponde al ganador observado; usar el pipeline clásico mejor puntuado únicamente como baseline. En Fase 4 se recomienda ampliar la evaluación al conjunto completo mediante extracción cacheada y, solo después, estudiar fine tuning parcial con early stopping. No se realizó fine tuning en esta fase.

# Fase 4 - MobileNetV2 Fine Tuning

## Resultados de validation

| Experimento | Mejor época | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A - Feature Extraction | 2 | 0.9600 | 0.9571 | 0.9616 | 0.9586 | 0.9616 |
| B - Fine Tuning parcial | 4 | 0.9300 | 0.9321 | 0.9358 | 0.9312 | 0.9358 |
| C - Fine Tuning completo | 6 | 0.8450 | 0.8452 | 0.8445 | 0.8273 | 0.8445 |

## Evaluación final en test

- Experimento seleccionado antes de abrir test: **A - Feature Extraction**.
- Mejor checkpoint: **época 2**, archivo `phase4_checkpoints/A_feature_extraction_best.pt`.
- Accuracy: **0.9050**.
- Macro Precision: **0.8995**.
- Macro Recall: **0.8984**.
- Macro F1: **0.8968**.
- Balanced Accuracy: **0.8984**.

## ¿Vale la pena Fine Tuning?

**No con la evidencia actual**. El cambio de Macro F1 frente al baseline 0.9200 fue **-0.0232**. Se exige una mejora superior a 0.01 para justificar el costo y complejidad adicionales en este benchmark.

## ¿Cuál es el mejor checkpoint?

Dentro de A/B/C, el mejor checkpoint por validation es **A - Feature Extraction, época 2**.

## ¿Cuál será el modelo definitivo?

**MobileNetV2 congelado + embeddings + Logistic Regression (Fase 3)**.

## Errores frecuentes del ganador en test

- cavallo → mucca: 4
- mucca → pecora: 2
- gallina → cane: 2
- cane → cavallo: 2
- gallina → ragno: 1

La evaluación test se ejecutó una sola vez después de bloquear experimento y checkpoint. No se usó test para seleccionar hiperparámetros.

# Fase 5 - Interpretabilidad

## Interpretabilidad

Se interpretó el modelo definitivo MobileNetV2 congelado + embeddings L2 + Logistic Regression (`C=10`). La auditoría utilizó las 3,936 imágenes completas de `test.csv`, sin modificar el modelo. Se generaron matriz de confusión, métricas por clase, 20 errores, PCA, t-SNE y Grad-CAM conectado a los coeficientes reales de Logistic Regression.

Métricas de la auditoría completa: Accuracy **0.9339**, Balanced Accuracy **0.9200**, Macro Precision **0.9396**, Macro Recall **0.9200** y Macro F1 **0.9290**.

## Principales errores

- La clase más difícil por F1 es mucca (0.8467).
- Los errores tienen confianza media 0.476; 29 superan 0.70 y merecen revisión prioritaria.
- La confusión dirigida principal es mucca → cavallo (23 casos).
- Los centroides más próximos son cavallo–mucca, coherente con posibles solapamientos visuales.

## Principales fortalezas

- Las clases más fáciles son ragno, gallina, cane, con recalls entre 96.8% y 98.1%.
- La mejor clase por F1 es ragno (0.9765).
- Los embeddings producen agrupaciones semánticas visibles; el backbone congelado conserva información discriminativa sin fine tuning.

## Confusiones relevantes

- mucca → cavallo: 23 casos
- pecora → mucca: 19 casos
- cavallo → mucca: 19 casos
- scoiattolo → cane: 17 casos
- gatto → cane: 16 casos

## Conclusiones

- El modelo se apoya en representaciones semánticas transferidas y no solo en color o fondo; Grad-CAM permite comprobarlo caso por caso.
- Las clases con centroides cercanos y errores recíprocos requieren más ejemplos diversos o revisión de etiquetas.
- Los errores de alta confianza deben auditarse primero, especialmente el par `cavallo`/`mucca` ya señalado en fases anteriores.
- La interpretabilidad respalda mantener MobileNetV2 congelado + Logistic Regression como modelo definitivo; no aporta evidencia para reabrir el fine tuning.

# Mapa narrativo de decisiones

Se agregó al notebook un diagrama integral de storytelling que conecta las cinco fases, la evidencia, las alternativas aceptadas y rechazadas y el modelo definitivo. El mapa está disponible en `animal10_decision_story.png` y `animal10_decision_story.svg`.

**Lección central:** MobileNetV2 congelado + embeddings + Logistic Regression fue seleccionado porque la representación semántica aportó más que el fine tuning o los píxeles aplanados, manteniendo cero leakage y una interpretación verificable.
