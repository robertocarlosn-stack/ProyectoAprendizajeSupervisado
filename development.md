# Guía de desarrollo del proyecto Animal-10

Este documento explica los comandos y parámetros de programación más importantes de `animal10_project.ipynb`, por qué se utilizaron y cómo afectan los resultados. No sustituye al notebook: funciona como referencia técnica para entender, reproducir y defender el experimento.

## 1. Principios generales

### Semillas de reproducibilidad

```python
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
```

- `SEED=42` hace reproducibles el muestreo, las divisiones y la inicialización de modelos.
- Cambiarla puede producir otras imágenes en el benchmark, curvas de entrenamiento y métricas ligeramente diferentes.
- No garantiza determinismo absoluto entre diferentes versiones de librerías o hardware.

### Resolución objetivo

```python
TARGET_SIZE = (224, 224)
```

- MobileNetV2 está diseñado para trabajar normalmente con imágenes de 224×224.
- Una resolución menor reduce el costo, pero puede eliminar detalles importantes.
- Una resolución mayor incrementa memoria y tiempo sin garantizar una mejora.

### Uso de rutas relativas

```python
metadata_df["filepath"] = metadata_df["archivo"].map(
    lambda p: Path(p).as_posix()
)
```

Las rutas relativas permiten mover el proyecto completo sin guardar rutas absolutas específicas de una computadora.

---

## 2. Fase 1: exploración y validación de imágenes

### Descubrimiento de archivos

```python
class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
files = [p for p in class_dir.rglob("*") if p.suffix.lower() in extensions]
```

- `iterdir()` localiza las carpetas que representan clases.
- `rglob("*")` busca archivos incluso si existen subdirectorios.
- El filtro por extensión evita procesar archivos que claramente no son imágenes.

### Verificación de integridad con Pillow

```python
with Image.open(path) as verify_img:
    verify_img.verify()
```

`verify()` comprueba la estructura del archivo sin decodificarlo completamente. Después se abre nuevamente para leer píxeles, porque `verify()` deja el objeto de imagen inutilizable para operaciones posteriores.

Efecto sobre los resultados:

- Impide que archivos dañados ingresen al pipeline.
- En este dataset se verificaron 26.179 imágenes válidas y cero corruptas.

### Conversión a RGB

```python
rgb = img.convert("RGB")
```

- Fuerza tres canales en un orden consistente.
- Evita errores con imágenes en escala de grises, paleta o RGBA.
- El canal alfa se descarta; por ello, si existieran transparencias importantes, deberían componerse explícitamente sobre un fondo.

### Miniaturas para estadísticas de color

```python
preview = np.asarray(rgb.resize((64, 64)), dtype=np.uint8)
```

La reducción a 64×64 se usa únicamente para calcular estadísticas RGB de manera eficiente. No es la resolución usada por el modelo.

- Acelera el EDA y reduce memoria.
- Conserva tendencias generales de color.
- No debe utilizarse para estudiar texturas finas.

### Criterio IQR para imágenes grandes

```python
q1, q3 = df["pixeles"].quantile([0.25, 0.75])
iqr = q3 - q1
large_threshold = q3 + 1.5 * iqr
```

- `Q3 + 1.5×IQR` es una regla robusta para detectar valores extremos.
- Detectó 1.970 imágenes grandes.
- Ser un outlier no implica que la imagen sea incorrecta; por eso no se eliminó automáticamente.

### Criterio para imágenes pequeñas

```python
df[["ancho", "alto"]].min(axis=1) < 64
```

Marca imágenes cuyo lado menor es inferior a 64 píxeles. Se encontraron cinco. Ampliarlas a 224×224 puede generar desenfoque, pero se conservaron para evitar decisiones destructivas sin revisión manual.

---

## 3. Fase 2: duplicados, splits y preprocesamiento

### MD5 para duplicados exactos

```python
digest = hashlib.md5()
with open(path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
```

Parámetros:

- `chunk_size=1 MB` limita el uso de memoria al leer archivos.
- MD5 se usa como identificador de igualdad binaria, no con fines criptográficos.

Efecto:

- Dos archivos con el mismo MD5 son iguales byte a byte.
- No detecta imágenes visualmente iguales que fueron recomprimidas o redimensionadas.
- En el dataset se encontraron cero pares exactos.

### Average hash

```python
gray8 = np.asarray(
    img.convert("L").resize((8, 8), Image.Resampling.LANCZOS),
    dtype=np.float32,
)
ahash = binary_hash_to_hex(gray8 >= gray8.mean())
```

- Convierte la imagen a 8×8 en escala de grises.
- Cada bit indica si un píxel está sobre o bajo la media.
- Es rápido, pero sensible a cambios de iluminación y composición.

### Perceptual hash con DCT

```python
gray32 = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32)
low_freq = dctn(gray32, type=2, norm="ortho")[:8, :8]
threshold = np.median(low_freq.ravel()[1:])
phash = binary_hash_to_hex(low_freq >= threshold)
```

Parámetros:

- `32×32`: resolución usada para calcular frecuencias.
- `[:8, :8]`: conserva frecuencias bajas, que describen estructura general.
- Se excluye el primer coeficiente al calcular la mediana porque representa principalmente luminosidad global.

### Distancia de Hamming

```python
distance = (int(left, 16) ^ int(right, 16)).bit_count()
```

Cuenta cuántos bits difieren entre dos hashes.

El notebook considera un par potencial cuando:

```python
phash_distance <= 5 and ahash_distance <= 5
```

- Un umbral menor reduce falsos positivos, pero puede omitir duplicados modificados.
- Un umbral mayor aumenta sensibilidad, pero puede agrupar imágenes distintas.
- Con el valor 5 se detectaron 1.870 pares potenciales para revisión.

### Árbol BK

```python
tree.query(pvalue, radius=5)
```

El árbol BK busca hashes cercanos sin comparar todos los pares posibles. `radius=5` debe coincidir con el umbral perceptual elegido.

### Unión de duplicados en componentes

```python
union_find.union(left, right)
```

La estructura Union-Find agrupa relaciones transitivas. Si A es parecido a B y B a C, los tres forman un mismo `duplicate_group`, aunque A y C no hayan sido comparados directamente.

Esto fue fundamental para evitar que copias relacionadas aparecieran en splits diferentes.

### División estratificada por grupos

```python
train_groups, temp_groups = train_test_split(
    group_table,
    test_size=0.30,
    stratify=group_table["class_name"],
    random_state=42,
)

validation_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    stratify=temp_groups["class_name"],
    random_state=42,
)
```

Efecto de los parámetros:

- Primer `test_size=0.30`: reserva 30% para validation+test.
- Segundo `test_size=0.50`: divide ese 30% en 15% validation y 15% test.
- `stratify`: conserva aproximadamente la distribución de clases.
- `random_state=42`: hace reproducible la asignación.
- La unidad dividida es `duplicate_group`, no cada imagen individual.

Resultados:

- Train: 18.310 imágenes.
- Validation: 3.933 imágenes.
- Test: 3.936 imágenes.

### Verificación de leakage

```python
len(set(left_values) & set(right_values))
```

Se comprueban intersecciones de:

- rutas;
- MD5;
- grupos de duplicados;
- pares perceptuales reportados.

```python
if not leakage_pass:
    raise RuntimeError("Se detectó data leakage")
```

La excepción detiene el pipeline si aparece una fuga. El resultado fue `PASS`.

### Resize preservando aspecto

```python
def resize_with_padding(image, target_size=(224, 224), fill=(0, 0, 0)):
    image = image.convert("RGB")
    resized = ImageOps.contain(
        image,
        target_size,
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", target_size, fill)
    offset = (
        (target_size[0] - resized.width) // 2,
        (target_size[1] - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas
```

Parámetros:

- `target_size=(224,224)`: salida fija requerida por el modelo.
- `fill=(0,0,0)`: padding negro.
- `LANCZOS`: interpolación de alta calidad para reducción.

Efecto:

- Conserva la relación de aspecto y evita deformar animales.
- Puede introducir bordes negros; el modelo podría aprenderlos si están correlacionados con alguna clase.
- El resize directo fue rechazado porque estira imágenes no cuadradas.

### Pesos de clase

```python
weights = compute_class_weight(
    class_weight="balanced",
    classes=train_classes,
    y=train_labels,
)
```

La fórmula asigna mayor peso a clases con menos ejemplos.

- Reduce el sesgo hacia `cane` y `ragno`.
- Incrementa la importancia de errores en clases minoritarias como `elefante`.
- Pesos demasiado altos pueden aumentar falsos positivos de las clases pequeñas.

---

## 4. Fase 3: comparación de representaciones

## Benchmark computacional

```python
BENCHMARK_SIZES = {
    "train": 500,
    "validation": 200,
    "test": 200,
}
```

Se utilizó un benchmark estratificado porque aplanar 18.310 imágenes de 224×224×3 produciría una matriz extremadamente grande.

- Permite ejecutar PCA exacto en CPU.
- Hace justa la comparación porque A, B y C usan las mismas imágenes.
- Las métricas son evidencia comparativa, no una estimación definitiva sobre todo el dataset.

### Flatten

```python
array.reshape(-1) / 255.0
```

- Convierte cada imagen en 150.528 variables.
- `/255.0` transforma intensidades a `[0,1]`.
- Flatten elimina la organización explícita de filas, columnas y vecindades espaciales.

### StandardScaler

```python
pixel_scaler = StandardScaler(copy=False)
X_train = pixel_scaler.fit_transform(X_train)
X_validation = pixel_scaler.transform(X_validation)
X_test = pixel_scaler.transform(X_test)
```

- `fit_transform` se usa únicamente en train.
- Validation y test usan `transform`; nunca recalculan medias o desviaciones.
- `copy=False` reduce memoria permitiendo modificación en el mismo arreglo.

Si el scaler se ajustara con test, existiría leakage porque usaría estadísticas de la evaluación final.

### PCA al 95%

```python
pixel_pca = PCA(n_components=0.95, svd_solver="full")
```

Parámetros:

- `n_components=0.95`: selecciona automáticamente componentes que explican al menos 95% de la varianza.
- `svd_solver="full"`: calcula una descomposición exacta compatible con el umbral de varianza.

Resultado: 247 componentes y 95,0219% de varianza.

Consecuencias:

- Reduce 150.528 variables a 247.
- Acelera los clasificadores y reduce multicolinealidad.
- La varianza no equivale necesariamente a información discriminativa.

### Logistic Regression

```python
LogisticRegression(
    C=C,
    max_iter=3000,
    solver="lbfgs",
    random_state=42,
)
```

Parámetros:

- `C`: inverso de la regularización.
  - `C` pequeño: mayor regularización, coeficientes más contenidos.
  - `C` grande: modelo más flexible, con mayor riesgo de sobreajuste.
- `max_iter=3000`: permite que el optimizador alcance convergencia.
- `solver="lbfgs"`: optimizador adecuado para clasificación multiclase y datos densos.

Grid evaluado exclusivamente con validation:

```python
C_GRID = [0.01, 0.1, 1, 10]
```

Resultados:

- PCA + Logistic Regression eligió `C=0.1`.
- Embeddings MobileNetV2 + Logistic Regression eligió `C=10`.

### Linear SVM

```python
LinearSVC(
    C=C,
    max_iter=10000,
    dual="auto",
    random_state=42,
)
```

- `C` controla la penalización de errores frente al margen.
- `max_iter=10000` da margen para converger.
- `dual="auto"` permite a scikit-learn elegir la formulación adecuada según dimensiones y muestras.
- LinearSVC no produce probabilidades calibradas por defecto.

### Criterio de selección

```python
results.sort_values(
    ["macro_f1", "balanced_accuracy", "C"],
    ascending=[False, False, True],
)
```

Se prioriza Macro F1 debido al desbalance. Balanced accuracy rompe empates y, si persiste el empate, se prefiere el menor `C` por regularización.

### Carga de MobileNetV2 preentrenado

```python
backbone = MobileNetV2Model.from_pretrained(
    "google/mobilenet_v2_1.0_224"
).to(device).eval()

for parameter in backbone.parameters():
    parameter.requires_grad_(False)
```

- `from_pretrained`: carga pesos aprendidos en ImageNet.
- `.to(device)`: envía el modelo a CPU o GPU.
- `.eval()`: desactiva comportamiento de entrenamiento como dropout y actualización de BatchNorm.
- `requires_grad_(False)`: impide calcular gradientes y modificar pesos.

En Fase 3 MobileNetV2 no se entrenó: solo transformó imágenes en embeddings de 1.280 dimensiones.

### Inferencia sin gradientes

```python
with torch.inference_mode():
    embeddings = backbone(pixel_values=pixel_values).pooler_output
```

`inference_mode()` reduce memoria y acelera el cálculo al no construir el grafo de gradientes.

### Procesamiento oficial

```python
processor(
    images=images,
    return_tensors="pt",
    do_resize=False,
    do_center_crop=False,
)
```

- `return_tensors="pt"`: devuelve tensores PyTorch.
- `do_resize=False` y `do_center_crop=False`: evita que el procesador vuelva a modificar imágenes que ya fueron preparadas con padding 224×224.
- El procesador conserva la normalización esperada por MobileNetV2.

### Normalización L2 de embeddings

```python
X_mobile = normalize(embeddings, norm="l2")
```

Cada embedding queda con norma euclidiana igual a 1.

- Enfatiza la dirección semántica más que la magnitud.
- Mejora la estabilidad del clasificador lineal.
- Es parte del modelo definitivo; eliminarla cambia el espacio que recibe Logistic Regression.

### Caché de embeddings

```python
np.savez_compressed(
    cache_path,
    paths=ordered_paths,
    embeddings=all_embeddings,
)
```

La caché evita ejecutar MobileNetV2 repetidamente. Antes de reutilizarla se comparan las rutas guardadas con las esperadas; una caché desactualizada podría asociar embeddings con etiquetas incorrectas.

---

## 5. Fase 4: feature extraction y fine tuning

### Configuración de experimentos

```python
experiment_configs = {
    "A_feature_extraction": {"lr": 1e-3, "mode": "frozen"},
    "B_partial_finetuning": {"lr": 1e-4, "mode": "partial"},
    "C_full_finetuning": {"lr": 3e-5, "mode": "full"},
}
```

La tasa de aprendizaje disminuye al descongelar más capas:

- `1e-3`: apropiada para entrenar únicamente un clasificador nuevo.
- `1e-4`: ajuste conservador de los últimos bloques.
- `3e-5`: protege el backbone completo frente a cambios destructivos.

Una tasa demasiado alta puede borrar representaciones preentrenadas; una demasiado baja puede impedir aprendizaje dentro de las pocas épocas disponibles.

### Reemplazo del clasificador

```python
model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes,
)
```

Sustituye el clasificador ImageNet por una capa de diez salidas, una por clase Animal-10.

### Congelamiento total

```python
for parameter in model.mobilenet_v2.parameters():
    parameter.requires_grad_(False)
```

Solo la capa clasificadora recibe actualizaciones. Reduce costo y riesgo de sobreajuste.

### Fine tuning parcial

```python
if name.startswith(("layer.14", "layer.15", "conv_1x1")):
    parameter.requires_grad_(True)
```

Descongela los últimos bloques, que representan patrones más específicos. Las primeras capas, asociadas a bordes y texturas generales, permanecen congeladas.

Cambiar qué capas se descongelan modifica el equilibrio entre adaptación y sobreajuste.

### Fine tuning completo

```python
for parameter in model.mobilenet_v2.parameters():
    parameter.requires_grad_(True)
```

Todos los pesos pueden cambiar. Es la opción más costosa y sensible al tamaño del dataset y a la tasa de aprendizaje.

### BatchNorm congelado

```python
if parameters and not any(p.requires_grad for p in parameters):
    module.eval()
```

Evita que capas BatchNorm congeladas actualicen silenciosamente sus medias y varianzas durante entrenamiento.

### Data augmentation de train

```python
if random.random() < 0.5:
    image = ImageOps.mirror(image)

image = image.rotate(random.uniform(-15, 15))
image = zoom_image(image, random.uniform(1.0, 1.15))
image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
```

Parámetros:

- Flip con probabilidad 0,5.
- Rotación entre −15° y +15°.
- Zoom entre 1,0 y 1,15.
- Brillo y contraste entre 0,85 y 1,15.

Efecto:

- Aumenta variación sin crear nuevos archivos.
- Se aplica solo a train.
- Transformaciones más agresivas podrían cambiar la clase visual o introducir artefactos.

### DataLoader

```python
DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
```

- `batch_size=32`: compromiso entre estabilidad, memoria y velocidad.
- `shuffle=True`: cambia el orden de train cada época.
- Validation usa `shuffle=False` para evaluación estable.
- `num_workers=0`: lectura en el proceso principal; es más compatible con Windows, aunque puede ser más lenta.

### Función de pérdida ponderada

```python
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
```

La entropía cruzada mide la discrepancia entre logits y clase real. `weight` aumenta la contribución de clases minoritarias.

### Optimizador AdamW

```python
optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=config["lr"],
    weight_decay=1e-4,
)
```

- `lr`: tamaño de actualización.
- `weight_decay=1e-4`: regularización que desalienta pesos excesivamente grandes.
- AdamW desacopla correctamente weight decay de la actualización adaptativa.

### Ciclo de entrenamiento

```python
optimizer.zero_grad(set_to_none=True)
logits = model(pixel_values=pixels).logits
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

Orden:

1. `zero_grad`: elimina gradientes anteriores.
2. Forward: calcula logits.
3. Loss: calcula error.
4. `backward`: obtiene gradientes.
5. `step`: actualiza parámetros entrenables.

`set_to_none=True` reduce operaciones y memoria frente a llenar gradientes con ceros.

### Early stopping

```python
MAX_EPOCHS = 6
PATIENCE = 2

if epoch >= 3 and stale >= PATIENCE:
    break
```

- Máximo de seis épocas.
- Se detiene después de dos épocas sin mejora, pero no antes de la tercera.
- Reduce costo y sobreajuste.
- Una paciencia mayor podría encontrar mejoras tardías, pero también entrenar de más.

### Selección de checkpoint

```python
improved = (
    val_macro_f1 > best_score
    or (
        val_macro_f1 == best_score
        and val_balanced_accuracy > best_balanced
    )
)
```

Solo validation decide qué época guardar. Test se crea después de fijar experimento y checkpoint.

Resultado: el fine tuning no superó al modelo de embeddings + Logistic Regression, por lo que no se convirtió en el modelo definitivo.

---

## 6. Métricas de evaluación

### Accuracy

```python
accuracy_score(y_true, y_pred)
```

Proporción total correcta. Puede ser optimista cuando existen clases mayoritarias.

### Balanced accuracy

```python
balanced_accuracy_score(y_true, y_pred)
```

Promedio del recall de todas las clases. Cada clase tiene la misma importancia, independientemente de su tamaño.

### Métricas macro

```python
precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_score(y_true, y_pred, average="macro", zero_division=0)
```

- `average="macro"`: calcula la métrica por clase y luego promedia sin ponderar por soporte.
- `zero_division=0`: devuelve cero si una clase no tiene predicciones, evitando una excepción.
- Macro F1 fue la métrica principal por el desbalance 3,36:1.

### Matriz de confusión

```python
confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)
```

`labels` fija el orden de filas y columnas, incluso si una clase no aparece en una muestra.

Normalización por fila:

```python
normalized = matrix / matrix.sum(axis=1, keepdims=True)
```

La diagonal normalizada equivale al recall de cada clase.

### Reporte por clase

```python
classification_report(
    y_test,
    predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
```

- `output_dict=True` permite convertir el reporte en DataFrame.
- Se identificó `ragno` como la mejor clase y `mucca` como la más difícil.

---

## 7. Fase 5: interpretabilidad

### Probabilidades de Logistic Regression

```python
probabilities = classifier.predict_proba(embeddings)
prediction = classifier.classes_[np.argmax(probabilities, axis=1)]
confidence = probabilities.max(axis=1)
```

- `predict_proba` devuelve una distribución por clase.
- La confianza es la probabilidad de la clase elegida.
- Alta confianza no garantiza que la predicción sea correcta ni que la probabilidad esté perfectamente calibrada.

### PCA para visualización

```python
PCA(n_components=2, random_state=42).fit_transform(embeddings)
```

Reduce a dos dimensiones lineales. Es reproducible y preserva la mayor varianza posible, pero puede ocultar relaciones no lineales.

### t-SNE

```python
TSNE(
    n_components=2,
    perplexity=35,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42,
)
```

Parámetros:

- `perplexity=35`: tamaño aproximado del vecindario considerado.
- `learning_rate="auto"`: adapta la tasa al número de muestras.
- `init="pca"`: inicio más estable que una posición totalmente aleatoria.
- `max_iter=1000`: número de iteraciones de optimización.
- `random_state=42`: proyección reproducible.

t-SNE preserva vecindarios locales, no distancias globales. Dos grupos alejados en la figura no necesariamente están igual de alejados en 1.280 dimensiones.

### Centroides

```python
centroid = embeddings[labels == class_id].mean(axis=0)
cosine_distances(centroids)
```

Los centroides resumen la posición promedio por clase. La distancia coseno reveló proximidad entre `cavallo` y `mucca`, coherente con sus confusiones.

### Silhouette

```python
silhouette_score(embeddings, labels, metric="cosine")
```

- Valores altos indican grupos compactos y separados.
- Valores cercanos a cero sugieren solapamiento.
- Valores negativos sugieren muestras posiblemente más cercanas a otra clase.

### Grad-CAM para el pipeline real

El notebook convierte los coeficientes de Logistic Regression en una capa lineal fija:

```python
self.linear.weight.copy_(torch.tensor(classifier.coef_))
self.linear.bias.copy_(torch.tensor(classifier.intercept_))
```

Después conecta un hook a la última convolución:

```python
target_layer = backbone.conv_1x1.convolution
handle = target_layer.register_forward_hook(forward_hook)
```

Grad-CAM calcula:

```python
weights = gradient.mean(dim=(1, 2), keepdim=True)
cam = torch.relu((weights * activation).sum(dim=0))
```

- Los gradientes indican sensibilidad de la clase respecto a cada mapa de activación.
- El promedio espacial produce un peso por canal.
- `ReLU` conserva contribuciones positivas.
- El mapa se interpola a 224×224 para superponerlo sobre la imagen.

Limitación: Grad-CAM muestra regiones influyentes, no una explicación causal. Tampoco prueba que el modelo ignore completamente el fondo.

---

## 8. Parámetros que más afectan los resultados

| Parámetro | Valor usado | Efecto principal |
|---|---:|---|
| Resolución | 224×224 | Equilibra detalle y costo; coincide con MobileNetV2. |
| Semilla | 42 | Hace reproducibles splits, muestras e inicializaciones. |
| Umbral pHash/aHash | 5 bits | Controla sensibilidad de duplicados potenciales. |
| Split | 70/15/15 | Separa entrenamiento, selección y evaluación final. |
| PCA | 95% | Conserva varianza reduciendo 150.528 variables a 247. |
| Grid de `C` | 0.01, 0.1, 1, 10 | Controla regularización de LR y SVM. |
| `C` definitivo | 10 | Menor regularización sobre embeddings normalizados. |
| Dimensión embedding | 1.280 | Representación semántica extraída por MobileNetV2. |
| Batch size | 32 | Compromiso entre memoria y estabilidad. |
| LR congelado | 1e-3 | Entrena rápidamente el head. |
| LR parcial | 1e-4 | Ajusta últimas capas con cautela. |
| LR completo | 3e-5 | Reduce destrucción de pesos preentrenados. |
| Weight decay | 1e-4 | Regulariza el entrenamiento neuronal. |
| Épocas máximas | 6 | Limita costo y sobreajuste. |
| Patience | 2 | Detiene entrenamiento tras ausencia de mejora. |
| t-SNE perplexity | 35 | Controla escala local de la visualización. |

---

## 9. Orden correcto de uso de los segmentos

```text
TRAIN
  ├─ Ajusta scaler y PCA
  ├─ Entrena clasificadores
  ├─ Ajusta el head o las capas descongeladas
  └─ Recibe data augmentation

VALIDATION
  ├─ Selecciona C
  ├─ Selecciona experimento A/B/C
  ├─ Selecciona checkpoint
  └─ No actualiza parámetros directamente durante cada época

TEST
  ├─ No ajusta scaler ni PCA
  ├─ No selecciona hiperparámetros
  ├─ No selecciona checkpoint
  └─ Evalúa e interpreta el modelo ya fijado
```

Usar test para elegir `C`, decidir capas o seleccionar épocas convertiría la evaluación final en parte del entrenamiento y produciría métricas optimistas.

---

## 10. Modelo definitivo

```text
Imagen
  ↓
Conversión RGB
  ↓
Resize con padding 224×224
  ↓
Normalización oficial MobileNetV2
  ↓
MobileNetV2 preentrenado y congelado
  ↓
Embedding de 1.280 dimensiones
  ↓
Normalización L2
  ↓
Logistic Regression (C=10)
```

El componente entrenado del modelo definitivo es Logistic Regression. MobileNetV2 permanece congelado y funciona como extractor de características.

El resultado de la auditoría sobre el test completo fue:

- Accuracy: 0,9339.
- Balanced Accuracy: 0,9200.
- Macro Precision: 0,9396.
- Macro Recall: 0,9200.
- Macro F1: 0,9290.

La evidencia del notebook indica que una mejor representación produjo más valor que aumentar la complejidad mediante fine tuning.
