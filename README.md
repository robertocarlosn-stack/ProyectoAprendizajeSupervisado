# Proyecto de Aprendizaje Supervisado: Animal-10

Proyecto académico de clasificación multiclase de imágenes desarrollado sobre el dataset Animal-10. El trabajo cubre exploración, preparación reproducible, prevención de data leakage, comparación de representaciones, fine tuning e interpretabilidad.

## Modelo definitivo

```text
Imagen RGB
  → resize con padding 224×224
  → MobileNetV2 preentrenado y congelado
  → embedding de 1.280 dimensiones
  → normalización L2
  → Logistic Regression (C=10)
```

En la auditoría sobre las 3.936 imágenes del conjunto de test completo se obtuvo:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,9339 |
| Balanced Accuracy | 0,9200 |
| Macro Precision | 0,9396 |
| Macro Recall | 0,9200 |
| Macro F1 | 0,9290 |

## Fases del proyecto

1. **EDA:** 26.179 imágenes válidas, 10 clases, análisis de balance, dimensiones, color y calidad.
2. **Preprocesamiento:** hashes MD5/aHash/pHash, agrupación de duplicados, split estratificado y control de leakage.
3. **Representaciones:** comparación de PCA + Logistic Regression, PCA + SVM y embeddings MobileNetV2.
4. **Fine tuning:** comparación de backbone congelado, ajuste parcial y ajuste completo.
5. **Interpretabilidad:** matriz de confusión, errores, métricas por clase, PCA, t-SNE y Grad-CAM.

## Decisiones principales

- Se seleccionó resize con padding para conservar la relación de aspecto.
- Los grupos de duplicados permanecen dentro de un único split.
- El leakage check fue `PASS`.
- MobileNetV2 congelado superó ampliamente las representaciones de píxeles aplanados.
- El fine tuning no justificó reemplazar el extractor congelado.
- El modelo definitivo utiliza embeddings MobileNetV2 y Logistic Regression.

El mapa completo de decisiones está disponible en:

- [`animal10_decision_story.png`](animal10_decision_story.png)
- [`animal10_decision_story.svg`](animal10_decision_story.svg)

## Archivos principales

- [`animal10_project.ipynb`](animal10_project.ipynb): notebook completo y ejecutado.
- [`avance.md`](avance.md): resultados y decisiones por fase.
- [`development.md`](development.md): explicación de comandos, parámetros y efectos.
- `train.csv`, `validation.csv`, `test.csv`: manifiestos de los splits.
- `phase3_metrics.csv`: comparación de representaciones.
- `phase4_validation_results.csv`: resultados de fine tuning.
- `phase5_per_class_metrics.csv`: métricas finales por clase.
- `phase5_test_predictions.csv`: predicciones de la auditoría final.

## Dataset

El directorio `data/` no se publica en el repositorio debido a su tamaño y a que el dataset debe obtenerse desde Kaggle.

La estructura esperada es:

```text
data/
├── cane/
├── cavallo/
├── elefante/
├── farfalla/
├── gallina/
├── gatto/
├── mucca/
├── pecora/
├── ragno/
└── scoiattolo/
```

## Instalación

Se recomienda Python 3.12 y un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para abrir el notebook:

```powershell
python -m pip install jupyter
jupyter notebook animal10_project.ipynb
```

## Reproducibilidad

- Semilla global: `42`.
- Resolución: `224×224` RGB.
- Split aproximado: 70% train, 15% validation y 15% test.
- Test no se utilizó para seleccionar hiperparámetros.
- Los embeddings y checkpoints se excluyen porque pueden regenerarse ejecutando el notebook.

## Autor

Roberto Carlos N. — [robertocarlosn-stack](https://github.com/robertocarlosn-stack)

