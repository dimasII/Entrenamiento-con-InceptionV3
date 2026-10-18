# Clasificación de residuos con InceptionV3 (TensorFlow/Keras)

Proyecto de clasificación de imágenes de residuos en 12 clases con
transferencia de aprendizaje (InceptionV3 preentrenado en ImageNet,
`include_top=False`, imágenes RGB 299×299, `preprocess_input` propio).

> ESTADO (17-oct-2026): **experimentos terminados; mejor modelo = fine-tuning
> de 80 capas (`exp_f_cap80_lr1e-5`)**
> - **Validación:** acc 0,9677 · F1 macro 0,9532
> - **Test (evaluado una sola vez):** acc **0,9641** · F1 macro **0,9481**
>
> Mejora sobre el baseline (40 capas): test F1 macro 0,9287 → **0,9481**.
> Detalles y conclusiones en §11 y §12. El modelo oficial es
> `results_tf/exp_f_cap80_lr1e-5/best_stage2.keras` (copiado a `best_model.keras`).
>
> ⚠️ Los pesos `.keras` **no están en el repositorio** (superan el límite de
> 100 MB/archivo de GitHub): se descargan del
> [Release v1.0.0](https://github.com/dimasII/Entrenamiento-con-InceptionV3/releases/tag/v1.0.0).
> Ver §13.

## Solución lograda: TensorFlow usando la RTX 3050

Se configuró TensorFlow GPU en Ubuntu sobre WSL2. La GPU quedó detectada por
TensorFlow 2.21.0 en el entorno `tf-gpu312` (Python 3.12.15). La verificación
final devolvió:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

La ruta de las bibliotecas CUDA y la del controlador WSL se añadieron a
`LD_LIBRARY_PATH` en `~/.bashrc`, por lo que la configuración persiste al
abrir nuevas terminales de Ubuntu.

### Cómo volver a usar la GPU

Desde PowerShell o Windows Terminal, entrar en Ubuntu:

```powershell
wsl -d Ubuntu
```

Luego, dentro de Ubuntu, activar el entorno del proyecto y comprobar la GPU:

```bash
cd "/mnt/c/Users/dimas/OneDrive/Documentos/26-2/IA 2/Entrenamiento"
source ~/venvs/tf-gpu312/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

El prompt debe mostrar `(tf-gpu312)` y la salida debe incluir
`/physical_device:GPU:0`. TensorFlow instalado directamente en Windows
nativo seguirá usando CPU; para usar la RTX, ejecutar el proyecto desde Ubuntu
en WSL2. La evaluación ya se realizó (`python train_tf.py evaluate`, ver §8);
ahora continúa la batería de experimentos de §11.

## 1. Bitácora de lo realizado

1. **Inspección del dataset real** antes de escribir código: estructura de
   carpetas, conteo por clase, extensiones, validez de imágenes (PIL) y
   duplicados exactos (MD5). Ver §3.
2. **Verificación del entorno**: TensorFlow en Windows nativo no detectaba
   la RTX 3050; se configuró Ubuntu en WSL2 y TensorFlow con GPU. Ver §4.
3. **Creación de `train_tf.py`** (sin tocar el `train.py` de PyTorch que ya
   existía): split estratificado sin fuga, pesos de clase, aumento solo en
   train, 2 etapas con BN congeladas, callbacks, selección por F1 macro de
   validación, test evaluado una vez.
4. **Paso a paso (CRUD por pasos)**: `train_tf.py` acepta un paso como
   argumento (`split`, `train1`, `train2`, `evaluate`, `full`). Cada paso
   guarda sus resultados en `results_tf/` y el siguiente los reutiliza.
   Ver §5 y §6.
5. **Creación de `predict_tf.py`**: inferencia con idéntico
   preprocesamiento, top-K, umbral configurable.
6. **Verificaciones ejecutadas**: `py_compile` OK, `--help` OK,
   `split --check-only` OK (lote 32×299×299×3 → logits 32×12),
   `evaluate --model noexiste.keras` → error controlado. Ver §7.
7. **Entrenamiento etapa 1 ejecutado por el usuario** (10 épocas en CPU,
   ~85 min): guardado `best_stage1.keras` + `historial_etapa1.csv`.
8. Limpieza de scripts temporales (`_tmp_*.py` eliminados).
9. **Sesión 17-oct-2026**: se instaló Ubuntu en WSL2, se verificó acceso a
   la RTX 3050 y se configuró TensorFlow 2.21.0 con GPU en Python 3.12.15.
   Ver §4.
10. **Corrección del paso `evaluate` en WSL2**: falló por dos motivos que se
    corrigieron: (a) `results_tf/splits.json` guardaba rutas con separador de
    Windows (`\`), que Linux no resuelve → normalizadas a `/` conservando el
    split intacto (respaldo en `splits_windows_backup.json`); (b) `train_tf.py`
    buscaba `best_etapa2.keras` en vez de `best_stage2.keras` → corregido para
    copiar la ruta real del modelo elegido.
11. **Evaluación en test ejecutada** (paso `evaluate`): generó
    `best_model.keras`, `metricas_test.json`, `reporte_por_clase.csv` y
    `confusion_matrix.csv`. Resultado: acc **0,9509**, F1 macro **0,9277** (§8).
12. **Scripts de experimentación**: `experimentos_etapa2.py` y
    `comparar_test.py` tenían el docstring con una ruta de Windows que provocaba
    `SyntaxError` (`\U`) → marcados como raw strings (`r"""`). Se añadió
    `experimentos_etapa1.py` (variación de épocas de la etapa 1).
13. **Batería de experimentos** para buscar mejores resultados (§11).
14. **Resultados de los experimentos**: el grid de fine-tuning mostró que
    **descongelar 80 capas** es lo mejor; más épocas de etapa 1 no aportan.
15. **Mejor modelo final**: `exp_f_cap80_lr1e-5` (80 capas, 15 épocas) →
    val F1 0,9532 · **test F1 0,9481** (frente a 0,9287 del baseline). Es el
    modelo ganador y quedó copiado a `results_tf/best_model.keras`.
16. **Ampliación de `train_tf.py`**: se añadieron las opciones CLI
    `--fine-tune-layers` y `--dropout` (antes solo se podía tocar el `CONFIG`).

## 2. Archivos del proyecto

| Archivo / carpeta | Origen | Descripción |
|---|---|---|
| `garbage_classification/` | dataset | 12 subcarpetas, una por clase (§3) |
| `train.py` | preexistente (PyTorch) | Flujo alternativo; **no se usa ni se modifica** |
| `train_tf.py` | creado | Entrenamiento TF/Keras por pasos o completo |
| `predict_tf.py` | creado | Inferencia de una imagen nueva |
| `experimentos_etapa2.py` | creado | Grid de fine-tuning (capas · LR · dropout) |
| `experimentos_etapa1.py` | creado | Variación de épocas de la etapa 1 |
| `comparar_test.py` | creado | Compara varios modelos en test |
| `matriz_confusion.py` | creado | Genera `confusion_matrix.png` desde `metricas_test.json` |
| `revisar_duplicados.py` | creado | Lista duplicados exactos que cruzan clases |
| `results_tf/splits_windows_backup.json` | creado | Respaldo del split original (rutas con `\`) |
| `class_names.json` | generado (paso `split`) | Lista de 12 clases en orden de índices |
| `results_tf/` | generado | Ver tabla de §6 (qué deja cada paso) |
| `README.md` | creado | Este documento |

## 3. Inspección del dataset (resultados reales)

Estructura encontrada = estructura esperada (no hubo que adaptar nada):
12 subcarpetas, 15.515 archivos, todos `.jpg`, **0 archivos inválidos**
(verificados con PIL uno por uno).

| Clase | Imágenes | % | Peso de clase (train) |
|---|---|---|---|
| clothes | 5325 | 34,3 | 0,243 |
| shoes | 1977 | 12,7 | 0,654 |
| paper | 1050 | 6,8 | 1,230 |
| biological | 985 | 6,3 | 1,312 |
| battery | 945 | 6,1 | 1,368 |
| cardboard | 891 | 5,7 | 1,451 |
| plastic | 865 | 5,6 | 1,495 |
| white-glass | 775 | 5,0 | 1,668 |
| metal | 769 | 5,0 | 1,681 |
| trash | 697 | 4,5 | 1,857 |
| green-glass | 629 | 4,1 | 2,064 |
| brown-glass | 607 | 3,9 | 2,133 |
| **TOTAL** | **15515** | 100 | — |

- Desbalanceo fuerte: `clothes` (34 %) vs `brown-glass` (3,9 %).
  Compensado con pesos (`total / (12 × n_clase)`, SOLO con train) +
  aumento moderado solo en train. Sin duplicar archivos.
- Tamaños heterogéneos: 892 tamaños distintos (top: 400×533, 512×384,
  225×225). Todo se redimensiona a 299×299.
- Duplicados exactos (MD5): **18 grupos / 19 archivos extra**, misma
  partición (sin fuga). **3 grupos cruzan clases** (revisados con
  `revisar_duplicados.py`), todos con `white-glass`:
  - `metal/metal459.jpg` = `white-glass/white-glass412.jpg`
  - `plastic/plastic159.jpg` = `white-glass/white-glass129.jpg`
  - `plastic/plastic556.jpg` = `white-glass/white-glass761.jpg`
  Es posible etiquetado cruzado (la misma foto asignada a dos clases); puede
  explicar parte de la confusión vidrio / plástico / metal.
- Split obtenido: train 10851 (69,9 %), val 2321 (15,0 %),
  test 2343 (15,1 %).

## 4. Entorno

- Windows 11 + PowerShell. GPU física RTX 3050 Laptop (4 GB).
- **TensorFlow 2.11 o posterior no admite CUDA en Windows nativo.**
  `tensorflow[and-cuda]` debe instalarse dentro de Linux/WSL2 para usar GPU;
  instalarlo en este entorno Windows no hará que TensorFlow detecte la RTX.
  Referencia: https://www.tensorflow.org/install/pip#windows-native
- WSL2 con Ubuntu tiene acceso a la RTX 3050 (`nvidia-smi` la muestra).
- Entorno TensorFlow GPU: `~/venvs/tf-gpu312`, Python 3.12.15,
  TensorFlow 2.21.0 instalado con `tensorflow[and-cuda]`.
- Verificación final en WSL2: `tf.config.list_physical_devices('GPU')` →
  `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.
- **Nota WSL (importante):** `LD_LIBRARY_PATH` se define en `~/.bashrc`, así que
  **solo aplica en terminales interactivas**. Al ejecutar `bash -lc "..."` de
  forma no interactiva, `~/.bashrc` termina antes de definir la variable y
  TensorFlow no ve la GPU; hay que exportar `LD_LIBRARY_PATH` explícitamente.
- La instalación inicial usó Python 3.14, que no tenía wheel compatible con
  TensorFlow 2.21. Se instaló Python 3.12 con `uv` y se creó un nuevo venv.
- En esta instalación fue necesario enlazar las bibliotecas NVIDIA del extra
  CUDA dentro del directorio de TensorFlow y añadir las rutas a
  `LD_LIBRARY_PATH`. El ajuste persistente se guardó en `~/.bashrc`.
- `matplotlib` 3.11.2 instalado en `tf-gpu312` (con `uv pip`) para generar la
  matriz de confusión.

### Configuración de WSL2 y TensorFlow GPU

#### A. Habilitar WSL e instalar Ubuntu (PowerShell)

Abrir PowerShell como administrador (clic derecho en Inicio → Terminal o
PowerShell (Administrador)) y ejecutar:

```powershell
wsl --install -d Ubuntu
```

Si Windows informa que habilitó características y que los cambios se aplicarán
al reiniciar, reiniciar el equipo. Después, abrir de nuevo PowerShell como
administrador y ejecutar `wsl --install -d Ubuntu` para completar la
instalación de la distribución si todavía no está instalada. Comprobar el
estado desde PowerShell:

```powershell
wsl --status
wsl --list --verbose
```

Debe aparecer Ubuntu en la lista. Para iniciarlo, hay dos opciones:

- Abrir el menú Inicio, buscar **Ubuntu** y abrir la aplicación.
- Desde PowerShell o Windows Terminal, ejecutar `wsl -d Ubuntu`.

La primera vez Ubuntu solicita crear un nombre de usuario y una contraseña
para Linux. Al escribir la contraseña, la terminal no muestra caracteres; es
normal. Cuando termine la configuración, el prompt cambia a una forma similar
a `dimas@Dimas:~$`. Ese prompt indica que ya se está dentro de Ubuntu.

En esta configuración, `nvidia-smi` dentro de Ubuntu mostró la RTX 3050. Si
`wsl --install` indica que requiere elevación, usar PowerShell como
administrador; si indica que los cambios se aplicarán tras reiniciar, reiniciar
antes de volver a ejecutar el comando. Si `wsl -d Ubuntu` responde
`WSL_E_DISTRO_NOT_FOUND`, Ubuntu aún no está instalado: volver a PowerShell
como administrador y completar `wsl --install -d Ubuntu`. No ejecutar
`sudo` ni `apt` en PowerShell.

#### B. Preparar Ubuntu y comprobar la GPU

Los siguientes comandos se ejecutan dentro de la terminal Ubuntu (el prompt
Linux termina en `$`):

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
nvidia-smi  # Debe mostrar la RTX 3050
```

#### C. Instalar TensorFlow GPU

Python 3.14, que venía como `python3` en esta instalación de Ubuntu, no tenía
un wheel compatible con TensorFlow 2.21. Se usó `uv` para instalar Python 3.12
sin cambiar el Python del sistema:

```bash
# Crear un entorno auxiliar e instalar uv.
mkdir -p ~/venvs
python3 -m venv ~/venvs/tf-gpu
source ~/venvs/tf-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
uv python install 3.12
uv venv --python 3.12 ~/venvs/tf-gpu312

# Instalar TensorFlow 2.21 y sus dependencias CUDA en el nuevo entorno.
/home/$USER/venvs/tf-gpu/bin/uv pip install \
  --python ~/venvs/tf-gpu312/bin/python \
  'tensorflow[and-cuda]==2.21.0'
source ~/venvs/tf-gpu312/bin/activate

# TensorFlow necesita encontrar las bibliotecas CUDA instaladas por pip.
TF_DIR="$(python -c 'import tensorflow as tf, os; print(os.path.dirname(tf.__file__))')"
pushd "$TF_DIR" >/dev/null
ln -svf ../nvidia/*/lib/*.so* .
popd >/dev/null
NVCC_DIR="$(python -c 'import nvidia.cuda_nvcc; print(next(iter(nvidia.cuda_nvcc.__path__)))')"
PTXAS="$(find "$NVCC_DIR" -type f -name ptxas -print -quit)"
if [ -n "$PTXAS" ]; then ln -sf "$PTXAS" "$VIRTUAL_ENV/bin/ptxas"; fi

# Hacer persistentes las rutas para nuevas terminales Ubuntu.
echo 'export LD_LIBRARY_PATH="$HOME/venvs/tf-gpu312/lib/python3.12/site-packages/tensorflow:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> ~/.bashrc
source ~/.bashrc

# Verificar que TensorFlow ya detecta la GPU.
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

La verificación final mostró TensorFlow `2.21.0` y
`PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`.

#### D. Usar el entorno desde Ubuntu

Para cada sesión nueva:

1. Abrir Ubuntu desde Inicio o ejecutar `wsl -d Ubuntu` en PowerShell.
2. Confirmar que el prompt es Linux (termina en `$`, por ejemplo
   `dimas@Dimas:~$`), no PowerShell (`PS C:\...`).
3. Desde Ubuntu, entrar al proyecto montado desde OneDrive, activar el entorno
   y comprobar la GPU:

```bash
cd "/mnt/c/Users/dimas/OneDrive/Documentos/26-2/IA 2/Entrenamiento"
source ~/venvs/tf-gpu312/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

El prompt debe indicar `(tf-gpu312)`. La salida debe incluir
`PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`. En esa
misma terminal ya se pueden ejecutar los comandos del proyecto, por ejemplo
`python train_tf.py evaluate`. Cerrar Ubuntu termina esa sesión; para volver,
abrir Ubuntu otra vez y repetir los pasos anteriores.

## 5. Modo paso a paso (CRUD de pasos)

El primer argumento de `train_tf.py` es el paso (si se omite = `full`).
Las opciones (`--batch-size`, `--epochs1/2`, etc.) valen **antes o
después** del paso.

| Paso | Comando | Tarda (CPU) | Entra | Sale (en `results_tf/`) |
|---|---|---|---|---|
| 1 split | `train_tf.py split` | ~2 min | dataset | `splits.json`, `config.json`, `duplicates_report.json`, `class_names.json` (+ copia) |
| 2 etapa 1 | `train_tf.py train1` | ~1,5 h | `splits.json` | `best_stage1.keras`, `historial_etapa1.csv`, `val_etapa1.json` |
| 3 etapa 2 | `train_tf.py train2 --batch-size 16` | ~2–3 h | `splits.json` + `best_stage1.keras` | `best_stage2.keras`, `historial_etapa2.csv`, `val_etapa2.json` |
| 4 evaluar | `train_tf.py evaluate` | ~15–30 min | mejor `.keras` + métricas val | `best_model.keras`, `metricas_test.json`, `reporte_por_clase.csv`, `confusion_matrix.csv`/`.png` |
| todo | `train_tf.py full` o sin paso | ~5 h | dataset | todo lo anterior |

Secuencia original en PowerShell (CPU):

```powershell
conda activate ia2
cd "C:\Users\dimas\OneDrive\Documentos\26-2\IA 2\Entrenamiento"
$py = "C:\Users\dimas\miniconda3\envs\ia2\python.exe"

& $py train_tf.py split                  # paso 1 (HECHO)
& $py train_tf.py train1                 # paso 2 (HECHO)
& $py train_tf.py train2 --batch-size 16 # paso 3 (HECHO)
& $py train_tf.py evaluate               # paso 4 (HECHO en WSL2/GPU)
```

Para usar la GPU, ejecutar el paso desde Ubuntu/WSL2 en el directorio del
proyecto, con `tf-gpu312` activo. Las etapas 1 y 2 **y la evaluación** ya están
completadas (la evaluación se ejecutó así):

```bash
cd "/mnt/c/Users/dimas/OneDrive/Documentos/26-2/IA 2/Entrenamiento"
source ~/venvs/tf-gpu312/bin/activate
python train_tf.py evaluate
```

Utilidades:

```powershell
& $py train_tf.py split --check-only  # verifica pipeline sin entrenar
& $py train_tf.py split --re-split    # recalcula split (cambia seed/ratios)
& $py train_tf.py evaluate --model results_tf\best_stage1.keras  # evalua un modelo dado
& $py train_tf.py train2 --batch-size 16 *> log_etapa2.txt 2>&1  # etapa 2 con log
& $py predict_tf.py garbage_classification\paper\paper1.jpg      # inferencia
```

## 6. Dónde configurar (bloque `CONFIG` en `train_tf.py`)

- `stage1_epochs` (10) / `stage2_epochs` (15).
- `stage1_lr` (1e-3; rango 1e-4–1e-3), `stage2_lr` (1e-5; BAJA siempre).
- `fine_tune_layers` (40): últimas N capas en etapa 2, BN congeladas.
- `dropout` (0,3), `patience_es` (5), `patience_lr` (3).
- `batch_size` (16 por defecto, punto de partida para GPU de 4 GB), `seed`
  (42), `train_ratio`/`val_ratio`.
- Todo sobreescribible por CLI sin editar el archivo, incluidas
  `--fine-tune-layers`, `--dropout`, `--epochs1/2`, `--batch-size` y `--seed`.

## 7. Verificaciones ejecutadas (todas OK)

```powershell
& $py -m py_compile train_tf.py predict_tf.py  # sintaxis OK
& $py train_tf.py --help                        # subcomandos OK
& $py train_tf.py split --check-only            # pipeline real OK:
#  Splits reutilizados (train=10851, val=2321, test=2343)
#  CHECK OK: batch (32, 299, 299, 3) -> logits (32, 12); softmax suma=1.000
& $py train_tf.py evaluate --model noexiste.keras  # error controlado:
#  FileNotFoundError: No existe el modelo: noexiste.keras
```

## 8. Resultados reales hasta ahora

**Etapa 1 (completada 15-oct-2026, 10 épocas, ~85 min CPU):**

| Época | train acc | train loss | val acc | val loss |
|---|---|---|---|---|
| 1 | 0,8282 | 0,7604 | 0,8147 | 0,6406 |
| 4 | 0,9283 | 0,3313 | 0,8802 | 0,3654 |
| 8 | 0,9477 | 0,2452 | **0,9087** | 0,2885 |
| 10 | 0,9487 | 0,2319 | 0,8897 | 0,3519 |

Mejor: época 8 → `best_stage1.keras` (84,5 MB), val acc **0,9087**,
F1 macro val **0,8485**. Leve overfitting al final (val_loss sube en
9–10): el checkpoint guardó la época 8, correcto.

**Etapa 2 (completada, 15 épocas en CPU):**

| Época | train acc | train loss | val acc | val loss |
|---|---|---|---|---|
| 1 | 0,9551 | 0,1800 | 0,9130 | 0,2646 |
| 5 | 0,9736 | 0,1026 | 0,9362 | 0,2028 |
| 8 | 0,9775 | 0,0856 | 0,9436 | 0,1842 |
| 15 | 0,9906 | 0,0325 | **0,9539** | 0,1663 |

Mejor: época 15 → `best_stage2.keras`, val acc **0,9539**,
F1 macro val **0,9294**. Supera a la etapa 1 en todas las métricas;
es el modelo ganador por F1 macro de validación.

> Resultados de etapa 2 guardados en `historial_etapa2.csv` y `val_etapa2.json`.

**Mejor modelo global:** `best_stage2.keras` (por F1 macro val 0,9294 >
0,8485 de la etapa 1).

**Evaluación en test (ejecutada una sola vez, 17-oct-2026):**

| Métrica | Validación | Test |
|---|---|---|
| Accuracy | 0,9539 | **0,9509** |
| F1 macro | 0,9294 | **0,9277** |

Peores clases en test: `brown-glass` (F1 0,8199; 18 de 92 se confunden con
`green-glass`), `green-glass` (0,8585) y `white-glass` (0,8960). El punto
débil son los **vidrios**. Detalle en `results_tf/metricas_test.json` y
`results_tf/reporte_por_clase.csv`.

**Modelo mejorado (mejor resultado de los experimentos, §11):**

`exp_f_cap80_lr1e-5` — fine-tuning descongelando 80 capas, 15 épocas:

| Métrica | Baseline (cap40) | Mejor (cap80) |
|---|---|---|
| val acc | 0,9539 | **0,9677** |
| val F1 macro | 0,9294 | **0,9532** |
| test acc | 0,9509 | **0,9641** |
| test F1 macro | 0,9277 | **0,9481** |

La mejora se concentra en las clases difíciles (test): `brown-glass` 0,82 → 0,90,
`green-glass` 0,86 → 0,89, `white-glass` 0,90 → 0,92. Los archivos
`results_tf/reporte_por_clase.csv` y `results_tf/confusion_matrix.csv`
corresponden a este modelo.

## 9. Inferencia (`best_model.keras` ya existe)

> `best_model.keras` es la copia del modelo elegido al ejecutar `evaluate`.
> Actualmente es **`exp_f_cap80_lr1e-5`** (fine-tuning de 80 capas), el mejor
> modelo del proyecto (test F1 0,9481).

```powershell
# Windows (PowerShell, entorno ia2)
& $py predict_tf.py garbage_classification\paper\paper1.jpg
& $py predict_tf.py ruta\imagen.jpg --top-k 5 --threshold 0.6
```

```bash
# Ubuntu/WSL2 (entorno tf-gpu312)
python predict_tf.py garbage_classification/paper/paper1.jpg
python predict_tf.py ruta/imagen.jpg --top-k 5 --threshold 0.6
```

Mismo preprocesamiento (299×299 + `preprocess_input`), top-K scores, y
aviso si confianza < umbral (default 0,5). El softmax NO es probabilidad
clínica ni garantiza acierto.

### Ejemplos de inferencia (modelo cap80, `best_model.keras`)

Se probó `predict_tf.py` sobre la primera imagen de 6 clases; **6/6 aciertos**:

| Imagen | Predicción | Confianza | 2ª clase |
|---|---|---|---|
| `paper/paper1.jpg` | paper | 1,000 | plastic (0,000) |
| `metal/metal1.jpg` | metal | 0,999 | plastic (0,001) |
| `white-glass/white-glass1.jpg` | white-glass | 0,992 | trash (0,006) |
| `green-glass/green-glass1.jpg` | green-glass | 1,000 | white-glass (0,000) |
| `brown-glass/brown-glass1.jpg` | brown-glass | 0,960 | green-glass (0,037) |
| `clothes/clothes1.jpg` | clothes | 1,000 | shoes (0,000) |

`brown-glass` es el caso más ajustado (2ª opción `green-glass`), coherente con
la confusión de vidrios observada en la matriz de confusión.

## 10. Pendiente (para documentar al terminar)

- [x] `evaluate`: ejecutado (baseline y modelo cap80). Mejor test F1: 0,9481.
- [x] Experimentos de §11 completados y conclusión anotada en §12.
- [x] Matriz de confusión generada (`confusion_matrix.png`); clases más
      confundidas: vidrios (`brown-glass` ↔ `green-glass`).
- [x] Los 3 duplicados multiclase revisados (§3).
- [x] `predict_tf.py` probado con 6 imágenes reales, 6/6 aciertos (§9).
- [x] `matplotlib` instalado; PNG generado.

## 11. Batería de experimentos (mejora de resultados)

**Objetivo:** además de las etapas 1 y 2 ya entrenadas, comparar varias
configuraciones para (a) subir el F1 macro y (b) poder extraer conclusiones
defendibles en el informe.

**Metodología (igual en todos los experimentos):**
- Se reutiliza SIEMPRE el mismo split (`results_tf/splits.json`) → sin fuga.
- Cada experimento parte de la etapa 1 ya entrenada (`best_stage1.keras`),
  salvo los que precisamente varían la etapa 1.
- La comparación entre variantes se hace con **F1 macro y accuracy en
  VALIDACIÓN** (nunca en test durante la búsqueda).
- El mejor modelo se evalúa **una sola vez en TEST** al final.

**Ejes y scripts:**

| Eje | Script | Variantes | Estado |
|---|---|---|---|
| Fine-tuning (etapa 2) | `experimentos_etapa2.py` | capas 20/40/60/80 · LR 1e-5/5e-6/1e-6 · dropout 0,2/0,3/0,5 (8 variantes, 5 épocas) | *completado (7/8; H dio OOM)* |
| Épocas de etapa 1 | `experimentos_etapa1.py` | 10 (actual) / 15 / 20 épocas + etapa 2 | *completado* |
| Comparación en test | `comparar_test.py` | evalúa N modelos `.keras` en test | disponible |

**Resultados de validación:**

_Fine-tuning (`experimentos_etapa2.py`) — criba de 5 épocas en validación_
_(referencia: baseline cap40/15 ép. → val F1 0,9294):_

| Variante | Capas | LR | Dropout | val acc | val F1 |
|---|---|---|---|---|---|
| **exp_f_cap80_lr1e-5** | **80** | **1e-5** | 0,3 | **0,9582** | **0,9386** |
| exp_c_cap60_lr1e-5 | 60 | 1e-5 | 0,3 | 0,9487 | 0,9229 |
| exp_e_cap40_lr1e-6 | 40 | 1e-6 | 0,3 | 0,9397 | 0,9058 |
| exp_b_cap40_lr1e-5 | 40 | 1e-5 | 0,3 | 0,9380 | 0,9022 |
| exp_g_cap40_lr1e-5_do0.2 | 40 | 1e-5 | 0,2 | 0,9371 | 0,8995 |
| exp_d_cap40_lr5e-6 | 40 | 5e-6 | 0,3 | 0,9354 | 0,8972 |
| exp_a_cap20_lr1e-5 | 20 | 1e-5 | 0,3 | 0,9285 | 0,8857 |
| exp_h_cap40_lr1e-5_do0.5 | 40 | 1e-5 | 0,5 | — | ERROR (OOM) |

**Hallazgo del grid:** lo que más mejora es descongelar **más capas (80)**;
con solo 5 épocas ya supera al baseline de 15 épocas (0,9386 > 0,9294). La
variante `do0.5` falló por **falta de memoria** (GPU de 4 GB): al correr 8
variantes en un mismo proceso la memoria se acumula y las últimas pueden
agotarla (solución: `--batch-size 8` o reiniciar el proceso entre variantes).

_Épocas de etapa 1 (`experimentos_etapa1.py`):_

| Variante | Épocas etapa 1 | val F1 etapa 1 | val F1 etapa 2 |
|---|---|---|---|
| base_ep10 (ya existente) | 10 | 0,8485 | 0,9294 |
| exp_e1_ep15 | 15 | 0,8473 | **0,9319** |
| exp_e1_ep20 | 20 | 0,8486 | 0,9299 |

**Hallazgo:** más épocas de etapa 1 **no mejoran de forma significativa**
(0,9294 / 0,9319 / 0,9299; dentro del ruido). La etapa 1 se estanca en
F1 ≈ 0,848 con 10, 15 y 20 épocas (el early stopping actuó pronto). El lever
que sí importa es el número de **capas descongeladas** en la etapa 2.

**Confirmación de la ganadora a 15 épocas** (`exp_f_cap80_lr1e-5`,
`--fine-tune-layers 80`):

| Métrica | validación | test |
|---|---|---|
| accuracy | 0,9677 | 0,9641 |
| F1 macro | 0,9532 | **0,9481** |

Confirma la hipótesis: 80 capas superan claramente a 40 (test F1 0,9481 vs
0,9287). Es el **mejor modelo** del proyecto.

**Cómo ejecutarlos (Ubuntu/WSL2, entorno `tf-gpu312`, con GPU):**
```bash
cd "/mnt/c/Users/dimas/OneDrive/Documentos/26-2/IA 2/Entrenamiento"
source ~/venvs/tf-gpu312/bin/activate
python experimentos_etapa2.py    # grid de fine-tuning
python experimentos_etapa1.py    # épocas de etapa 1
```

> GPU de 4 GB: ante un error de memoria (OOM), usar `--batch-size 8`.

**Notas de ejecución y reproducibilidad:**
- `train2 --results-dir <carpeta_nueva>` **regenera el split** si esa carpeta no
  tiene `splits.json`; es determinista (misma seed) y se comprobó que coincide
  con `results_tf/splits.json` (mismo conjunto de test).
- El primer intento de confirmar `exp_f` a 15 épocas usó **40 capas**, no 80,
  porque `train_tf.py` aún no tenía la opción `--fine-tune-layers` (ya añadida,
  junto con `--dropout`). Ese intento dio val F1 0,9281 (≈ baseline). Comando
  correcto para cap80:
  `python train_tf.py train2 --results-dir results_tf/exp_f_cap80_lr1e-5 --fine-tune-layers 80 --epochs2 15 --batch-size 16 --weights results_tf/best_stage1.keras`
- El mismo modelo evaluado en test en corridas distintas varía ≈ ±0,001 en acc
  (no determinismo de la GPU): `best_stage2.keras` dio 0,9505 / 0,9509 / 0,9513
  según la corrida.
- Al evaluar con `--model`, `metricas_test.json` guarda las métricas de
  **validación** de `results_tf/` (baseline) junto al **test** del modelo
  evaluado; por eso ahí `val_etapa2` (0,9294) NO corresponde al cap80. Las
  métricas correctas del cap80 están en
  `results_tf/exp_f_cap80_lr1e-5/val_etapa2.json` (val F1 0,9532).

## 12. Conclusión (para redacción)

> Sección pensada como **fuente para redactar el informe**: hechos y cifras
> verificables. Los experimentos de §11 ya terminaron.

**Punto de partida (baseline, ya verificado):**

| Modelo | val acc | val F1 macro | test acc | test F1 macro |
|---|---|---|---|---|
| `best_stage2.keras` (10 ép. etapa 1 + 15 ép. etapa 2) | 0,9539 | 0,9294 | **0,9509** | **0,9277** |

**Hallazgos hasta ahora:**
1. El fine-tuning (etapa 2) mejora claramente a la etapa 1
   (F1 val 0,8485 → 0,9294).
2. El modelo generaliza bien: test (0,9277) queda muy cerca de val (0,9294).
3. El punto débil son los **vidrios**: `brown-glass` (F1 0,82; se confunde con
   `green-glass`), `green-glass` (0,86) y `white-glass` (0,90).
4. **Descongelar más capas ayuda** (cap80 → val F1 0,9386 > cap40 → 0,9294).
5. **Más épocas de etapa 1 no ayudan** (10/15/20 → 0,9294/0,9319/0,9299).
6. Se corrigieron 3 bugs que impedían evaluar en WSL2: rutas con `\` en
   `splits.json`, nombre de archivo en `evaluate` y docstrings con `\U`.
7. Los 3 duplicados que cruzan clases implican siempre a `white-glass` (con
   `metal` y `plastic`): posible etiquetado cruzado (§3).
8. La inferencia con 6 imágenes reales acertó 6/6 (§9).

**Conclusión final:**
- **Mejor variante de fine-tuning:** descongelar **80 capas**
  (`exp_f_cap80_lr1e-5`). Confirmado a 15 épocas: val F1 0,9532 y
  **test F1 0,9481** (vs 0,9287 del baseline cap40). Es el mejor modelo.
- **¿Más épocas de etapa 1?:** no aportan mejora significativa (0,9294 / 0,9319
  / 0,9299 con 10/15/20 épocas).
- **Efecto principal:** descongelar más capas mejora sobre todo las clases
  difíciles (vidrios): `brown-glass` 0,82 → 0,90, `white-glass` 0,90 → 0,92.
- **Mejor modelo global y su F1 en test:** `exp_f_cap80_lr1e-5` → **0,9481**.

## 13. Modelos entrenados (descarga desde Releases)

Los pesos `.keras` **no están versionados en este repositorio**: superan el
límite de 100 MB por archivo de GitHub (el mayor pesa 181 MB) y en total suman
~1,8 GB, por encima de la cuota gratuita de Git LFS. Se publican como *assets*
del Release:

- **[Modelos entrenados v1.0.0](https://github.com/dimasII/Entrenamiento-con-InceptionV3/releases/tag/v1.0.0)** (14 archivos, ~1,8 GB)

En el Release, cada `/` de la ruta original aparece como `_`:

| Archivo en el proyecto | Asset del Release |
|---|---|
| `results_tf/best_model.keras` | `results_tf_best_model.keras` |
| `results_tf/best_stage1.keras` | `results_tf_best_stage1.keras` |
| `results_tf/exp_f_cap80_lr1e-5/best_stage2.keras` | `results_tf_exp_f_cap80_lr1e-5_best_stage2.keras` |
| `results_tf/exp_c_cap60_lr1e-5/best_stage2.keras` | `results_tf_exp_c_cap60_lr1e-5_best_stage2.keras` |
| … (mismo patrón para los `exp_*`) | … |

Tras descargarlos, se colocan en su ruta original dentro de `results_tf/`.
`results_tf/best_model.keras` es el **modelo oficial** (copia de
`exp_f_cap80_lr1e-5/best_stage2.keras`) y es el que usa `predict_tf.py` por
defecto:

```bash
python predict_tf.py ruta/a/imagen.jpg --model results_tf/best_model.keras
```

> **Nota:** el resto del proyecto (código, dataset `garbage_classification/` y
> resultados en `results_tf/`, incluidos `splits.json`, métricas y gráficas)
> **sí** está en el repositorio. Solo se excluyen `__pycache__/`, `*.pyc` y los
> `*.keras` (ver `.gitignore`).

