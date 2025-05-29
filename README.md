# Referencias:
# https://www.kaggle.com/code/zaralavii/data-mining-on-car-auction-data
# https://www.kaggle.com/competitions/DontGetKicked/overview
# https://jemsethio.github.io/project/do_not_get_kicked/
#
# Desarrollado por : Miguel R. Gonzalez
# Apoyo programación: ChatGPTo3
# Herramientas: MinIO,Apache Spark, MS VScode, Python, PySpark, Java JDK 
# Auto-Auction Kick Prediction 🚗💥  con modelo RandomForest

Proyecto de **Data Science con Apache Spark, MinIO y Python** que:

1. 📥  Lee los datos “Bronze” (`training.parquet`) desde MinIO  
2. 🧹  Genera la **capa Silver** (`training.silver.parquet`) con limpieza y nuevas variables  
3. 🔍  Ejecuta **Data Quality + EDA** con gráficos automáticos  
4. 🤖  Entrena y evalúa un modelo **Random Forest** para predecir si una compra será un **_kick_** (`IsBadBuy = 1`)

---

## 🗂️ Estructura

.
├─ preview_from_minio.py # Visualización rápida de la capa Silver
├─ utils_spark.py # Helper SparkSession (sin S3A) + exportadores CSV
├─ data_quality_check_silver.py # Resumen de unicidad + EDA con gráficos
├─ kick_prediction_silver.py # Entrenamiento y evaluación del modelo
├─ eda_report/ # PNG generados por el EDA
└─ requirements.txt

---
## ⚙️ Despliegue infraestructura docker
docker-compose.yml

## ⚙️ Requisitos

| Componente | Versión mínima |
|------------|----------------|
| Python     | 3.8           |
| Apache Spark (local) | 3.3+ |
| Java JDK   | 11            |
| MinIO      | en ejecución con bucket `datasets` |
| Dependencias Python | ver `requirements.txt` |

Instala con:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
🔑 Variables de entorno
Crea un archivo .env en la raíz con:

env
Copiar
Editar
AWS_ACCESS_KEY_ID=<tu-access-key>
AWS_SECRET_ACCESS_KEY=<tu-secret-key>
Las credenciales corresponden al usuario MinIO que tenga acceso de lectura/escritura al bucket datasets.

🚀 Uso rápido
1. Vista previa de la capa Silver

python preview_from_minio.py
Muestra box-plots básicos y exporta tres CSV:

esquema_output_semicolon.csv

describe_output_semicolon.csv

muestra_output_semicolon.csv

2. Data Quality + EDA

python data_quality_check_silver.py
Genera valores_unicos_silver.csv

Crea gráficos en ./eda_report/

3. Modelado

python kick_prediction_silver.py

mathematica:
AUC, F1-score, Accuracy
y guarda:
Importancia de variables (PNG)
Matriz de confusión (PNG)
Curva ROC (PNG)

📂 Detalles de las capas
Capa	Archivo	Descripción
Bronze	training.parquet	Datos originales de la subasta
Silver	training.silver.parquet	Limpieza: casting, indices Transmission_idx/Size_idx, selección de columnas

La creación de Silver se hace con el script ETL (no incluido aquí) que encontrarás en /etl/prepare_from_minio.py.

🛠️ Troubleshooting
Mensaje	Causa /Solución
NoSuchMethodError: PrefetchingStatistics	Evitado: usamos boto3, no S3A
AWS_ACCESS_KEY_ID no definido	Verifica tu archivo .env
Spark no arranca	Revisa variables JAVA_HOME, JDK 11

📜 Licencia
MIT © 2025 Miguel González
Suministrado solo con fines educativos/demostrativos.

