## ================= preview_from_minio.py (capa Silver, vía boto3) =================
"""
Lee *training.silver.parquet* desde MinIO **sin usar S3A**:
1. Descarga el archivo con boto3 a /tmp.
2. Lo carga en Spark desde disco local.
3. Genera box‑plots y count‑plot.
Esto evita los conflictos de versiones Hadoop‑AWS.
"""

from dotenv import load_dotenv
import os
import boto3
import tempfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils_spark import (
    export_schema_info,
    export_describe,
    export_sample,
    iniciar_spark,
)

# ────────────────────────────
# Configuración & credenciales
# ────────────────────────────
load_dotenv()
PARQUET_FILENAME = "training.silver.parquet"
BUCKET_NAME = "datasets"
MINIO_URL = "http://localhost:9100"
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY en .env")

# Ruta temporal local
local_dir = tempfile.mkdtemp(prefix="silver_")
local_path = os.path.join(local_dir, PARQUET_FILENAME)

try:
    # ───── 1. Descargar de MinIO ─────
    print("☁️ Descargando Silver desde MinIO → local …")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    s3.download_file(BUCKET_NAME, PARQUET_FILENAME, local_path)
    print(f"✅ Descargado en {local_path}")

    # ───── 2. Iniciar Spark y leer Parquet local ─────
    spark, _ = iniciar_spark(app_name="silver_preview_local")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(local_path)

    # Exports CSV (profiling)
    export_schema_info(df)
    export_describe(df)
    export_sample(df)

    # ───── 3. Visualizaciones ─────
    numeric_cols = ["VehicleAge", "VehOdo", "VehBCost"]
    idx_cols = ["Transmission_idx", "Size_idx"]

    pandas_df = (
        df.select(["IsBadBuy"] + numeric_cols + idx_cols)
        .sample(fraction=0.1, seed=42)
        .toPandas()
    )

    sns.set(style="whitegrid", palette="pastel")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, col in enumerate(numeric_cols):
        ax = axes[i // 2, i % 2]
        sns.boxplot(data=pandas_df, x="IsBadBuy", y=col, ax=ax)
        ax.set_title(f"{col} vs IsBadBuy")

    sns.countplot(data=pandas_df, x="Transmission_idx", hue="IsBadBuy", ax=axes[1, 1])
    axes[1, 1].set_title("Transmission_idx por IsBadBuy")

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    try:
        spark.stop()
        print("🛑 Spark detenido.")
    except Exception:
        pass
    shutil.rmtree(local_dir, ignore_errors=True)