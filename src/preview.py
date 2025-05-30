
## ================= preview_from_minio.py (capa Silver, vía S3A) =================
"""
Lee el *directorio* training.silver.parquet directamente desde MinIO **usando S3A**:
1. Configura Spark para acceso S3A a MinIO.
2. Carga el Parquet directamente desde MinIO.
3. Genera box-plots y count-plot.
Esto es más eficiente y directo que descargar localmente primero.
"""

from dotenv import load_dotenv
import os
# boto3, tempfile, shutil are no longer needed for this direct approach
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd # Used by toPandas()
from pyspark.sql import SparkSession # Import SparkSession
from utils_spark import (
    export_schema_info,
    export_describe,
    export_sample,
    # iniciar_spark, # We will create a custom Spark session with S3A config
)

# ────────────────────────────
# Configuración & credenciales
# ────────────────────────────
load_dotenv()
PARQUET_DIR_NAME = "training.silver.parquet" # This is the "directory" in MinIO
BUCKET_NAME = "datasets"
# This MINIO_ENDPOINT is for Spark's S3A connector
MINIO_SPARK_ENDPOINT = "http://localhost:9100" # Ensure this matches your MinIO S3 API port (e.g., 9100)
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY en .env")

print("🔐 AWS_ACCESS_KEY_ID:", ACCESS_KEY)
print("🔐 AWS_SECRET_ACCESS_KEY:", "********" if SECRET_KEY else None)


def create_spark_session_s3a(app_name="SilverPreviewS3A"):
    """Crea una sesión de Spark configurada para MinIO S3A"""
    # Choose versions compatible with your Spark/Hadoop setup
    hadoop_aws_version = "3.3.4"  # Example, adjust if needed (same as in claude-prepare_from_minio.py)
    aws_sdk_version = "1.12.367" # Example, adjust if needed

    print(f"🛠️ Configurando Spark para S3A con endpoint: {MINIO_SPARK_ENDPOINT}")
    return (
        SparkSession.builder
            .appName(app_name)
            .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY)
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_SPARK_ENDPOINT)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.jars.packages",
                    f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version},"
                    f"com.amazonaws:aws-java-sdk-bundle:{aws_sdk_version}")
            # .master("local[*]") # Uncomment if running locally without spark-submit defining master
            .getOrCreate()
    )

spark_session = None # Initialize for finally block

try:
    # ───── 1. Iniciar Spark con configuración S3A ─────
    spark_session = create_spark_session_s3a()
    spark_session.sparkContext.setLogLevel("ERROR") # Keep log level setting

    # ───── 2. Leer Parquet directamente desde MinIO ─────
    s3a_path = f"s3a://{BUCKET_NAME}/{PARQUET_DIR_NAME}"
    print(f"📖 Leyendo Parquet directamente desde MinIO: {s3a_path}")
    
    df = spark_session.read.parquet(s3a_path)
    
    print(f"✅ Datos leídos exitosamente desde MinIO. Registros: {df.count():,}")
    print("📊 Esquema del dataset:")
    df.printSchema()

    # Exports CSV (profiling)
    export_schema_info(df)
    export_describe(df)
    export_sample(df)

    # ───── 3. Visualizaciones ─────
    print("📊 Generando visualizaciones...")
    numeric_cols = ["VehicleAge", "VehOdo", "VehBCost"]
    idx_cols_to_check = ["Transmission_idx", "Size_idx"]
    
    available_cols = df.columns
    idx_cols = [col for col in idx_cols_to_check if col in available_cols]
    
    if not idx_cols:
        print(f"⚠️ Advertencia: Ninguna de las columnas {idx_cols_to_check} encontradas para visualización.")

    select_cols_for_pandas = ["IsBadBuy"] + numeric_cols
    if idx_cols:
         select_cols_for_pandas += idx_cols

    if "IsBadBuy" not in available_cols:
        raise ValueError("❌ La columna 'IsBadBuy' no se encuentra en el DataFrame. Verifica el script de preparación Silver.")

    pandas_df = (
        df.select(select_cols_for_pandas)
        .sample(fraction=0.1, seed=42)
        .toPandas()
    )

    sns.set(style="whitegrid", palette="pastel")
    
    num_plots = len(numeric_cols)
    if idx_cols:
        num_plots += 1 # Assuming one plot for Transmission_idx or similar
    
    if num_plots == 0:
        print("🤷 No hay columnas numéricas o categóricas indexadas para graficar.")
    else:
        n_subplot_cols = 2
        n_subplot_rows = (num_plots + n_subplot_cols - 1) // n_subplot_cols
        
        fig, axes = plt.subplots(n_subplot_rows, n_subplot_cols, figsize=(14, 5 * n_subplot_rows))
        axes = axes.flatten()

        plot_idx = 0
        for col in numeric_cols:
            if col not in pandas_df.columns:
                print(f"⚠️ Advertencia: La columna '{col}' no está en el DataFrame de Pandas. Saltando gráfico.")
                continue
            ax = axes[plot_idx]
            sns.boxplot(data=pandas_df, x="IsBadBuy", y=col, ax=ax)
            ax.set_title(f"{col} vs IsBadBuy")
            plot_idx +=1

        if "Transmission_idx" in idx_cols and "Transmission_idx" in pandas_df.columns :
            ax = axes[plot_idx]
            sns.countplot(data=pandas_df, x="Transmission_idx", hue="IsBadBuy", ax=ax)
            ax.set_title("Transmission_idx por IsBadBuy")
            plot_idx +=1
        
        # Add more plots for other idx_cols if needed, similar to Transmission_idx

        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        #plt.show()
        # Save the figure
        plot_filename = "silver_data_preview.png"
        plt.savefig(plot_filename)
        print(f"✅ Gráfico guardado como: {plot_filename}")
        plt.close(fig) # Close the figure object to free memory

except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()

finally:
    if spark_session:
        try:
            spark_session.stop()
            print("🛑 Spark detenido.")
        except Exception as e_spark_stop:
            print(f"⚠️ Error deteniendo Spark: {e_spark_stop}")
    # No local directory to clean up with this direct S3A approach
    print("🧹 Limpieza (si es necesaria) completada.")