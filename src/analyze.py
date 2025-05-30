# ================= data_quality_check_silver.py =================
"""
Analiza *training.silver.parquet* directamente desde MinIO (vía S3A) y genera:
• CSV con # valores únicos + ejemplos.
• Exploratory Data Analysis (EDA) automático:
   – Hist/Box para numéricas.
   – Count‑plot para categóricas.
   – Pairplot (scatter‑matrix) para numéricas principales.
Los gráficos se guardan en la carpeta ./eda_report/, y el CSV en HOST_DIR.
"""

from dotenv import load_dotenv
import os
import json
import pathlib
# boto3, shutil, tempfile no longer needed for direct S3A read
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession # Import SparkSession
from pyspark.sql.functions import col, countDistinct
# from utils_spark import iniciar_spark # We'll use a custom S3A Spark session

# ────────────────────────────
# Configuración & credenciales
# ────────────────────────────
load_dotenv()
# PARQUET_DIR_NAME refers to the "directory" in MinIO (e.g., training.silver.parquet)
PARQUET_DIR_NAME = "training.silver.parquet"
BUCKET_NAME      = "datasets"
MINIO_SPARK_ENDPOINT = "http://localhost:9100" # S3A endpoint for Spark (e.g., http://localhost:9100)
ACCESS_KEY       = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY       = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- directorios de salida locales ---------------------------------
# HOST_DIR is where the CSV will be saved.
HOST_DIR = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/data" # lado host
os.makedirs(HOST_DIR, exist_ok=True) # Ensure HOST_DIR exists for the CSV

OUT_CSV_FILENAME = "valores_unicos_silver.csv"
OUT_CSV_PATH = os.path.join(HOST_DIR, OUT_CSV_FILENAME) # Full path for CSV output

EDA_DIR = pathlib.Path("eda_report") # Relative to script execution directory
EDA_DIR.mkdir(exist_ok=True)

if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID / SECRET_KEY en .env")

print("🔐 AWS_ACCESS_KEY_ID:", ACCESS_KEY)
print("🔐 AWS_SECRET_ACCESS_KEY:", "********" if SECRET_KEY else None)


def create_spark_session_s3a(app_name="DataQualityCheckS3A"):
    """Crea una sesión de Spark configurada para MinIO S3A"""
    hadoop_aws_version = "3.3.4"  # Ensure this matches your Spark/Hadoop setup
    aws_sdk_version = "1.12.367" # Ensure this matches

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
            # .master("local[*]") # Uncomment if not using spark-submit with a master defined
            .getOrCreate()
    )

spark = None # Initialize for finally block

try:
    # 1️⃣ Spark con configuración S3A
    spark = create_spark_session_s3a()
    spark.sparkContext.setLogLevel("ERROR")

    # 2️⃣ Leer Parquet directamente desde MinIO
    s3a_parquet_path = f"s3a://{BUCKET_NAME}/{PARQUET_DIR_NAME}"
    print(f"📖 Leyendo Silver Parquet directamente desde MinIO: {s3a_parquet_path}")
    df = spark.read.parquet(s3a_parquet_path)
    print(f"✅ Datos leídos. Registros: {df.count():,}, Columnas: {len(df.columns)}")
    df.printSchema()

    # 3️⃣ CSV de valores únicos
    # The output path for CSV is local, which is fine.
    # Spark's .csv writer will create a directory named OUT_CSV_PATH,
    # and inside it will be the part-csv files.
    # If you want a single CSV file, you'd need an extra step after Spark writes it.
    
    # Let's define the directory where Spark will write the CSV parts
    # Spark will create this directory.
    spark_csv_output_dir = os.path.join(HOST_DIR, "valores_unicos_silver_spark_output")

    print(f"🔍 Generando resumen de unicidad en {spark_csv_output_dir} (directorio Spark CSV)...")
    summary_data = []
    for c_name in df.columns:
        n_uni = df.select(countDistinct(col(c_name))).collect()[0][0]
        # Limit examples for performance and reasonable JSON size
        ejemplos = df.select(c_name).distinct().limit(20).rdd.flatMap(lambda x: x).collect()
        summary_data.append((c_name, n_uni, json.dumps(ejemplos, default=str)))

    summary_df = spark.createDataFrame(summary_data, ["Columna", "ValoresUnicos", "EjemplosUnicos"])

    (summary_df
        .coalesce(1) # To get a single part-file inside the directory
        .write
        .option("header", "true")
        .option("sep", ";")
        .mode("overwrite")
        .csv(spark_csv_output_dir) # Spark writes to this directory
    )
    print(f"✅ Resumen CSV (partes) escrito por Spark en el directorio: {spark_csv_output_dir}")

    # --- Optional: Combine Spark CSV output parts into a single file ---
    # Find the part-xxxxx.csv file and rename/move it
    # This assumes coalesce(1) resulted in one part file
    part_file_found = False
    for item in os.listdir(spark_csv_output_dir):
        if item.startswith("part-") and item.endswith(".csv"):
            source_part_file = os.path.join(spark_csv_output_dir, item)
            print(f"   Encontrado archivo CSV de Spark: {source_part_file}")
            # Move and rename to the desired single CSV file path
            os.rename(source_part_file, OUT_CSV_PATH)
            print(f"✅ Resumen CSV final guardado como archivo único: {OUT_CSV_PATH}")
            part_file_found = True
            # Clean up the (now empty or mostly empty) Spark output directory
            try:
                # Remove other files like _SUCCESS
                for remaining_item in os.listdir(spark_csv_output_dir):
                    os.remove(os.path.join(spark_csv_output_dir, remaining_item))
                os.rmdir(spark_csv_output_dir)
                print(f"   Directorio temporal de Spark CSV '{spark_csv_output_dir}' eliminado.")
            except OSError as e:
                print(f"   Advertencia: No se pudo limpiar completamente el directorio '{spark_csv_output_dir}': {e}")
            break
    if not part_file_found:
        print(f"⚠️ Advertencia: No se encontró el archivo part-*.csv en {spark_csv_output_dir}. "
              f"El CSV único en {OUT_CSV_PATH} no se creó.")
    # --- End Optional single CSV step ---


    # 4️⃣ Convertir a Pandas para EDA (consider sampling for very large data)
    # df.count() was already called, so df is materialized.
    # If df is very large, toPandas() can cause OOM on the driver.
    # Consider df.sample(fraction=0.x).toPandas() if memory is an issue.
    print(f"🐼 Convirtiendo {df.count():,} filas a Pandas para EDA (esto puede tardar y consumir memoria)...")
    pdf = df.toPandas() # No need to sample again if previous count was on full df
    print("✅ Conversión a Pandas completada.")


    # Identificar dtypes
    num_cols  = pdf.select_dtypes(include=["number"]).columns.tolist()
    cat_cols  = [c for c in pdf.columns if c not in num_cols]

    sns.set(style="whitegrid", palette="pastel")
    print(f"📊 Generando gráficos EDA en ./{EDA_DIR.name}/")

    # 4.a Hist & box para numéricas
    for colname in num_cols:
        plt.figure() # Create a new figure for each plot
        fig, ax = plt.subplots(1,2,figsize=(12,5)) # Adjusted size
        sns.histplot(pdf[colname].dropna(), kde=True, ax=ax[0], bins=30); ax[0].set_title(f"Histograma: {colname}")
        sns.boxplot(x=pdf[colname], ax=ax[1]); ax[1].set_title(f"Box Plot: {colname}")
        plt.tight_layout(); plt.savefig(EDA_DIR / f"{colname}_dist.png"); plt.close(fig)
        print(f"   ✓ {colname}_dist.png")


    # 4.b Count‑plot categóricas (máx N categorías para legibilidad)
    MAX_UNIQUE_FOR_COUNTPLOT = 30 # Adjustable
    for colname in cat_cols:
        unique_count = pdf[colname].nunique()
        if unique_count > 0 and unique_count <= MAX_UNIQUE_FOR_COUNTPLOT:
            plt.figure(figsize=(max(8, unique_count * 0.5), 5)) # Dynamic width
            sns.countplot(y=pdf[colname], order = pdf[colname].value_counts().index) # Horizontal for many categories
            plt.title(f"Conteo de Categorías: {colname} ({unique_count} únicas)")
            plt.tight_layout();
            plt.savefig(EDA_DIR / f"{colname}_count.png"); plt.close()
            print(f"   ✓ {colname}_count.png")
        elif unique_count == 0:
            print(f"   ∅ {colname} no tiene valores (o todos son NaN) - saltando countplot.")
        else:
            print(f"   → {colname} tiene {unique_count} valores únicos (> {MAX_UNIQUE_FOR_COUNTPLOT}) - saltando countplot detallado.")


    # 4.c Pairplot principales (hasta N numéricas)
    MAX_COLS_FOR_PAIRPLOT = 11 # Adjustable
    # Select a subset of numeric columns, perhaps by variance or importance if known
    # For now, just takes the first N
    relevant_num_cols = [col for col in num_cols if pdf[col].nunique() > 1] # Avoid constant columns
    top_nums_for_pairplot = relevant_num_cols[:MAX_COLS_FOR_PAIRPLOT]

    if len(top_nums_for_pairplot) >= 2:
        print(f"   Pairplot para: {top_nums_for_pairplot}...")
        pair_plot = sns.pairplot(pdf[top_nums_for_pairplot].dropna(), diag_kind='kde', corner=True)
        pair_plot.fig.suptitle("Pair Plot de Columnas Numéricas Seleccionadas", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98]); # Adjust layout to make space for suptitle
        plt.savefig(EDA_DIR / "pairplot_numeric.png"); plt.close()
        print(f"   ✓ pairplot_numeric.png")
    else:
        print(f"   → No hay suficientes columnas numéricas (>1) para generar pairplot.")


    print(f"✅ Gráficos EDA guardados en ./{EDA_DIR.name}/")

except Exception as e:
    import traceback
    print(f"❌ Error en Data Quality Check / EDA: {e}")
    traceback.print_exc()

finally:
    if spark:
        try:
            spark.stop()
            print("\n🛑 Sesión Spark detenida.")
        except Exception as e_spark:
            print(f"⚠️ Error deteniendo Spark: {e_spark}")
    # No local download to clean up