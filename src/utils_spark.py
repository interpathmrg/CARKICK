
# ================= utils_spark.py (stable Hadoop‑AWS) =================
"""
⚙️ Utilidades Spark ⇄ MinIO — versión simple & robusta
• **NO** crea un SparkContext antes del builder (evita que los paquetes se pierdan).
• Usa por defecto Hadoop 3.3.6 + aws‑sdk‑bundle 1.12.648 ← funcionan con Spark ≥3.3.
• Si ya existe una sesión Spark, simplemente la devuelve (sin añadir paquetes).
"""

from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os

DEFAULT_HADOOP_AWS = "3.3.6"
DEFAULT_AWS_SDK    = "1.12.648"

# ────────────────────────────
# Inicio / reutilización de Spark
# ────────────────────────────

def iniciar_spark(
    app_name: str = "Spark App",
    master: str = "local[*]",
    parquet_filename: str = "data.parquet",
    bucket_name: str = "datasets",
    minio_url: str = "http://localhost:9100",
    hadoop_aws_version: str = DEFAULT_HADOOP_AWS,
    aws_sdk_version: str = DEFAULT_AWS_SDK,
):
    """Crea (o reutiliza) una sesión Spark configurada para S3A/MinIO."""

    # 1. Si ya hay Spark activo, úsalo sin reconstruir.
    active = SparkSession.getActiveSession()
    if active:
        parquet_s3_path = f"s3a://{bucket_name}/{parquet_filename}"
        return active, parquet_s3_path

    # 2. Cargar credenciales env
    load_dotenv()
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY en .env")

    # 3. Paquetes Hadoop‑AWS
    packages = (
        f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version},"
        f"com.amazonaws:aws-java-sdk-bundle:{aws_sdk_version}"
    )

    # 4. Ruta parquet S3A
    parquet_s3_path = f"s3a://{bucket_name}/{parquet_filename}"

    # 5. Construir SparkSession con los paquetes
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.jars.packages", packages)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", minio_url)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark, parquet_s3_path

# ────────── helpers export (mismos cuerpos que versiones previas) ──────────
from pyspark.sql.functions import col

def export_schema_info(df, output_path="esquema_output_semicolon.csv", separator=";"):
    schema_info = []
    for field in df.schema:
        name = field.name
        dtype = str(field.dataType)
        nullable = field.nullable
        non_null_count = df.filter(col(name).isNotNull()).count()
        null_count = df.filter(col(name).isNull()).count()
        schema_info.append((name, dtype, nullable, non_null_count, null_count))

    schema_df = df.sparkSession.createDataFrame(
        schema_info,
        ["Nombre", "Tipo", "Permite Nulos", "Registros No Vacíos", "Registros Nulos o Vacíos"],
    )
    schema_df.coalesce(1).write.option("header", True).option("sep", separator).mode("overwrite").csv(output_path)

def export_describe(df, output_path="describe_output_semicolon.csv", separator=";"):
    df.describe().coalesce(1).write.option("header", True).option("sep", separator).mode("overwrite").csv(output_path)

def export_sample(df, output_path="muestra_output_semicolon.csv", sample_size=100, separator=";"):
    df.limit(sample_size).coalesce(1).write.option("header", True).option("sep", separator).mode("overwrite").csv(output_path)
