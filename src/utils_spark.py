# utils_spark.py
from pyspark.sql.functions import col, trim
from pyspark.sql import SparkSession
import os
from dotenv import load_dotenv

def iniciar_spark(app_name="Spark App", master="local[*]", parquet_filename="data.parquet", bucket_name="datasets", minio_url="http://localhost:9100"):
    load_dotenv()

    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not access_key or not secret_key:
        raise EnvironmentError("❌ Las variables AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY no están definidas.")

    parquet_s3_path = f"s3a://{bucket_name}/{parquet_filename}"

    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0,com.amazonaws:aws-java-sdk-bundle:1.11.1026") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", minio_url) \
        .config("spark.hadoop.fs.s3a.access.key", access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", secret_key) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark, parquet_s3_path

def export_schema_info(df, output_path="esquema_output_semicolon.csv", separator=";"):
    """
    Exporta el esquema de un DataFrame de Spark incluyendo:
    - Nombre de columna
    - Tipo de dato
    - Si permite nulos
    - Cantidad de registros no vacíos por columna
    - Cantidad de registros nulos o vacíos por columna
    """
    print("📊 Analizando esquema del DataFrame...")
    schema_info = []
    for field in df.schema:
        name = field.name
        dtype = str(field.dataType)
        nullable = field.nullable
        non_null_count = df.filter(col(name).isNotNull() & (trim(col(name)) != "")).count()
        null_count = df.filter(col(name).isNull() | (trim(col(name)) == "")).count()
        schema_info.append((name, dtype, nullable, non_null_count, null_count))

    schema_df = df.sparkSession.createDataFrame(
        schema_info, ["Nombre", "Tipo", "Permite Nulos", "Registros No Vacíos", "Registros Nulos o Vacíos"]
    )

    schema_df.coalesce(1).write \
        .option("header", True) \
        .option("sep", separator) \
        .mode("overwrite") \
        .csv(output_path)

    print(f"✅ Esquema exportado a {output_path}")

def export_describe(df, output_path="describe_output_semicolon.csv", separator=";"):
    """Exporta las estadísticas descriptivas del DataFrame (count, mean, stddev, min, max)"""
    print("📊 Generando estadísticas descriptivas...")
    df.describe().coalesce(1).write \
        .option("header", True) \
        .option("sep", separator) \
        .mode("overwrite") \
        .csv(output_path)
    print(f"✅ Describe exportado a {output_path}")

def export_sample(df, output_path="muestra_output_semicolon.csv", sample_size=100, separator=";"):
    """Exporta los primeros registros del DataFrame como muestra"""
    print(f"📊 Exportando muestra de {sample_size} registros...")
    df.limit(sample_size).coalesce(1).write \
        .option("header", True) \
        .option("sep", separator) \
        .mode("overwrite") \
        .csv(output_path)
    print(f"✅ Muestra exportada a {output_path}")
