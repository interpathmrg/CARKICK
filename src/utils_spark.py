
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
import socket
import shutil


DEFAULT_HADOOP_AWS = "3.3.6"
DEFAULT_AWS_SDK    = "1.12.648"
DRIVER_HOST = "172.17.0.1"   # gateway docker0
DRIVER_PORT = "7078"
BLOCKM_PORT = "7079"

# ────────────────────────────
# Inicio / reutilización de Spark
# ────────────────────────────


def test_spark_connection(spark_master_url):
    """Verifica si el Spark Master está accesible"""
    try:
        host = spark_master_url.replace('spark://', '').split(':')[0]
        port = int(spark_master_url.replace('spark://', '').split(':')[1])
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        return result == 0
    except Exception as e:
        print(f"❌ Error probando conexión: {e}")
        return False
    
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
     
     
     #Opciones de conexión a Spark Master (prioridad de más probable a menos probable)
    spark_master_options = [
        "spark://localhost:7077",     # Si ejecutas desde host
        #"spark://spark-master:7077",  # Usando nombre del contenedor
        #"spark://172.20.0.3:7077"     # Tu IP original
        #"spark://0.0.0.0:7077"     # Tu IP original
        #"local[*]",                   # Modo local primero (para debugging)
    ]
    
    spark_master_url = None
    
    for master_url in spark_master_options:
        print(f"🔍 Probando conexión a: {master_url}")
        
        if master_url == "local[*]":
            print("⚠️ Usando modo local de Spark (sin cluster)")
            spark_master_url = master_url
            break
        elif test_spark_connection(master_url):
            print(f"✅ Conexión exitosa a: {master_url}")
            spark_master_url = master_url
            break
        else:
            print(f"❌ No se pudo conectar a: {master_url}")
    
    if not spark_master_url:
        raise RuntimeError("❌ No se pudo conectar a ningún Spark Master")
    
    print(f"🚀 Inicializando Spark con: {spark_master_url}")
    # Inicializar Spark
    spark = (
    SparkSession.builder
        .appName("carkick")                 # nombre de tu aplicación
        .master(spark_master_url)           # URL del master: "local[4]" o "spark://host:7077"
        # ─ Driver (host) ─
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.host", DRIVER_HOST)
        .config("spark.driver.port", DRIVER_PORT)
        .config("spark.blockManager.port", BLOCKM_PORT)
        # ───── Recursos executor ─────
        .config("spark.executor.instances", "1")
        .config("spark.executor.cores",     "2")
        .config("spark.executor.memory",    "2g")
        .config("spark.cores.max",          "2")   # límite global para la app
        # ───── Otras optimizaciones ─────
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
)
     # Validar que Spark está funcionando
    print("🧪 Probando funcionalidad de Spark...")
    test_df = spark.range(1, 5)
    test_count = test_df.count()
    print(f"✅ Spark funcionando correctamente. Test count: {test_count}")
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
