from dotenv import load_dotenv
import os
# import boto3 # boto3 might still be useful for pre-checks like bucket existence, but not for data writing via Spark
import glob
import pandas as pd
from pyspark.sql import SparkSession
import socket
# import shutil # No longer needed for local parquet cleanup if writing directly to S3A

# Cargar variables del entorno
load_dotenv()

HOST_CSV_DIR = '/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/csv'
CONTAINER_CSV_DIR_FOR_SPARK = '/data/csv'

BUCKET_NAME = 'datasets' # MinIO Bucket Name

# S3A Configuration (ensure these are correct for your MinIO setup)
MINIO_SPARK_ENDPOINT = "http://172.17.0.1:9100" # S3A endpoint for Spark
ACCESS_KEY       = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY       = os.getenv("AWS_SECRET_ACCESS_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID / SECRET_KEY en .env")

# No need to create local PARQUET_DIR if writing directly to S3A
# os.makedirs(PARQUET_DIR, exist_ok=True)

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
        print(f"❌ Error probando conexión a Spark Master '{spark_master_url}': {e}")
        return False

# <<<--- CONFIGURACIÓN CRÍTICA ---<<<
driver_announce_ip = "172.17.0.1" # La IP de docker0 que los contenedores SÍ PUEDEN VER
print(f"📢 Anunciando IP del driver para los ejecutores: {driver_announce_ip}")

spark = None
try:
    spark_master_options = [
    "spark://172.17.0.1:7077",     # Si ejecutas desde host y está mapeado
    "spark://localhost:7077",     # Si ejecutas desde host y está mapeado
    "spark://spark-master:7077",  # Si el script se ejecuta DENTRO de un contenedor en la misma red Docker
    # "spark://<IP_DEL_HOST>:7077", # Otra opción si localhost no funciona desde el script
    "local[*]"                    # Modo local como último recurso
    ]

    spark_master_url = None
    for master_url_option in spark_master_options:
        print(f"🔍 Probando conexión a Spark Master: {master_url_option}")
        if master_url_option == "local[*]":
            print("⚠️ Usando modo local de Spark (sin cluster).")
            spark_master_url = master_url_option
            break
        elif test_spark_connection(master_url_option):
            print(f"✅ Conexión exitosa a Spark Master: {master_url_option}")
            spark_master_url = master_url_option
            break
        else:
            print(f"❌ No se pudo conectar a Spark Master: {master_url_option}")
    
    if not spark_master_url:
        raise RuntimeError("❌ No se pudo conectar a ningún Spark Master de las opciones probadas.")
    
    print(f"🚀 Inicializando Spark con Master: {spark_master_url}")

    # ... (cerca de donde defines hadoop_aws_version y aws_sdk_version) ...
    hadoop_aws_version = "3.3.4"
    aws_sdk_version = "1.12.367"
    
    spark_builder = SparkSession.builder \
        .appName("CSVtoParquetDirectToMinIO") \
        .master(spark_master_url) \
        .config("spark.driver.host", driver_announce_ip) \
            .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY) \
            .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY)\
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_SPARK_ENDPOINT)\
            .config("spark.hadoop.fs.s3a.path.style.access", "true")\
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")\
            .config("spark.jars.packages",
                    f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version},"\
                    f"com.amazonaws:aws-java-sdk-bundle:{aws_sdk_version}")\
    
    spark = spark_builder.getOrCreate()

    print("🧪 Probando funcionalidad de Spark...")
    test_df = spark.range(1, 5)
    test_count = test_df.count()
    print(f"✅ Spark funcionando correctamente. Test count: {test_count}")

    if not os.path.exists(HOST_CSV_DIR):
        raise FileNotFoundError(f"❌ Directorio CSV no existe: {HOST_CSV_DIR}")
    
    csv_files = [f for f in os.listdir(HOST_CSV_DIR) if f.endswith('.csv')]

    if not csv_files:
        print(f"⚠️ No se encontraron archivos CSV en: {HOST_CSV_DIR}")
        exit(0)
    
    print(f"📁 Encontrados {len(csv_files)} archivos CSV: {csv_files}")

    # (Optional) Pre-check MinIO bucket existence with boto3 if desired
    # import boto3
    # try:
    #     s3_check_client = boto3.client(
    #         's3',
    #         endpoint_url=MINIO_SPARK_ENDPOINT, # Use the same endpoint
    #         aws_access_key_id=ACCESS_KEY,
    #         aws_secret_access_key=SECRET_KEY
    #     )
    #     s3_check_client.head_bucket(Bucket=BUCKET_NAME)
    #     print(f"✅ Bucket MinIO '{BUCKET_NAME}' verificado.")
    # except Exception as e_boto:
    #     print(f"❌ Error verificando bucket MinIO '{BUCKET_NAME}': {e_boto}")
    #     print("💡 Asegúrate de que MinIO esté ejecutándose y el bucket exista antes de que Spark intente escribir.")
    #     # raise # You might want to raise an error here if bucket check is critical

    for filename in csv_files:
        try:
            spark_worker_csv_path = os.path.join(CONTAINER_CSV_DIR_FOR_SPARK, filename)
            # Define the S3A path for the output Parquet "directory"
            # Spark will create a directory named parquet_filename in the bucket
            parquet_filename_on_s3 = filename.replace('.csv', '.parquet')
            s3a_output_path = f"s3a://{BUCKET_NAME}/{parquet_filename_on_s3}"

            print(f"\n📦 Procesando: {filename}")
            print(f"   📍 Origen (local CSV): {spark_worker_csv_path }")
            print(f"   📍 Destino (MinIO S3A): {s3a_output_path}")

            print("   🔍 Validando CSV con Pandas...")
            # Dejar que Spark lea el CSV directamente
            print("   🔄 Leyendo CSV directamente con Spark...")
            df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv(spark_worker_csv_path) # Spark lee desde la ruta del worker
            
            if df.rdd.isEmpty():
                print(f"   ⚠️ DataFrame de Spark vacío después de leer, saltando: {filename}")
                continue

            # Esta cuenta ahora sí se ejecutará distribuida
            print(f"   📊 Filas leídas por Spark: {df.count()}, Columnas: {len(df.columns)}")
            
            # ... justo antes de escribir ...
            count_before_write = df.count()
            print(f"   📏 Verificación ANTES de escribir: DataFrame tiene {count_before_write} filas.")
            if count_before_write == 0:
                print(f"   🛑 ALERTA: DataFrame para {filename} está vacío. No se escribirán datos Parquet.")
                # Decide si quieres continuar y crear un directorio vacío con _SUCCESS o detenerte/manejarlo.
                # Por ahora, para debugging, podrías dejar que continúe y cree el _SUCCESS
                # o podrías hacer un 'continue' para saltar la escritura.

            (df.write
            .mode("overwrite")
            .parquet(s3a_output_path)
)
            # The output at s3a_output_path is a directory containing _SUCCESS and part-*.parquet files.
            # If you need a single Parquet file named exactly 'file.parquet' directly in the bucket
            # (not a directory), you'd need the post-processing step with MinIO client as in claude-prepare_from_minio.py
            # For now, this writes a directory, which is standard Spark behavior.

            print(f"   ✅ {filename} procesado y escrito a MinIO como Parquet en el directorio: {parquet_filename_on_s3}")

        except Exception as file_error:
            import traceback
            print(f"   🚨 Error procesando {filename}: {file_error}")
            traceback.print_exc() # Print full traceback for file errors
            continue

    print("\n🎉 Conversión y escritura directa a MinIO completadas con Spark.")

except Exception as e:
    import traceback
    print(f"🚨 Error general durante la ejecución: {e}")
    traceback.print_exc() # Print full traceback for general errors
    print("\n🔧 Sugerencias de debugging:")
    print("1. Verifica que los contenedores de Spark (master/worker) estén ejecutándose.")
    print("2. Verifica que MinIO esté ejecutándose y accesible en el endpoint configurado.")
    print(f"3. Verifica que el bucket '{BUCKET_NAME}' exista en MinIO.")
    print(f"4. Verifica que el directorio CSV '{HOST_CSV_DIR}' existe y tiene archivos.")
    print("5. Revisa las credenciales de MinIO en tu archivo .env.")
    print("6. Asegúrate de que las versiones de hadoop-aws y aws-java-sdk-bundle sean compatibles con tu Spark.")


finally:
    if spark:
        try:
            spark.stop()
            print("🛑 Spark detenido correctamente.")
        except Exception as stop_error:
            print(f"⚠️ Error deteniendo Spark: {stop_error}")