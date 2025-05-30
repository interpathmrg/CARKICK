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

CSV_DIR = '/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/csv'
# PARQUET_DIR = '/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/parquet' # No longer needed for primary output
BUCKET_NAME = 'datasets' # MinIO Bucket Name

# S3A Configuration (ensure these are correct for your MinIO setup)
MINIO_SPARK_ENDPOINT = "http://localhost:9100" # S3A endpoint for Spark
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

spark = None
try:
    spark_master_options = [
        "local[*]",
        "spark://localhost:7077",
        "spark://spark-master:7077",
        # "spark://172.20.0.3:7077" # You can add specific IPs if needed
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

    # Configure Spark Session for S3A
    hadoop_aws_version = "3.3.4"  # Ensure this matches your Spark/Hadoop setup
    aws_sdk_version = "1.12.367" # Ensure this matches

    spark_builder = SparkSession.builder \
        .appName("CSVtoParquetDirectToMinIO") \
        .master(spark_master_url) \
        .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_SPARK_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.jars.packages",
                f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version},"
                f"com.amazonaws:aws-java-sdk-bundle:{aws_sdk_version}") \
        .config("spark.executor.instances", "1") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.memory", "1g") \
        .config("spark.cores.max", "4") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    spark = spark_builder.getOrCreate()

    print("🧪 Probando funcionalidad de Spark...")
    test_df = spark.range(1, 5)
    test_count = test_df.count()
    print(f"✅ Spark funcionando correctamente. Test count: {test_count}")

    if not os.path.exists(CSV_DIR):
        raise FileNotFoundError(f"❌ Directorio CSV no existe: {CSV_DIR}")
    
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"⚠️ No se encontraron archivos CSV en: {CSV_DIR}")
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
            csv_path = os.path.join(CSV_DIR, filename)
            # Define the S3A path for the output Parquet "directory"
            # Spark will create a directory named parquet_filename in the bucket
            parquet_filename_on_s3 = filename.replace('.csv', '.parquet')
            s3a_output_path = f"s3a://{BUCKET_NAME}/{parquet_filename_on_s3}"

            print(f"\n📦 Procesando: {filename}")
            print(f"   📍 Origen (local CSV): {csv_path}")
            print(f"   📍 Destino (MinIO S3A): {s3a_output_path}")

            print("   🔍 Validando CSV con Pandas...")
            pdf = pd.read_csv(csv_path) # Consider using spark.read.csv for very large CSVs
            print(f"   📊 Filas: {len(pdf)}, Columnas: {len(pdf.columns)}")
            
            if pdf.empty:
                print(f"   ⚠️ Archivo CSV vacío, saltando: {filename}")
                continue

            print("   🔄 Convirtiendo a Spark DataFrame...")
            # For very large CSVs, it's more scalable to let Spark read the CSV directly:
            # df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
            # However, createDataFrame from pandas is fine for moderately sized CSVs
            df = spark.createDataFrame(pdf)
            
            # No need to manually clean up S3A destination; Spark's "overwrite" mode handles it.
            # Spark's "overwrite" mode for S3A will delete the target directory if it exists.

            print(f"   💾 Escribiendo Parquet directamente a MinIO en {s3a_output_path}...")
            (df.coalesce(1) # To write a single part-file within the S3A "directory"
               .write
               .mode("overwrite") # Overwrites the S3A "directory" if it exists
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
    print(f"4. Verifica que el directorio CSV '{CSV_DIR}' existe y tiene archivos.")
    print("5. Revisa las credenciales de MinIO en tu archivo .env.")
    print("6. Asegúrate de que las versiones de hadoop-aws y aws-java-sdk-bundle sean compatibles con tu Spark.")


finally:
    if spark:
        try:
            spark.stop()
            print("🛑 Spark detenido correctamente.")
        except Exception as stop_error:
            print(f"⚠️ Error deteniendo Spark: {stop_error}")