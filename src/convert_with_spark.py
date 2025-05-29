from dotenv import load_dotenv
import os
import boto3
import glob
import pandas as pd
from pyspark.sql import SparkSession
import socket
import shutil


# Cargar variables del entorno
load_dotenv()

CSV_DIR = '/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/csv'
PARQUET_DIR = '/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/parquet'
BUCKET_NAME = 'datasets'

os.makedirs(PARQUET_DIR, exist_ok=True)

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

# Inicializar Spark
spark = None
try:
    # Opciones de conexión a Spark Master (prioridad de más probable a menos probable)
    spark_master_options = [
        "local[*]",                   # Modo local primero (para debugging)
        "spark://localhost:7077",     # Si ejecutas desde host
        "spark://spark-master:7077",  # Usando nombre del contenedor
        "spark://172.20.0.3:7077"     # Tu IP original
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
    
    spark = SparkSession.builder \
        .appName("convert_with_spark.py") \
        .master(spark_master_url) \
        .config("spark.executor.instances", "1") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.memory", "1g") \
        .config("spark.cores.max", "4") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    # Validar que Spark está funcionando
    print("🧪 Probando funcionalidad de Spark...")
    test_df = spark.range(1, 5)
    test_count = test_df.count()
    print(f"✅ Spark funcionando correctamente. Test count: {test_count}")

    # Verificar si hay archivos CSV
    if not os.path.exists(CSV_DIR):
        raise FileNotFoundError(f"❌ Directorio CSV no existe: {CSV_DIR}")
    
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"⚠️ No se encontraron archivos CSV en: {CSV_DIR}")
        exit(0)
    
    print(f"📁 Encontrados {len(csv_files)} archivos CSV: {csv_files}")

    # Inicializar cliente MinIO/S3
    try:
        print("🔌 Conectando a MinIO...")
        
        # Obtener credenciales del archivo .env
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError("❌ Credenciales AWS no encontradas en .env")
        
        print(f"🔑 Usando credenciales: {aws_access_key}")
        
        s3 = boto3.client(
            's3', 
            endpoint_url='http://localhost:9100',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Verificar conexión a MinIO
        s3.head_bucket(Bucket=BUCKET_NAME)
        print("✅ Conexión a MinIO exitosa")
        
    except Exception as e:
        print(f"❌ Error conectando a MinIO: {e}")
        print("💡 Asegúrate de que MinIO esté ejecutándose y el bucket existe")
        raise

    # Procesar CSVs
    for filename in csv_files:
        try:
            csv_path = os.path.join(CSV_DIR, filename)
            parquet_filename = filename.replace('.csv', '.parquet')
            #output_path = os.path.join(PARQUET_DIR, parquet_filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(PARQUET_DIR, base_name)

            print(f"\n📦 Procesando: {filename}")
            print(f"   📍 Origen: {csv_path}")
            print(f"   📍 Destino: {output_path}")

            # Leer CSV con pandas primero para validar
            print("   🔍 Validating CSV...")
            pdf = pd.read_csv(csv_path)
            print(f"   📊 Filas: {len(pdf)}, Columnas: {len(pdf.columns)}")
            
            if pdf.empty:
                print(f"   ⚠️ Archivo CSV vacío, saltando: {filename}")
                continue

            # Crear DataFrame de Spark
            print("   🔄 Convirtiendo a Spark DataFrame...")
            df = spark.createDataFrame(pdf)

            

            # Eliminar el archivo o directorio de salida si ya existe
            if os.path.exists(output_path):
                print(f"🧹 Limpiando destino anterior: {output_path}")
                try:
                    if os.path.isfile(output_path):
                        os.remove(output_path)
                    else:
                        shutil.rmtree(output_path)
                except Exception as e:
                    print(f"⚠️ Error eliminando destino previo: {e}")
                    raise

            # Escribir Parquet
            print(f"   💾 Escribiendo Parquet...")
            df.coalesce(1).write.mode("overwrite").parquet(output_path)

            # Buscar archivo part-*.parquet
            part_files = glob.glob(os.path.join(output_path, 'part-*.parquet'))
            if part_files:
                part_file = part_files[0]
                print(f"   ☁️ Subiendo a MinIO: {parquet_filename}")
                
                with open(part_file, 'rb') as data:
                    s3.upload_fileobj(data, BUCKET_NAME, parquet_filename)
                
                print(f"   ✅ {filename} procesado exitosamente")
            else:
                print(f"   ❌ No se encontró archivo parquet generado para {filename}")

        except Exception as file_error:
            print(f"   🚨 Error procesando {filename}: {file_error}")
            continue

    print("\n🎉 Conversión y subida completadas con Spark.")

except Exception as e:
    print(f"🚨 Error durante la ejecución: {e}")
    print("\n🔧 Sugerencias de debugging:")
    print("1. Verifica que los contenedores de Spark estén ejecutándose:")
    print("   docker ps | grep spark")
    print("2. Verifica que MinIO esté ejecutándose:")
    print("   docker ps | grep minio")
    print("3. Verifica que el directorio CSV existe y tiene archivos:")
    print(f"   ls -la {CSV_DIR}")
    print("4. Si ejecutas desde un contenedor, usa 'spark-master:7077'")
    print("5. Si ejecutas desde el host, usa 'localhost:7077'")

finally:
    if spark:
        try:
            spark.stop()
            print("🛑 Spark detenido correctamente.")
        except Exception as stop_error:
            print(f"⚠️ Error deteniendo Spark: {stop_error}")