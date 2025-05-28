from dotenv import load_dotenv
import os
from pyspark.sql import SparkSession
from utils_spark import export_schema_info, export_describe, export_sample
from utils_spark import iniciar_spark
# Cargar variables de entorno
load_dotenv()

# Configuración
PARQUET_FILENAME = 'training.parquet'
BUCKET_NAME = 'datasets'
MINIO_URL = 'http://localhost:9100'
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Validar credenciales
if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Las variables AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY no están definidas.")


spark, parquet_s3_path = iniciar_spark(app_name="preview_from_minio", parquet_filename=PARQUET_FILENAME)

# Reducir el nivel de logs
spark.sparkContext.setLogLevel("ERROR")

try:
    print(f"📂 Leyendo archivo: {parquet_s3_path}")
    df = spark.read.parquet(parquet_s3_path)

    export_schema_info(df)
    export_describe(df)
    export_sample(df)

except Exception as e:
    print(f"❌ Error al leer el archivo: {e}")
finally:
    spark.stop()
    print("🛑 Spark detenido.")
