
"""
prepare_from_minio_direct.py
Convierte training.parquet (Bronze) ➜ training.silver.parquet usando Spark con acceso directo a MinIO
Lee y escribe directamente desde/hacia MinIO sin descargas locales
"""

from dotenv import load_dotenv
load_dotenv()

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, when, trim
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# ────────── CONFIG ──────────
BRONZE_FILE = "training.parquet"
SILVER_FILE = "training.silver.parquet"
BUCKET = "datasets"
# MINIO_ENDPOINT = "localhost:9100"  # Sin http:// para Spark - This was potentially confusing
# MINIO_URL = "http://localhost:9100"  # Con http:// para display - This was potentially confusing

# Correct MinIO endpoint for Spark and display (assuming MinIO runs on 9000 for S3 API)
MINIO_SPARK_ENDPOINT = "http://localhost:9100"
MINIO_DISPLAY_URL = "http://localhost:9100" # For consistency in display messages

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

print("🔐 AWS_ACCESS_KEY_ID:", ACCESS_KEY) # Good to print the actual value being used
print("🔐 AWS_SECRET_ACCESS_KEY:", "********" if SECRET_KEY else None) # Mask secret key

def create_spark_session():
    """Crea una sesión de Spark configurada para MinIO"""
    # Choose versions compatible with your Spark/Hadoop setup
    # For Spark 3.3.x, Hadoop 3.3.x is common
    # Check https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-aws
    # and https://mvnrepository.com/artifact/com.amazonaws/aws-java-sdk-bundle
    hadoop_aws_version = "3.3.4"  # Example, adjust if needed
    aws_sdk_version = "1.12.367" # Example, adjust if needed

    return (
        SparkSession.builder
            .appName("MinIO Bronze to Silver") # More descriptive app name
            # .appName("App") # Redundant, removed
            .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY)
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_SPARK_ENDPOINT) # Use the defined constant
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            # --- ADD THESE LINES ---
            .config("spark.jars.packages", 
                    f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version},"
                    f"com.amazonaws:aws-java-sdk-bundle:{aws_sdk_version}")
            # For local development, ensure Spark can run
            # .master("local[*]") # Uncomment if not running through spark-submit with a master defined
            .getOrCreate()
    )

def main():
    print("🚀 Iniciando procesamiento Bronze → Silver con acceso directo a MinIO")

    # Crear sesión Spark
    spark = create_spark_session()

    try:
        # Rutas S3A para MinIO
        bronze_path = f"s3a://{BUCKET}/{BRONZE_FILE}"
        silver_path = f"s3a://{BUCKET}/{SILVER_FILE}"

        print(f"📖 Leyendo archivo Bronze desde: {bronze_path}")
        print(f"ℹ️ Usando MinIO endpoint para Spark: {MINIO_SPARK_ENDPOINT}")

        # 1️⃣ Leer datos Bronze directamente desde MinIO
        df = spark.read.parquet(bronze_path)

        print(f"✅ Datos leídos exitosamente. Registros: {df.count():,}")
        print("📊 Esquema del dataset:")
        df.printSchema()

        # 2️⃣ Selección y limpieza de datos
        print("🧹 Aplicando transformaciones y limpieza...")

        # ... (rest of your data transformation logic remains the same) ...
        df_cleaned = (
            df.select(
                "IsBadBuy", "VehYear", "VehicleAge", "VehOdo", "VehBCost",
                "IsOnlineSale", "Transmission", "Size",
                "MMRAcquisitionRetailAveragePrice",
                "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
            )
            .withColumn(
                "Transmission_clean",
                when(upper(trim(col("Transmission"))) == "MANUAL", "MANUAL")
                .when(upper(trim(col("Transmission"))) == "AUTO", "AUTO")
                .otherwise(None), # Consider using a specific "UNKNOWN" string if None causes issues downstream
            )
            .withColumn(
                "Size_clean",
                when(
                    trim(upper(col("Size"))).isin( # Standardize to upper for comparison
                        "SPORTS", "SMALL SUV", "CROSSOVER", "VAN", "COMPACT",
                        "SMALL TRUCK", "SPECIALTY", "MEDIUM", "MEDIUM SUV",
                        "LARGE SUV", "LARGE TRUCK", "LARGE"
                    ),
                    trim(upper(col("Size"))), # Store the standardized upper case version
                ).otherwise(None), # Consider "UNKNOWN"
            )
        )

        # 3️⃣ Aplicar StringIndexer para variables categóricas
        print("🔢 Aplicando StringIndexer...")

        indexers = [
            StringIndexer(
                inputCol="Transmission_clean",
                outputCol="Transmission_idx",
                handleInvalid="keep", # 'keep' will assign a special index to unseen/null values
            ),
            StringIndexer(
                inputCol="Size_clean",
                outputCol="Size_idx",
                handleInvalid="keep",
            ),
        ]

        pipeline = Pipeline(stages=indexers)
        pipeline_model = pipeline.fit(df_cleaned)
        df_indexed = pipeline_model.transform(df_cleaned)

        # 4️⃣ Selección final de columnas Silver
        silver_columns = [
            "IsBadBuy", "VehYear", "VehicleAge", "VehOdo", "VehBCost",
            "IsOnlineSale", "Transmission_idx", "Size_idx",
            "MMRAcquisitionRetailAveragePrice",
            "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
        ]

        silver_df = df_indexed.select(silver_columns)

        print("📈 Dataset Silver preparado:")
        print(f"   • Registros: {silver_df.count():,}")
        print(f"   • Columnas: {len(silver_df.columns)}")

        print("\n📊 Primeras 5 filas del dataset Silver:")
        silver_df.show(5)

        # 5️⃣ Guardar directamente en MinIO
        print(f"💾 Guardando archivo Silver en: {silver_path}")

        (
            silver_df
            .repartition(1)
            .write
            .mode("overwrite")
            .option("compression", "snappy")
            .parquet(silver_path)
        )

        print("✅ Archivo Silver guardado exitosamente en MinIO!")
        # Use the consistent display URL
        print(f"🔗 Ubicación: {MINIO_DISPLAY_URL}/{BUCKET}/{SILVER_FILE}")

        print("\n🔍 Verificando archivo guardado...")
        verification_df = spark.read.parquet(silver_path)
        print(f"   ✓ Registros verificados: {verification_df.count():,}")

    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for better debugging
        # raise # Re-raise if you want the script to exit with an error code

    finally:
        print("\n🧹 Cerrando sesión Spark...")
        if 'spark' in locals() and spark: # Check if spark session was created
            spark.stop()
        print("🛑 Procesamiento completado.")

if __name__ == "__main__":
    main()