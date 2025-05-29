"""
silver_to_minio.py  –  Graba el DF “Silver” localmente y lo sube a MinIO
"""
from dotenv import load_dotenv
load_dotenv()

import os, shutil, glob, boto3
from pyspark.sql.functions import col, upper, when, trim
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from utils_spark import iniciar_spark

# ────────── CONFIG ──────────
BRONZE  = "training.parquet"
SILVER  = "training.silver.parquet"      # nombre en MinIO
TMP_DIR = "/tmp/carkick_silver_tmp"      # directorio local
BUCKET  = "datasets"
ENDPOINT = "http://localhost:9100"

ACCESS  = os.getenv("AWS_ACCESS_KEY_ID")
SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY")

# ────────── Spark: crear DF Silver ──────────
spark, bronze_path = iniciar_spark(app_name="silver_to_minio", parquet_filename=BRONZE)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(bronze_path)

# limpieza rápida (mismo logic que antes)
df = (
    df.select(
        "IsBadBuy", "VehYear", "VehicleAge", "VehOdo", "VehBCost",
        "IsOnlineSale", "Transmission", "Size"
    )
    .withColumn("Transmission_clean",
                when(upper(trim(col("Transmission"))) == "MANUAL", "MANUAL")
                .when(upper(trim(col("Transmission"))) == "AUTO", "AUTO")
                .otherwise(None))
    .withColumn("Size_clean",
                when(trim(col("Size")).isin(
                    "SPORTS","SMALL SUV","CROSSOVER","VAN","COMPACT",
                    "SMALL TRUCK","SPECIALTY","MEDIUM","MEDIUM SUV",
                    "LARGE SUV","LARGE TRUCK","LARGE"), trim(col("Size")))
                .otherwise(None))
)

indexers = [
    StringIndexer(inputCol="Transmission_clean", outputCol="Transmission_idx", handleInvalid="keep"),
    StringIndexer(inputCol="Size_clean",          outputCol="Size_idx",          handleInvalid="keep")
]

pipe_df = Pipeline(stages=indexers).fit(df).transform(df)

silver_cols = [
    "IsBadBuy", "VehYear", "VehicleAge", "VehOdo", "VehBCost",
    "IsOnlineSale", "Transmission_idx", "Size_idx"
]
silver_df = pipe_df.select(silver_cols)

# ────────── Guardar local (reparition(1) → solo un part-*.parquet) ──────────
print("💾 Escribiendo Silver local …")
shutil.rmtree(TMP_DIR, ignore_errors=True)
(
    silver_df
    .repartition(1)
    .write.mode("overwrite").parquet(TMP_DIR, compression="snappy")
)

# Buscar el único part-file
part_path = glob.glob(os.path.join(TMP_DIR, "part-*.parquet"))[0]
print(f"   ➜ Archivo listo: {part_path}")

# ────────── Subir a MinIO con boto3 ──────────
print(f"☁️ Subiendo a MinIO → {BUCKET}/{SILVER}")
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS,
    aws_secret_access_key=SECRET,
)

with open(part_path, "rb") as data:
    s3.upload_fileobj(data, BUCKET, SILVER)

print("✅ Silver cargado en MinIO.")

# ────────── Limpieza ──────────
shutil.rmtree(TMP_DIR, ignore_errors=True)
spark.stop()

