"""
prepare_from_minio.py
Convierte training.parquet (Bronze) ➜ training.silver.parquet y lo sube a MinIO
Ahora SIN S3A: se descarga con boto3, se procesa localmente.
"""

from dotenv import load_dotenv
load_dotenv()

import os, shutil, glob, boto3, tempfile
from pyspark.sql.functions import col, upper, when, trim
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from utils_spark import iniciar_spark

# ────────── CONFIG ──────────
BRONZE  = "training.parquet"
SILVER  = "training.silver.parquet"
BUCKET  = "datasets"
ENDPOINT = "http://localhost:9100"

ACCESS  = os.getenv("AWS_ACCESS_KEY_ID")
SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY")

# 📥 1. Descargar parquet Bronze a tmp
tmp_dir = tempfile.mkdtemp(prefix="bronze_")
local_bronze = os.path.join(tmp_dir, BRONZE)

# Parametros de conexion a MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS,
    aws_secret_access_key=SECRET,
)
print("☁️ Descargando Bronze …")
s3.download_file(BUCKET, BRONZE, local_bronze)
print("✅ Bronze local:", local_bronze)

# 🧪 2. Spark local → crear DF Silver
spark, _ = iniciar_spark(app_name="silver_to_minio_local")
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(local_bronze)

"""
Dicionario de Variables a extraer y limpiar:
IsBadBuy - Identifica si el vehículo marcado como kick fue una compra evitable.
VehYear - Año de fabricación del vehículo.
VehicleAge - Años transcurridos desde el año de fabricación.
VehOdo - Lectura del odómetro del vehículo.
VehBCost - Costo de adquisición pagado al momento de la compra.
IsOnlineSale - Indica si el vehículo se compró originalmente en línea.
Transmission - Tipo de transmisión del vehículo (Automática, Manual).
Size - Categoría de tamaño del vehículo (Compacto, SUV, etc.).
MMRAcquisitionRetailAveragePrice - Precio de adquisición de este vehículo en el mercado minorista, en condición promedio, al momento de la compra.
MMRCurrentAuctionAveragePrice - Precio de adquisición de este vehículo en subasta en condición promedio a la fecha actual.
MMRCurrentAuctionCleanPrice - Precio de adquisición de este vehículo en subasta en condición superior a la fecha actual.
"""

# Limpieza / ingeniería de variables
df = (
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
        .otherwise(None),
    )
    .withColumn(
        "Size_clean",
        when(
            trim(col("Size")).isin(
                "SPORTS","SMALL SUV","CROSSOVER","VAN","COMPACT",
                "SMALL TRUCK","SPECIALTY","MEDIUM","MEDIUM SUV",
                "LARGE SUV","LARGE TRUCK","LARGE"
            ),
            trim(col("Size")),
        ).otherwise(None),
    )
)

indexers = [
    StringIndexer(
        inputCol="Transmission_clean",
        outputCol="Transmission_idx",
        handleInvalid="keep",
    ),
    StringIndexer(
        inputCol="Size_clean",
        outputCol="Size_idx",
        handleInvalid="keep",
    ),
]

pipe_df = Pipeline(stages=indexers).fit(df).transform(df)

silver_cols = [
    "IsBadBuy", "VehYear", "VehicleAge", "VehOdo", "VehBCost",
    "IsOnlineSale", "Transmission_idx", "Size_idx",
    "MMRAcquisitionRetailAveragePrice",
    "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
]
silver_df = pipe_df.select(silver_cols)

# 💾 3. Guardar Silver local (un solo part-file)
tmp_silver = os.path.join(tmp_dir, "silver_tmp")
print("💾 Escribiendo Silver local …")
shutil.rmtree(tmp_silver, ignore_errors=True)
(
    silver_df
    .repartition(1)
    .write.mode("overwrite")
    .parquet(tmp_silver, compression="snappy")
)
part_path = glob.glob(os.path.join(tmp_silver, "part-*.parquet"))[0]
print("   ➜ Archivo listo:", part_path)

# ☁️ 4. Subir Silver a MinIO
print(f"☁️ Subiendo a MinIO → {BUCKET}/{SILVER}")
with open(part_path, "rb") as data:
    s3.upload_fileobj(data, BUCKET, SILVER)
print("✅ Silver cargado en MinIO.")

# 🧹 5. Limpieza
shutil.rmtree(tmp_dir, ignore_errors=True)
spark.stop()
print("🛑 Spark detenido y temporales eliminados.")
