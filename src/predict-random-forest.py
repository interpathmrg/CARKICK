from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT # Import VectorUDT
from pyspark.sql.functions import col, when, concat_ws, upper, trim, regexp_replace, udf # Import udf
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.types import DoubleType # Import DoubleType

# Configura las rutas
csv_path = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/csv/test.csv"
pipeline_path = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/modelos/pipeline_rf"
assembler_path = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/modelos/assembler"
model_path = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/modelos/rf_model_final"

print("🚀 Iniciando predicción en producción...")

spark = SparkSession.builder \
    .appName("Kick Prediction - Production") \
    .master("local[*]") \
    .getOrCreate()

print("📄 Cargando archivo de prueba...")
df_raw = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)
print(f"✅ {df_raw.count():,} registros cargados.")
print("Schema of raw test data:")
df_raw.printSchema()

has_isbadbuy_column = "IsBadBuy" in df_raw.columns
has_refid_column = "RefId" in df_raw.columns

print("🧹 Aplicando transformaciones y limpieza...")

select_cols_from_raw = [
    "VehYear", "VehicleAge", "VehOdo", "VehBCost",
    "IsOnlineSale", "Transmission", "Size",
    "MMRAcquisitionRetailAveragePrice",
    "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice", "Make"
]

if has_refid_column:
    select_cols_from_raw.append("RefId")
    print("ℹ️ 'RefId' column found in test data. Will be carried through.")
else:
    print("⚠️ 'RefId' column NOT found in test data. It won't be in the output.")

if has_isbadbuy_column:
    select_cols_from_raw.append("IsBadBuy")
    print("ℹ️ 'IsBadBuy' column found in test data. Will be carried through.")
else:
    print("⚠️ 'IsBadBuy' column NOT found in test data. Predictions will be generated without it.")

df_selected_raw = df_raw.select(*select_cols_from_raw)

df_cleaned = (
    df_selected_raw
    .withColumn(
        "Transmission_clean",
        when(upper(trim(col("Transmission"))) == "MANUAL", "MANUAL")
        .when(upper(trim(col("Transmission"))) == "AUTO", "AUTO")
        .when(upper(trim(col("Transmission"))) == "AUTOMATIC", "AUTO")
        .otherwise(None),
    )
    .withColumn(
        "Size_clean",
        when(
            trim(upper(col("Size"))).isin(
                "SPORTS", "SMALL SUV", "CROSSOVER", "VAN", "COMPACT",
                "SMALL TRUCK", "SPECIALTY", "MEDIUM", "MEDIUM SUV",
                "LARGE SUV", "LARGE TRUCK", "LARGE"
            ),
            trim(upper(col("Size"))),
        ).otherwise(None),
    )
)

df_cleaned = (
    df_cleaned
    .withColumn("VehYear", col("VehYear").cast(DoubleType()))
    .withColumn("VehicleAge", col("VehicleAge").cast(DoubleType()))
    .withColumn("VehOdo", col("VehOdo").cast(DoubleType()))
    .withColumn("VehBCost", col("VehBCost").cast(DoubleType()))
    .withColumn("IsOnlineSale", col("IsOnlineSale").cast(DoubleType()))
    .withColumn("MMRAcquisitionRetailAveragePrice", regexp_replace(col("MMRAcquisitionRetailAveragePrice"), "[$,]", "").cast(DoubleType()))
    .withColumn("MMRCurrentAuctionAveragePrice", regexp_replace(col("MMRCurrentAuctionAveragePrice"), "[$,]", "").cast(DoubleType()))
    .withColumn("MMRCurrentAuctionCleanPrice", regexp_replace(col("MMRCurrentAuctionCleanPrice"), "[$,]", "").cast(DoubleType()))
    .withColumn("Make_clean", trim(upper(col("Make"))))
    .withColumn("IsAutomatic", F.when(F.col("Transmission") == "Automatic", 1).otherwise(0).cast(DoubleType()))
    .withColumn("IsLuxury", F.when(F.col("Make").isin(["BMW","MINI","LINCOLN", "CADILLAC", "MERCEDES", "LEXSUS", "AUDI", "TESLA", "ACURA"]), 1).otherwise(0).cast(DoubleType()))
    .withColumn("Size_SUV", F.when(F.col("Size").like("%SUV%"), 1).otherwise(0).cast(DoubleType()))
    .withColumn("Make_Size", F.concat_ws("_", col("Make_clean"), col("Size_clean")))
)
print("Schema after cleaning and transformations:")
df_cleaned.printSchema()

print("🛠️ Cargando pipeline de preprocesamiento (StringIndexers)...")
pipeline_model_indexers = PipelineModel.load(pipeline_path)

print("⚙️ Aplicando pipeline de StringIndexers al conjunto de prueba...")
df_indexed = pipeline_model_indexers.transform(df_cleaned)
print("Schema after StringIndexer pipeline:")
df_indexed.printSchema()

print("🧩 Cargando VectorAssembler...")
assembler_model = VectorAssembler.load(assembler_path)

print("🧬 Aplicando VectorAssembler...")
df_assembled = assembler_model.transform(df_indexed)
print("Schema after VectorAssembler:")
df_assembled.printSchema()

print("🔍 Cargando modelo entrenado...")
model = RandomForestClassificationModel.load(model_path)

print("🔮 Generando predicciones...")
predicciones = model.transform(df_assembled)

# --- Handle probability column for CSV output ---
def get_prob_class_1(prob_vector):
    if prob_vector is None:
        return None
    try:
        # Assuming prob_vector is a DenseVector or SparseVector
        return float(prob_vector.toArray()[1]) # Convert to array then access element
    except (IndexError, TypeError, AttributeError): # Added AttributeError for cases like None.toArray()
        return None

# Register the UDF
# The input type to the UDF is a Vector (from the probability column)
# VectorUDT might not be the direct schema here for udf registration in newer Spark versions for this simple case.
# Spark often infers the input type. The return type is important.
get_prob_class_1_udf = udf(get_prob_class_1, DoubleType()) # <--- CORRECTED: UDF definition assigned to variable

predicciones_for_csv = predicciones.withColumn("probability_class_1", get_prob_class_1_udf(col("probability")))


print("📊 Resultados (primeras 10 filas):")
show_cols_final = ["prediction", "probability_class_1"]
if has_refid_column and "RefId" in predicciones_for_csv.columns:
    show_cols_final.insert(0, "RefId")
if has_isbadbuy_column and "IsBadBuy" in predicciones_for_csv.columns:
    show_cols_final.insert(1, "IsBadBuy")
# Show original probability vector as well for comparison during debugging
predicciones_for_csv.select(*show_cols_final, "probability").show(10, truncate=False)


output_predictions_path = "/home/mrgonzalez/PYTHON/CARKICK/output/predicciones_rf"
print(f"💾 Guardando predicciones en: {output_predictions_path}")

save_cols_final = ["prediction", "probability_class_1"]
if has_refid_column and "RefId" in predicciones_for_csv.columns:
    save_cols_final.insert(0, "RefId")
if has_isbadbuy_column and "IsBadBuy" in predicciones_for_csv.columns:
    save_cols_final.insert(1, "IsBadBuy")

(predicciones_for_csv.select(*save_cols_final)
    .coalesce(1)
    .write.option("header", True)
    .mode("overwrite")
    .csv(output_predictions_path)
)

if has_isbadbuy_column and "IsBadBuy" in predicciones.columns:
    print("🎯 Evaluando modelo en datos de prueba...")
    predicciones_eval = predicciones.withColumn("IsBadBuy", col("IsBadBuy").cast(DoubleType()))
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(labelCol="IsBadBuy", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predicciones_eval)
    print(f"🎯 AUC en Producción (sobre datos de prueba): {auc:.4f}")
else:
    print("⚠️ 'IsBadBuy' column was not in the input test data or not carried through. Skipping evaluation.")

spark.stop()
print("✅ Predicción completada.")