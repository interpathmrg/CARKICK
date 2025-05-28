from utils_spark import iniciar_spark
from pyspark.sql.functions import col, countDistinct
from dotenv import load_dotenv
import os
import json

# Cargar variables de entorno
load_dotenv()

# Iniciar Spark
spark, parquet_s3_path = iniciar_spark(app_name="Data Quality Checker", parquet_filename="training.parquet")

try:
    print(f"📂 Leyendo archivo: {parquet_s3_path}")
    df = spark.read.parquet(parquet_s3_path)

    print("🔎 Calculando valores únicos por columna...")
    resultados = []

    for col_name in df.columns:
        # Contar valores únicos
        unique_count = df.select(countDistinct(col(col_name))).collect()[0][0]
        
        # Obtener hasta 20 ejemplos únicos
        ejemplos = df.select(col_name).distinct().limit(20).rdd.flatMap(lambda x: x).collect()
        ejemplos_serializados = json.dumps(ejemplos, default=str)

        resultados.append((col_name, unique_count, ejemplos_serializados))
        

    # Crear DataFrame de resultados
    resultado_df = spark.createDataFrame(resultados, ["Columna", "Valores Únicos", "Ejemplos Únicos"])

    # Guardar como CSV
    resultado_df.coalesce(1).write \
        .option("header", True) \
        .option("sep", ";") \
        .option("quote", '"') \
        .mode("overwrite") \
        .csv("valores_unicos_output.csv")

    print("✅ Análisis de calidad completado: valores únicos exportados a valores_unicos_output.csv")

except Exception as e:
    print(f"❌ Error durante el análisis: {e}")

finally:
    spark.stop()
    print("🛑 Spark detenido.")
