# ================= kick_prediction_silver.py (with CrossValidation) =================
"""
Entrena y evalúa un RandomForest con balanceo de clases y parámetros optimizados
(usando CrossValidator) sobre la **capa Silver (training.silver.parquet)**
leyendo directamente desde MinIO vía S3A.
• Ajusta pesos de la clase minoritaria en `weightCol`.
• Optimiza hiperparámetros con CrossValidator.
• Métricas + gráficas del mejor modelo.
"""

from dotenv import load_dotenv
import os
import json # If used, keep it
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator # Added for tuning
from pyspark.sql.functions import col, when
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ────────────────────────────
# Configuración
# ────────────────────────────
load_dotenv()
PARQUET_DIR_NAME = "training.silver.parquet"
BUCKET_NAME      = "datasets"
MINIO_SPARK_ENDPOINT = "http://localhost:9100"
ACCESS_KEY       = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY       = os.getenv("AWS_SECRET_ACCESS_KEY")

FIGURES_HOST_DIR = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/figures"
os.makedirs(FIGURES_HOST_DIR, exist_ok=True)

if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID / SECRET_KEY en .env")

print("🔐 AWS_ACCESS_KEY_ID:", ACCESS_KEY)
print("🔐 AWS_SECRET_ACCESS_KEY:", "********" if SECRET_KEY else None)


def create_spark_session_s3a(app_name="KickPredictionCV_S3A"):
    """Crea una sesión de Spark configurada para MinIO S3A"""
    hadoop_aws_version = "3.3.4"
    aws_sdk_version = "1.12.367"

    print(f"🛠️ Configurando Spark para S3A con endpoint: {MINIO_SPARK_ENDPOINT}")
    return (
        SparkSession.builder
            .appName(app_name)
            .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY)
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_SPARK_ENDPOINT)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            # ───── Recursos executor ─────
            .config("spark.executor.instances", "1")
            .config("spark.executor.cores",     "2")
            .config("spark.executor.memory",    "2g")
            .config("spark.cores.max",          "2")   # límite global para la app
            .config("spark.jars.packages",
                    f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version},"
                    f"com.amazonaws:aws-java-sdk-bundle:{aws_sdk_version}")
            .getOrCreate()
    )

spark = None
train_df_cached = None # To keep track if train_df was cached

try:
    # 1️⃣ Spark con configuración S3A
    spark = create_spark_session_s3a()
    spark.sparkContext.setLogLevel("ERROR")

    # 2️⃣ Leer Parquet directamente desde MinIO
    s3a_parquet_path = f"s3a://{BUCKET_NAME}/{PARQUET_DIR_NAME}"
    print(f"📖 Leyendo Silver Parquet directamente desde MinIO: {s3a_parquet_path}")
    df_full = spark.read.parquet(s3a_parquet_path)
    print(f"✅ Datos originales leídos. Registros: {df_full.count():,}, Columnas: {len(df_full.columns)}")
    df_full.printSchema()

    # 3️⃣ Selección / casting
    # Eliminé la columna "Make_Size_idx",
    label = "IsBadBuy"
    feature_cols = [
             "VehYear", "VehicleAge", "VehOdo", "VehBCost",
            "IsOnlineSale", "Transmission_idx", "Size_idx", "Make_idx",
            "Size_SUV","IsLuxury","IsAutomatic", "MMRAcquisitionRetailAveragePrice",
            "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
    ]

    missing_cols = [col_name for col_name in [label] + feature_cols if col_name not in df_full.columns]
    if missing_cols:
        raise ValueError(f"❌ Columnas faltantes en el DataFrame: {missing_cols}. Columnas disponibles: {df_full.columns}")

    df_selected = df_full.select([label] + feature_cols)
    print(f"🔍 Después de seleccionar columnas, registros: {df_selected.count():,}")
    
    # Handle NaNs more explicitly if needed, e.g., Imputation
    # For now, using dropna as in the original script
    df_cleaned = df_selected.dropna()
    rows_dropped = df_selected.count() - df_cleaned.count()
    if rows_dropped > 0:
        print(f"⚠️ {rows_dropped:,} filas eliminadas debido a valores NaN.")
    
    df = df_cleaned.withColumn(label, col(label).cast("double"))
    print(f"Después de quitar NaNs y castear label, registros: {df.count():,}")

    # 3.a Balanceo de clase
    pos_count = df.filter(col(label) == 1).count()
    neg_count = df.filter(col(label) == 0).count()
    
    if pos_count == 0:
        print("⚠️ Advertencia: No hay instancias positivas (IsBadBuy=1). El balanceo no se aplicará efectivamente.")
        balancing_ratio = 1.0
    elif neg_count == 0:
        print("⚠️ Advertencia: No hay instancias negativas (IsBadBuy=0). El balanceo no se aplicará efectivamente.")
        balancing_ratio = 1.0
    else:
        balancing_ratio = neg_count / pos_count
    
    print(f"Ratio de balanceo (neg/pos): {balancing_ratio:.2f} ({neg_count}/{pos_count})")
    df_balanced = df.withColumn(
        "classWeightCol",
        when(col(label) == 1, balancing_ratio).otherwise(1.0)
    )

    # 4️⃣ Ensamblado features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    data = assembler.transform(df_balanced).select("features", label, "classWeightCol")
    print(f"Después de ensamblar features, registros para modelado: {data.count():,}")

    # --- HYPERPARAMETER TUNING SETUP ---
    # 5️⃣ Modelo RandomForest (base para tuning)
    rf = RandomForestClassifier(
        labelCol=label,
        featuresCol="features",
        weightCol="classWeightCol",
        seed=42
    )

    # Define Parameter Grid
    # Start with a smaller grid for faster initial tuning, then expand if needed
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [100, 200])       # Fewer options initially
                 .addGrid(rf.maxDepth, [5, 8, 12])        # Explore depth
                 .addGrid(rf.minInstancesPerNode, [5, 10]) # Fewer options
                 .addGrid(rf.maxBins, [64, 100])           # Ensure > max categories
                 # .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"]) # Can add later
                 # .addGrid(rf.subsamplingRate, [0.8, 0.9])       # Can add later
                 .build())
    print(f"\n⚙️ Parámetros a probar en Cross-Validation: {len(paramGrid)} combinaciones.")

    # Define Evaluator
    cv_evaluator = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

    # Setup CrossValidator
    crossval = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=cv_evaluator,
        numFolds=3,  # 3 for faster iteration; 5 for more robust results
        seed=42,
        parallelism=2 # Number of models to train in parallel
    )

    # 6️⃣ Train/Test Split
    train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
    
    print(f"🏋️ Datos de entrenamiento: {train_df.count():,} registros")
    print(f"🧪 Datos de prueba: {test_df.count():,} registros")

    train_df_cached = train_df.cache() # Cache training data
    
    print("\n⏳ Iniciando Cross-Validation para optimizar RandomForest (esto puede tardar)...")
    cvModel = crossval.fit(train_df_cached)
    print("✅ Cross-Validation completada.")

    # Get the best model
    bestModel = cvModel.bestModel

    print("\n✨ Mejores Parámetros Encontrados por Cross-Validation:")
    print(f"  numTrees: {bestModel.getNumTrees}")
    print(f"  maxDepth: {bestModel.getMaxDepth()}")
    print(f"  minInstancesPerNode: {bestModel.getMinInstancesPerNode()}")
    print(f"  maxBins: {bestModel.getMaxBins()}")
    # These might not be set if not in paramGrid or if they default
    try: print(f"  featureSubsetStrategy: {bestModel.getFeatureSubsetStrategy()}")
    except: pass
    try: print(f"  subsamplingRate: {bestModel.getSubsamplingRate()}")
    except: pass
    
    # Make predictions on the test set using the best model
    print("\n🧪 Evaluando el mejor modelo en el conjunto de prueba...")
    preds = bestModel.transform(test_df)

    # 7️⃣ Métricas
    eval_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
    eval_f1  = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    eval_acc = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")

    print("\n📈 Resultados del Mejor Modelo (de Cross-Validation):")
    auc_score = eval_auc.evaluate(preds)
    f1_score = eval_f1.evaluate(preds)
    acc_score = eval_acc.evaluate(preds)
    print(f"🔹 AUC:       {auc_score:.4f}")
    print(f"🔹 F1-score:  {f1_score:.4f}")
    print(f"🔹 Accuracy:  {acc_score:.4f}")

    # ───────────────────── Gráficas (se guardan localmente) ───────────────
    print("\n📊 Generando y guardando gráficas del mejor modelo...")

    # Importancia de features
    importances = pd.Series(bestModel.featureImportances.toArray(), index=feature_cols)
    imp_df = importances.sort_values(ascending=False).reset_index()
    imp_df.columns = ["Feature","Importance"]
    plt.figure(figsize=(10,6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Importancia de Características (Mejor Modelo RF)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_HOST_DIR, "rf_cv_feat_importance.png"))
    plt.close()
    print(f"   Figuras guardada: rf_cv_feat_importance.png")

    # Confusion Matrix
    y_true_pd = preds.select(label).toPandas()
    y_pred_pd = preds.select("prediction").toPandas()
    cm = confusion_matrix(y_true_pd[label], y_pred_pd["prediction"])
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Matriz de Confusión (Mejor Modelo RF)")
    plt.savefig(os.path.join(FIGURES_HOST_DIR, "rf_cv_conf_matrix.png"))
    plt.close()
    print(f"   Figuras guardada: rf_cv_conf_matrix.png")

    # ROC Curve
    y_score_rdd = preds.select("probability").rdd.map(lambda row: float(row[0][1]))
    y_score_list = y_score_rdd.collect()

    fpr, tpr, _ = roc_curve(y_true_pd[label], y_score_list)
    roc_auc_value = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Característica Operativa del Receptor (ROC) - Mejor Modelo RF')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_HOST_DIR, "rf_cv_roc_curve.png"))
    plt.close()
    print(f"   Figuras guardada: rf_cv_roc_curve.png")

    print(f"\n📊 Todas las figuras guardadas en: {FIGURES_HOST_DIR}")

except Exception as e:
    import traceback
    print(f"❌ Error durante el modelado: {e}")
    traceback.print_exc()

finally:
    if train_df_cached:
        try:
            train_df_cached.unpersist()
            print("\n🧹 train_df des-cacheado.")
        except Exception as e_unpersist:
            print(f"⚠️ Error des-cacheando train_df: {e_unpersist}")
    if spark:
        try:
            spark.stop()
            print("🛑 Sesión Spark detenida.")
        except Exception as e_spark:
            print(f"⚠️ Error deteniendo Spark: {e_spark}")