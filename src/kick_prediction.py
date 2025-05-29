# ================= kick_prediction_silver.py =================
"""
Entrena y evalúa un RandomForest con balanceo de clases y parámetros optimizados
sobre la **capa Silver (training.silver.parquet)** sin usar S3A.
• Descarga vía boto3 → Spark local.
• Ajusta pesos de la clase minoritaria en `weightCol`.
• Métricas + gráficas.
"""

from dotenv import load_dotenv
import os, shutil, tempfile, boto3, json
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when
from utils_spark import iniciar_spark
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ────────────────────────────
# Configuración
# ────────────────────────────
load_dotenv()
PARQUET = "training.silver.parquet"
BUCKET   = "datasets"
ENDPT    = "http://localhost:9100"
AK       = os.getenv("AWS_ACCESS_KEY_ID")
SK       = os.getenv("AWS_SECRET_ACCESS_KEY")
# --- directorio compartido ---------------------------------
HOST_DIR   = "/home/mrgonzalez/Desktop/PYTHON/CARKICK/data"  # lado host
CONTAINER_DIR = "/data"                                      # cómo lo ve el worker
PARQUET = "training.silver.parquet"


if not AK or not SK:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID / SECRET_KEY en .env")


# ruta real donde descarga el host
local_download = os.path.join(HOST_DIR, PARQUET)
os.makedirs(HOST_DIR, exist_ok=True)


try:
    # 1️⃣ Descargar
    s3 = boto3.client("s3", endpoint_url=ENDPT, aws_access_key_id=AK, aws_secret_access_key=SK)
    s3.download_file(BUCKET, PARQUET, local_download)
    print("✅ Parquet local →", local_download)


    # 2️⃣ Spark
    spark,  parquet_s3_path = iniciar_spark(app_name="kick_prediction_balanced")
    shared_path_for_spark = os.path.join(CONTAINER_DIR, PARQUET)
    df = spark.read.parquet(shared_path_for_spark)

    # 3️⃣ Selección / casting
    label = "IsBadBuy"
    feature_cols = [
        "VehicleAge", "VehOdo", "VehBCost", "IsOnlineSale",
        "Transmission_idx", "Size_idx","MMRAcquisitionRetailAveragePrice",
        "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
    ]
    df = df.select([label] + feature_cols).dropna()
    df = df.withColumn(label, col(label).cast("double"))

    # 3.a Balanceo de clase
    pos = df.filter(col(label) == 1).count()
    neg = df.filter(col(label) == 0).count()
    balancing_ratio = neg / pos if pos else 1.0
    df = df.withColumn(
        "classWeightCol",
        when(col(label) == 1, balancing_ratio).otherwise(1.0)
    )

    # 4️⃣ Ensamblado features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df).select("features", label, "classWeightCol")

    # 5️⃣ Modelo RandomForest optimizado
    rf = RandomForestClassifier(
        labelCol            = label,
        featuresCol         = "features",
        weightCol           = "classWeightCol",
        numTrees            = 200,
        maxDepth            = 5,
        minInstancesPerNode = 5,
        maxBins             = 64,
        featureSubsetStrategy = "sqrt",
        subsamplingRate     = 0.8,
        seed                = 42,
    )

    # 6️⃣ Train/Test
    train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
    model = rf.fit(train_df)
    preds = model.transform(test_df)

    # 7️⃣ Métricas
    eval_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
    eval_f1  = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    eval_acc = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")
    print("📈 Resultados ↴")
    print(f"🔹 AUC:       {eval_auc.evaluate(preds):.4f}")
    print(f"🔹 F1-score:  {eval_f1.evaluate(preds):.4f}")
    print(f"🔹 Accuracy:  {eval_acc.evaluate(preds):.4f}")

  # ───────────────────── Gráficas (se guardan) ───────────────
    FIG_DIR = HOST_DIR  # guarda junto al parquet

    # Importancia de features
    a = pd.Series(model.featureImportances.toArray(), index=feature_cols)
    imp_df = a.sort_values(ascending=False).reset_index()
    imp_df.columns = ["Feature","Importance"]
    plt.figure(figsize=(8,4)); sns.barplot(data=imp_df,x="Importance",y="Feature")
    plt.title("Importancia de Características"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"feat_importance.png"))
    plt.close()

    # Confusion Matrix
    y_true = preds.select(label).toPandas()
    y_pred = preds.select("prediction").toPandas()
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues"); plt.title("Matriz de Confusión")
    plt.savefig(os.path.join(FIG_DIR,"conf_matrix.png")); plt.close()

    # ROC
    y_score = preds.select("probability").rdd.map(lambda r: float(r[0][1])).collect()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5,5)); plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.2f}")
    plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend()
    plt.savefig(os.path.join(FIG_DIR,"roc_curve.png")); plt.close()

    print("📊 Figuras guardadas en", FIG_DIR)

except Exception as e:
    print("❌ Error modelado Silver:", e)

finally:
    try:
        spark.stop()
    except Exception:
        pass
    
    print("🛑 Spark detenido y temporales eliminados.")