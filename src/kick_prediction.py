# ================= kick_prediction_silver.py =================
"""
Entrena y evalúa un modelo RandomForest para predecir `IsBadBuy`
utilizando la **capa Silver (training.silver.parquet)** y evita S3A
leyendo vía boto3 ➜ disco local.
Variables usadas:
  • VehicleAge, VehOdo, VehBCost, IsOnlineSale (numéricas)
  • Transmission_idx, Size_idx (ya numéricas)
"""

from dotenv import load_dotenv
import os
import boto3
import tempfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from utils_spark import iniciar_spark
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ────────────────────────────
# Configuración & credenciales
# ────────────────────────────
load_dotenv()
PARQUET_FILENAME = "training.silver.parquet"
BUCKET_NAME      = "datasets"
MINIO_URL        = "http://localhost:9100"
ACCESS_KEY       = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY       = os.getenv("AWS_SECRET_ACCESS_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY en .env")

# Temporal
tmp_dir   = tempfile.mkdtemp(prefix="silver_model_")
local_parquet = os.path.join(tmp_dir, PARQUET_FILENAME)

try:
    # 1️⃣ Descargar parquet desde MinIO
    print("☁️ Descargando Silver …")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    s3.download_file(BUCKET_NAME, PARQUET_FILENAME, local_parquet)
    print(f"✅ Parquet en {local_parquet}")

    # 2️⃣ Spark local
    spark, _ = iniciar_spark(app_name="kick_prediction_silver_local")
    df = spark.read.parquet(local_parquet)

    # 3️⃣ Selección y casting
    label = "IsBadBuy"
    feature_cols = [
        "VehicleAge", "VehOdo", "VehBCost", "IsOnlineSale",
        "Transmission_idx", "Size_idx"
    ]
    df = df.select([label] + feature_cols).dropna()
    df = df.withColumn(label, col(label).cast("double"))

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    rf = RandomForestClassifier(labelCol=label, featuresCol="features", numTrees=200, seed=42, maxBins=64)

    # 4️⃣ Pipeline manual
    data = assembler.transform(df).select("features", label)
    train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
    model = rf.fit(train_df)
    preds = model.transform(test_df)

    # 5️⃣ Métricas
    eval_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
    eval_f1  = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    eval_acc = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")

    print("📈 Resultados:")
    print(f"🔹 AUC:       {eval_auc.evaluate(preds):.4f}")
    print(f"🔹 F1-score:  {eval_f1.evaluate(preds):.4f}")
    print(f"🔹 Accuracy:  {eval_acc.evaluate(preds):.4f}")

    # 6️⃣ Gráficos
    importances = model.featureImportances.toArray()
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(data=imp_df, x="Importance", y="Feature")
    plt.title("Importancia de Características (Silver)")
    plt.tight_layout(); plt.show()

    y_true = preds.select(label).toPandas()
    y_pred = preds.select("prediction").toPandas()
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues"); plt.title("Matriz de Confusión"); plt.show()

    y_score = preds.select("probability").rdd.map(lambda r: r[0][1]).collect()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend(); plt.show()

except Exception as e:
    print(f"❌ Error modelado Silver: {e}")

finally:
    try:
        spark.stop()
    except Exception:
        pass
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("🛑 Spark detenido y temporales limpiados.")