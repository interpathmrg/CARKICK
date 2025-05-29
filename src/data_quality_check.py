# ================= data_quality_check_silver.py =================
"""
Analiza *training.silver.parquet* y genera:
• CSV con # valores únicos + ejemplos.
• Exploratory Data Analysis (EDA) automático:
   – Hist/Box para numéricas.
   – Count‑plot para categóricas.
   – Pairplot (scatter‑matrix) para numéricas principales.
Los gráficos se guardan en la carpeta ./eda_report/.
Se descarga el parquet vía boto3 → disco local (sin S3A).
"""

from dotenv import load_dotenv
import os, json, shutil, tempfile, pathlib, boto3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql.functions import col, countDistinct
from utils_spark import iniciar_spark

# ────────────────────────────
# Configuración & credenciales
# ────────────────────────────
load_dotenv()
PARQUET = "training.silver.parquet"
BUCKET  = "datasets"
ENDPT   = "http://localhost:9100"
AK      = os.getenv("AWS_ACCESS_KEY_ID")
SK      = os.getenv("AWS_SECRET_ACCESS_KEY")
OUT_CSV = "valores_unicos_silver.csv"
EDA_DIR = pathlib.Path("eda_report")
EDA_DIR.mkdir(exist_ok=True)

if not AK or not SK:
    raise EnvironmentError("❌ Falta AWS_ACCESS_KEY_ID / SECRET_KEY en .env")

tmp = tempfile.mkdtemp(prefix="silver_qc_")
local_pq = os.path.join(tmp, PARQUET)

try:
    # 1️⃣ Descarga parquet
    s3 = boto3.client("s3", endpoint_url=ENDPT, aws_access_key_id=AK, aws_secret_access_key=SK)
    s3.download_file(BUCKET, PARQUET, local_pq)
    print("✅ Parquet descargado →", local_pq)

    # 2️⃣ Spark → leer
    spark, _ = iniciar_spark(app_name="dq_eda_local")
    df = spark.read.parquet(local_pq)

    # 3️⃣ CSV de valores únicos
    print("🔍 Generando resumen de unicidad …")
    summary = []
    for c in df.columns:
        n_uni = df.select(countDistinct(col(c))).collect()[0][0]
        ejemplos = df.select(c).distinct().limit(20).rdd.flatMap(lambda x: x).collect()
        summary.append((c, n_uni, json.dumps(ejemplos, default=str)))
    spark.createDataFrame(summary, ["Columna","Valores Únicos","Ejemplos Únicos"])\
        .coalesce(1).write.option("header",True).option("sep",";").mode("overwrite").csv(OUT_CSV)
    print("✅ CSV →", OUT_CSV)

    # 4️⃣ Convertir a Pandas para EDA (72 k filas ≈ ok)
    pdf = df.sample(fraction=1.0).toPandas()

    # Identificar dtypes
    num_cols  = pdf.select_dtypes(include=["number"]).columns.tolist()
    cat_cols  = [c for c in pdf.columns if c not in num_cols]

    sns.set(style="whitegrid", palette="pastel")

    # 4.a Hist & box para numéricas
    for colname in num_cols:
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        sns.histplot(pdf[colname].dropna(), kde=True, ax=ax[0]); ax[0].set_title(f"Hist {colname}")
        sns.boxplot(x=pdf[colname], ax=ax[1]); ax[1].set_title(f"Box {colname}")
        plt.tight_layout(); plt.savefig(EDA_DIR / f"{colname}_dist.png"); plt.close()

    # 4.b Count‑plot categóricas (máx 20 categorías)
    for colname in cat_cols:
        if pdf[colname].nunique() <= 20:
            plt.figure(figsize=(8,4))
            sns.countplot(x=pdf[colname]); plt.xticks(rotation=45)
            plt.title(f"Count {colname}"); plt.tight_layout();
            plt.savefig(EDA_DIR / f"{colname}_count.png"); plt.close()

    # 4.c Pairplot principales (hasta 6 num)
    top_nums = num_cols[:6]
    if len(top_nums) >= 2:
        sns.pairplot(pdf[top_nums]); plt.tight_layout();
        plt.savefig(EDA_DIR / "pairplot_numeric.png"); plt.close()

    print("✅ Gráficos EDA guardados en ./eda_report/")

except Exception as e:
    print("❌ Error en DQ / EDA:", e)

finally:
    try:
        spark.stop()
    except Exception:
        pass
    shutil.rmtree(tmp, ignore_errors=True)
    print("🛑 Spark detenido y temporales eliminados.")

