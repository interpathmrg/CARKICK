from pyspark.sql import SparkSession

DRIVER_HOST = "172.17.0.1"   # gateway docker0
DRIVER_PORT = "7078"
BLOCKM_PORT = "7079"

spark = (
    SparkSession.builder
      .appName("carkick")
      .master("spark://localhost:7077")
      # ─ Driver (host) ─
      .config("spark.driver.bindAddress", "0.0.0.0")
      .config("spark.driver.host", DRIVER_HOST)
      .config("spark.driver.port", DRIVER_PORT)
      .config("spark.blockManager.port", BLOCKM_PORT)

      .getOrCreate()
)


spark.sparkContext.setLogLevel("ERROR")
print("Spark iniciado, Driver anunciado en", DRIVER_HOST)
print("Contando…", spark.range(1, 1_000_000).count())
