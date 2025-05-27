
Conectar con Minio y crear un sataset si no existe

mc alias set local http://localhost:9100 ROOTUSER ******

Crea un nuevo bucket
mc mb local/datasets  # solo si el bucket no existe

Permisos para correr el script de spark
sudo chown -R mrgonzalez:mrgonzalez /home/mrgonzalez/Desktop/PYTHON/CARKICK/data/parquet
chmod -R u+rwX /home/mrgonzalez/Desktop/PYTHON/CARKICK/data/parquet