from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import boto3

# Configuración
CSV_DIR = '/home/mrgonzalez/Desktop/PYTHON/CARKICK/data/csv'
PARQUET_DIR = '/home/mrgonzalez/Desktop/PYHTON/CARKICK/data/parquet'
BUCKET_NAME = 'datasets'

# MinIO client config (S3 compatible)
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9100',    
)

# Crear carpeta local si no existe
os.makedirs(PARQUET_DIR, exist_ok=True)

# Procesar cada CSV
for filename in os.listdir(CSV_DIR):
    if filename.endswith('.csv'):
        csv_path = os.path.join(CSV_DIR, filename)
        parquet_filename = filename.replace('.csv', '.parquet')
        parquet_path = os.path.join(PARQUET_DIR, parquet_filename)
        try:
            print(f"📦 Procesando: {csv_path}")
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path, engine='pyarrow', index=False, compression='snappy')
            # Subida directa a MinIO
            print(f"☁️ Subiendo a MinIO: {parquet_filename}")
            with open(parquet_path, 'rb') as data:
                s3.upload_fileobj(data, BUCKET_NAME, parquet_filename)
            print(f"✅ Subido: {parquet_filename}")
        except Exception as e:
                print(f"❌ Error con {filename}: {e}")

print("✅ Conversión y subida completadas.")
