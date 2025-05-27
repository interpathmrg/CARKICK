
# 🚀 Conversor CSV a Parquet con Apache Spark y MinIO

Este proyecto convierte archivos CSV en formato Parquet utilizando Apache Spark y los carga automáticamente en un bucket S3 compatible (MinIO).

## 🧰 Tecnologías utilizadas

- Apache Spark
- Pandas
- Boto3
- MinIO (S3 compatible)
- Python 3.8+
- Docker + docker-compose

## 📁 Estructura esperada

```
data/
├── csv/
│   └── archivo.csv
├── parquet/
│   └── archivo.parquet
.env
convert_with_spark.py
```

## ⚙️ Configuración

1. Crea un archivo `.env` con tus credenciales S3:

```
AWS_ACCESS_KEY_ID=ROOTUSER
AWS_SECRET_ACCESS_KEY= ********
```

2. Asegúrate de que MinIO esté corriendo y el bucket `datasets` exista.

3. Ubica los archivos `.csv` en el directorio `data/csv/`.

## ▶️ Ejecución

```bash
python3 convert_with_spark.py
```

El script:

- Detecta si puede usar el clúster de Spark o cae en modo local.
- Valida cada archivo CSV.
- Convierte el archivo a DataFrame de Spark.
- Lo guarda como `.parquet` en el directorio `data/parquet`.
- Sube automáticamente el archivo Parquet a MinIO.

---

# 🚀 CSV to Parquet Converter with Apache Spark and MinIO

This project converts CSV files to Parquet using Apache Spark and uploads them to an S3-compatible bucket (MinIO).

## 🧰 Tech Stack

- Apache Spark
- Pandas
- Boto3
- MinIO (S3 compatible)
- Python 3.8+
- Docker + docker-compose

## 📁 Expected Structure

```
data/
├── csv/
│   └── file.csv
├── parquet/
│   └── file.parquet
.env
convert_with_spark.py
```

## ⚙️ Setup

1. Create a `.env` file with your S3 credentials:

```
AWS_ACCESS_KEY_ID=ROOTUSER
AWS_SECRET_ACCESS_KEY=AyTCg5GNoXBFiM
```

2. Ensure MinIO is running and the `datasets` bucket exists.

3. Place your `.csv` files under `data/csv/`.

## ▶️ Run

```bash
python3 convert_with_spark.py
```

The script:

- Automatically detects and connects to a Spark cluster (or falls back to local mode).
- Validates each CSV file.
- Converts it to a Spark DataFrame.
- Saves it as `.parquet` locally.
- Uploads the resulting file to MinIO.
