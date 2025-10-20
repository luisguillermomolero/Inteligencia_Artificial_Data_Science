from pyspark.sql import SparkSession
import requests
import warnings

warnings.filterwarnings('ignore')

spark = SparkSession.builder \
    .appName("BigData_Volumen") \
    .config("spark.sql.debug.maxToStringFields", "100") \
    .config("spark.sql.adaptive.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
local_path = "yellow_tripdata_2024-01.parquet"

try:
    print("Descargando dataset...")
    response = requests.get(url, timeout=30)
    with open(local_path, "wb") as f:
        f.write(response.content)
    print("Descarga completada")
except Exception as e:
    print(f"Error en descarga: {e}")

df = spark.read.parquet(local_path)

print("INFORMACIÓN DEL DATASET NYC TAXI")

total_records = df.count()
print(f"\nTOTAL DE REGISTROS: {total_records:,}")

print("\nPRIMERAS 3 FILAS:")
df.select("VendorID", "tpep_pickup_datetime", "fare_amount", "tip_amount", "total_amount") \
  .show(3, truncate=False)

print("\nESQUEMA DEL DATASET:")
schema_dict = {}
for field in df.schema.fields:
    schema_dict[field.name] = str(field.dataType)

for i, (field_name, field_type) in enumerate(schema_dict.items(), 1):
    print(f"  {i:2d}. {field_name:<25} → {field_type}")

print("\nANÁLISIS ESTADÍSTICO BÁSICO")

numeric_cols = ['fare_amount', 'tip_amount', 'total_amount', 'trip_distance']

for col in numeric_cols:
    if col in df.columns:
        stats = df.select(col).summary("count", "mean", "stddev", "min", "max").collect()
        if stats:
            count = stats[0][col]
            mean = stats[1][col]
            std = stats[2][col]
            min_val = stats[3][col]
            max_val = stats[4][col]
            
            print(f"\n  {col.upper()}:")
            print(f"    • Total: {count}")
            try:
                print(f"    • Promedio: ${float(mean):.2f}")
            except Exception:
                print(f"    • Promedio: {mean} (no convertible a float)")
            try:
                print(f"    • Desviación: ${float(std):.2f}")
            except Exception:
                print(f"    • Desviación: {std} (no convertible a float)")
            try:
                print(f"    • Rango: ${float(min_val):.2f} - ${float(max_val):.2f}")
            except Exception:
                print(f"    • Rango: {min_val} - {max_val}")

spark.stop()
