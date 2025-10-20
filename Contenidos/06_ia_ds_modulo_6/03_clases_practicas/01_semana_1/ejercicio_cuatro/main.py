from pyspark.sql import SparkSession
from pyspark.sql.functions import col,count, desc, when
import requests

spark = SparkSession.builder.appName("Valor_Datos").getOrCreate()

print("*** EXTRACCIÓN DE VALOR A PARTIR DE LOS DATOS ***")
print("Transformando datos en conocimiento útil")

url = "https://raw.githubusercontent.com/datasets/airport-codes/master/data/airport-codes.csv"
archivo = "aeropuertos.csv"

print(f"Descargando dataset desde: {url}")

try:
    response = requests.get(url, timeout=30)
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Dataset descargado exitosamente")
except Exception as e:
    print(f"Error descargando: {e}")

# Validamos que el archivo .csv tiene header
df = spark.read.option("header", True).csv(archivo)
print(f"Total de registros: {df.count()}")

print("\n*** 1. TOP 5 PAÍSES CON MÁS AEROPUERTOS ***")
top_countries = (
    df.groupBy("iso_country")
    .agg(count("*").alias("total"))
    .orderBy(desc("total"))
    .limit(5)
)
top_countries.show(truncate=False)

print("\n*** 2. DISTRIBUCIÓN POR TIPO ***")
market_segments = (
    df.groupBy("type")
    .agg(count("*").alias("cantidad"))
    .orderBy(desc("cantidad"))
)
market_segments.show(truncate=False)

print("\n*** 3. PAISES CON AEROPUERTOS PEQUEÑOS ***")
opportunities = (
    df.groupBy("iso_country")
    .agg(
        count("*").alias("total"),
        count(when(col("type") == "small_airport", True)).alias("pequeños"),
        count(when(col("type") == "large_airport", True)).alias("grandes"))
    .filter((col("grandes") == 0) & (col("total") > 10))
    .orderBy(desc("total"))
    .limit(5)
)
opportunities.show(truncate=False)

print("\n*** REPORTE FINAL ***")
total_airports = df.count()
large_airports = df.filter(col("type") == "large_airport").count()
small_airports = df.filter(col("type") == "small_airport").count()

print("RESUMEN:")
print(f"    - Aeropuertos grandes: {large_airports} ({large_airports/total_airports*100:.1f}%)")
print(f"    - Aeropuertos pequeños: {small_airports} ({small_airports/total_airports*100:.1f}%)")

strategic_value = (large_airports * 0.4 + small_airports * 0.1) / total_airports * 100
print(f"VALOR ESTRATÉGICO: {strategic_value:.1f}%")

if strategic_value >= 20:
    print("BUENO: Oportunidades claras identificadas")
else:
    print("MODERADO: Require análisis adicional")
    
print("\nAPLICACIONES:")
print("     - Identificar mercados objetivo")
print("     - Oportunidades de inversión")
print("     - Estratégias de expansión")

spark.stop()
