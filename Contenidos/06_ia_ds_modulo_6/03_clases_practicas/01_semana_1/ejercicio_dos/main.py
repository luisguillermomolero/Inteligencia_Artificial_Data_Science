from pyspark.sql import SparkSession
import requests
import os

spark = SparkSession.builder.appName("Streaming_class").getOrCreate()

print("Descargando dataset desde Internet...")

airports_url = "https://raw.githubusercontent.com/datasets/airport-codes/master/data/airport-codes.csv"
airports_file = "airport_from_internet.csv"

print(f"Descargando dataset CVS Aeropuertos desde: {airports_url}")

try:
    response = requests.get(airports_url, timeout=30)
    with open(airports_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Dataset de aeropuertos descargado exitosamente...")
except Exception as e:
    print(f"ERROR al intentar descargar el dataset de aeropuertos: {e}")

countries_url = "https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv"
countries_file = "country_from_internet.csv"

print(f"Descargando CSV de paises desde: {countries_url}")

try:
    response = requests.get(countries_url, timeout=30)
    with open(countries_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Dataset de paises descargado exitosamente...")
except Exception as e:
    print(f"ERROR al intentar descargar el dataset de paises: {e}")

print(F"Leer dataset descargados")

df_airports = spark.read.option("header", True).csv(airports_file)
df_country = spark.read.option("header", True).csv(countries_file)

print("Esquema del CSV aeropuertos")
df_airports.printSchema()

print("Esquema del CSV de paises")
df_country.printSchema()

print("Muestra de datos del CSV de aeropuertos")
df_airports.select("ident", "name", "iso_country", "type").show(5, truncate=False)

print("Muestra de datos cdl CSV de paises")
df_country.select("ISO3166-1-Alpha-2", "official_name_en", "Continent").show(5, truncate=False)

spark.stop()