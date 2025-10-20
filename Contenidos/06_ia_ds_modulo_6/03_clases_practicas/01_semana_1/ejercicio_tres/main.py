from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, length, regexp_replace, upper, trim
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import requests

spark = SparkSession.builder.appName("Veracidad_Datos").getOrCreate()

# Proceso de extracción de datos dentro del ETL

print("=== SISTEMA DE VERACIDAD DE DATOS ===")
print("Garantozando datos confiables, precisos y sin errores")
print("\n*** DESCARGANDO EL DATASET DESDE INTERNET ***")

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

# Proceso de transformación de datos dentro del ETL

print("\n*** 1. DETECCIÓN DE DATOS CORRUPTOS ***")

# Validamos que el archivo .csv tiene header
df = spark.read.option("header", True).csv(archivo)

# Imprimir la información inicial de nuestro dataset
print("ESTADO INICIAL DEL DATASET")
print(f"Total de registros: {df.count()}")
print(f"Total de columnas: {len(df.columns)}") # 13 columnas

# Detectar calores faltantes: isNull, NULL ó string vacios
print("\nVALORES FALTANTES POR COLUMNA:")
for col_name in df.columns:
    missing_count = df.filter(col(col_name).isNull()).count()
    if missing_count > 0:
        print(f" {col_name}: {missing_count} valores faltantes")

# Valores duplicados en todas las columnas
duplicates = df.groupBy(df.columns).count().filter(col("count") > 1)
duplicate_count = duplicates.count()
print(f"\nDUPLICADOS ENCONTRADOS {duplicate_count}")

if duplicate_count > 0:
    print("Ejemplos de duplicados:")
    duplicates.show(5, truncate=False)

# El proceso de transformación => Limpieza y Validación

print("\n*** 2. limpieza y validación de datos ***")
df_clean = df.na.drop(subset=["ident", "name"])

df_clean = df_clean.filter(length(col("iso_country")) == 2)

df_clean = df_clean.withColumn("name_clean", trim(upper(col("name"))))

df_clean = df_clean.filter(col("coordinates").contains(","))

print(f"Registros luego de la limpieza: {df_clean.count()}")

print("\n*** 3. CONTROL DE CALIDAD ***")
validation_results = []

# Regla 1: Registros limpios; Deben tener "ident" no nulo
ident_count = df_clean.filter(col("ident").isNotNull()).count()
total_count =df_clean.count()
validation_results.append(("Ident único", ident_count == total_count, f"{ident_count}/{total_count}"))

# Regla 2. Código del país valido; Longitud = 2
valid_countries = df_clean.filter(length(col("iso_country")) == 2).count()
validation_results.append(("Código de país valido", valid_countries == total_count, f"{valid_countries}/{total_count}"))

# Regla 3: 'type' debe estar dentro de los ipos esperados
valid_types = ["large_airport", "medium_airport", "small_airport", "heliport", "seaplane_base"]
valid_type_count = df_clean.filter(col("type").isin(valid_types)).count()
validation_results.append(("Tipos validados", valid_type_count == total_count, f"{valid_type_count}/{total_count}"))

# Imprimir los resultados
print("RESULTADOS DE VALIDACIÓN:")
for rule, passed, details in validation_results:
    status = "PASÓ" if passed else  "FALLÓ"
    print(f" {rule}: {status} ({details})")

print("\n*** 4. DETECCIÓN DE ANOMALIAS ***")

sin_coordenadas = df_clean.filter(col("coordinates").isNull() | (col("coordinates") == "")).count()
print(f"Aeropuertos sin coordenadas: {sin_coordenadas}")

nombres_cortos = df_clean.filter(length(col("name")) < 3).count()
print(f"Nombre muy cortos (<3 chars): {nombres_cortos}")

tipos_raros = df_clean.filter(~col("type").isin(valid_types)).count()
print(f"Tipos de aeropuertos desconocidos: {tipos_raros}")

print("\n*** REPORTE FINAL DE CALIDAD ***")

quality_score = sum(1 for _, passed, _ in validation_results if passed) / len(validation_results) * 100
print(f"PUNTAJE DE CALIDAD: {quality_score:.1f}%")

if quality_score >= 90:
    print("EXCELENTE: Los datos son altamente confiables")
elif quality_score >= 70:
    print("BUENO: Los datos son confiables con algunas excepciones")
else:
    print("ADVERTENCIA: Los datos requieren revisión manual")

print(f"RESUMEN FINAL:")
print(f"    - Registros originales: {df.count()}")
print(f"    - Registros después de la limpieza: {df_clean.count()}")
print(f"    - Registros perdidos: {df.count() - df_clean.count()}")
print(f"    - Porcentaje de retención: {(df_clean.count() / df.count() * 100):.1f}%")

print("\n*** MUESTRA DE DATOS VALIDADOS ***")
df_clean.select("ident", "name_clean", "iso_country", "type", "coordinates").show(5, truncate=False)

spark.stop()
