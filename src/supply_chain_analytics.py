from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("Supply Chain Analytics PySpark Pipeline") \
    .getOrCreate()

# Read source file
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("data/supply_chain_data.csv")

# Bronze layer
bronze_df = df.withColumn("ingestion_timestamp", current_timestamp())

# Silver layer - cleaning and transformations
silver_df = bronze_df \
    .dropDuplicates(["SKU"]) \
    .filter(col("SKU").isNotNull()) \
    .filter(col("Revenue generated") > 0) \
    .withColumn("profit", round(col("Revenue generated") - col("Costs"), 2)) \
    .withColumn("profit_margin_percentage",
                round((col("profit") / col("Revenue generated")) * 100, 2)) \
    .withColumn("stock_status",
                when(col("Stock levels") < 30, "Low Stock")
                .when(col("Stock levels") <= 70, "Medium Stock")
                .otherwise("High Stock")) \
    .withColumn("defect_status",
                when(col("Defect rates") > 3, "High Defect")
                .when(col("Defect rates") >= 1, "Medium Defect")
                .otherwise("Low Defect")) \
    .withColumn("shipping_cost_category",
                when(col("Shipping costs") > 7, "High Cost")
                .when(col("Shipping costs") >= 4, "Medium Cost")
                .otherwise("Low Cost"))

print("Silver Layer Data")
silver_df.show(10, truncate=False)

# Gold 1: Product type summary
product_summary_df = silver_df.groupBy("Product type") \
    .agg(
        count("SKU").alias("total_skus"),
        sum("Number of products sold").alias("total_products_sold"),
        round(sum("Revenue generated"), 2).alias("total_revenue"),
        round(sum("profit"), 2).alias("total_profit"),
        round(avg("Defect rates"), 2).alias("avg_defect_rate")
    ) \
    .orderBy(col("total_revenue").desc())

print("Product Type Summary")
product_summary_df.show(truncate=False)

# Gold 2: Supplier performance
supplier_summary_df = silver_df.groupBy("Supplier name") \
    .agg(
        count("SKU").alias("total_skus"),
        round(sum("Revenue generated"), 2).alias("total_revenue"),
        round(avg("Lead time"), 2).alias("avg_lead_time"),
        round(avg("Manufacturing costs"), 2).alias("avg_manufacturing_cost"),
        round(avg("Defect rates"), 2).alias("avg_defect_rate")
    ) \
    .orderBy(col("avg_defect_rate").desc())

print("Supplier Performance Summary")
supplier_summary_df.show(truncate=False)

# Gold 3: Location wise revenue
location_summary_df = silver_df.groupBy("Location") \
    .agg(
        round(sum("Revenue generated"), 2).alias("location_revenue"),
        round(sum("profit"), 2).alias("location_profit"),
        round(avg("Shipping costs"), 2).alias("avg_shipping_cost"),
        round(avg("Shipping times"), 2).alias("avg_shipping_time")
    ) \
    .orderBy(col("location_revenue").desc())

print("Location Wise Summary")
location_summary_df.show(truncate=False)

# Window function: Top revenue SKU by location
window_spec = Window.partitionBy("Location").orderBy(col("Revenue generated").desc())

top_sku_by_location_df = silver_df.withColumn(
    "revenue_rank",
    dense_rank().over(window_spec)
).filter(col("revenue_rank") == 1)

print("Top Revenue SKU by Location")
top_sku_by_location_df.select(
    "Location",
    "SKU",
    "Product type",
    "Revenue generated",
    "profit",
    "revenue_rank"
).show(truncate=False)

# Write outputs
product_summary_df.write.mode("overwrite").parquet("output/gold/product_summary")
supplier_summary_df.write.mode("overwrite").parquet("output/gold/supplier_summary")
location_summary_df.write.mode("overwrite").parquet("output/gold/location_summary")
top_sku_by_location_df.write.mode("overwrite").parquet("output/gold/top_sku_by_location")

print("Supply Chain Analytics PySpark Pipeline Completed Successfully")

spark.stop()
