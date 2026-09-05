import pyspark.sql.functions as F
import dlt
from pyspark.sql.types import *

catalog_name = spark.conf.get('catalog_name')
volume_path = f"/Volumes/{catalog_name}/bronze/earthquake_data"
primary_key = "id"
properties_schema = StructType(
    [
        StructField("mag", StringType()),
        StructField("place", StringType()),
        StructField("time", StringType()),
        StructField("status", StringType()),
        StructField("tsunami", StringType()),
        StructField("type", StringType()),
        StructField("url", StringType()),
        StructField("detail", StringType()),
        StructField("felt", StringType()),
        StructField("cdi", StringType()),
        StructField("mmi", StringType()),
        StructField("alert", StringType()),
        StructField("sig", StringType()),
        StructField("net", StringType()),
        StructField("code", StringType()),
        StructField("ids", StringType()),
        StructField("sources", StringType()),
        StructField("types", StringType()),
        StructField("nst", StringType()),
        StructField("dmin", StringType()),
        StructField("rms", StringType()),
        StructField("gap", StringType()),
        StructField("magType", StringType()),
        StructField("title", StringType()),
    ]
)

geometry_schema = StructType([StructField("coordinates", ArrayType(DoubleType()))])

feature_schema = StructType(
    [
        StructField("id", StringType()),
        StructField("properties", properties_schema),
        StructField("geometry", geometry_schema),
    ]
)

schema = ArrayType(feature_schema)


@dlt.view(name="earthquake_data_vw")
def earthquake_data():
    df = (
        spark.readStream.format("cloudfiles")
        .option("cloudfiles.format", "json")
        .load(volume_path)
        .withColumn("load_timestamp", F.current_timestamp())
    )
    df = df.withColumn("parsed_data", F.from_json(F.col("features"), schema))
    df = df.select(
        F.explode(F.col("parsed_data")).alias("features"),
        F.col("load_timestamp")
    )
    df = df.select(
        "features.properties.*",
        "features.id",
        F.col("features.geometry.coordinates")[0].alias("longitude"),
        F.col("features.geometry.coordinates")[1].alias("latitude"),
        F.col("features.geometry.coordinates")[2].alias("depth"),
        F.col("load_timestamp")
    )
    df = (
        df.withColumn("time", F.from_unixtime(F.col("time") / 1000).cast("timestamp"))
        .withColumn("mag", F.col("mag").cast("double"))
        .withColumn("nst", F.col("nst").cast("double"))
        .withColumn("sig", F.col("sig").cast("double"))
        .withColumn("tsunami", F.col("tsunami").cast("double"))
        .withColumn("felt", F.col("felt").cast("double"))
    )
    return df

dlt.create_streaming_table(name="earthquake_data_final")

dlt.apply_changes(
    target="earthquake_data_final",
    source="earthquake_data_vw",
    keys=[primary_key],
    sequence_by="load_timestamp",
    stored_as_scd_type= '1'
)