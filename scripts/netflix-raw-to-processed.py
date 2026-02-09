"""
Netflix Content Pipeline - Bronze to Silver Layer Transformation
==================================================================
Business Context:
    Transforms raw Netflix catalog data into analytics-ready format for:
    - Content performance dashboards
    - Acquisition strategy analysis
    - Regional content planning
    - Viewer targeting and recommendations

Technical Approach:
    - Medallion Architecture: Bronze (raw) → Silver (cleansed & enriched)
    - Data Quality: Validation, rejection handling, quality scoring
    - Performance: Partitioned Parquet with Snappy compression
    
Author: Mohamed Khasim
Created: 02-04-2026
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime


# ============================================================================
# JOB INITIALIZATION & CONFIGURATION
# ============================================================================

def initialize_job():
    """Initialize Glue context with graceful parameter handling"""
    try:
        args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_BUCKET'])
    except Exception:
        # Fallback for local testing
        args = getResolvedOptions(sys.argv, ['JOB_NAME'])
        args['S3_BUCKET'] = 'netflix-pipeline-khasim-2026'
    
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)
    
    return spark, job, args


def log_section(title, char="="):
    """Consistent logging format for better monitoring"""
    print(f"\n{char * 80}")
    print(f"{title.upper()}")
    print(f"{char * 80}")


# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

def get_netflix_schema():
    """
    Explicit schema prevents inferSchema issues and ensures type safety.
    
    Business Note: show_id is non-nullable as it's our primary key.
    All other fields nullable to handle real-world data quality issues.
    """
    return StructType([
        StructField("show_id", StringType(), nullable=False),
        StructField("type", StringType(), nullable=True),
        StructField("title", StringType(), nullable=True),
        StructField("director", StringType(), nullable=True),
        StructField("cast", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("date_added", StringType(), nullable=True),
        StructField("release_year", IntegerType(), nullable=True),
        StructField("rating", StringType(), nullable=True),
        StructField("duration", StringType(), nullable=True),
        StructField("listed_in", StringType(), nullable=True),
        StructField("description", StringType(), nullable=True)
    ])


# ============================================================================
# DATA VALIDATION & QUALITY
# ============================================================================

def validate_and_separate_records(df):
    """
    Separate valid records from rejected records with clear audit trail.
    
    Business Rule: Records must have show_id (PK) and title (user-facing field).
    Invalid records are logged separately for data quality investigation.
    """
    log_section("Data Validation", "-")
    
    # Valid records criteria
    df_valid = df.filter(
        col("show_id").isNotNull() & 
        (trim(col("show_id")) != "") &
        col("title").isNotNull() & 
        (trim(col("title")) != "")
    )
    
    # Capture rejected records with reason
    df_rejected = df.subtract(df_valid) \
        .withColumn("rejection_reason", lit("Missing required fields: show_id or title")) \
        .withColumn("rejected_at", current_timestamp())
    
    valid_count = df_valid.count()
    rejected_count = df_rejected.count()
    
    print(f"✓ Valid records: {valid_count:,}")
    print(f"✗ Rejected records: {rejected_count:,}")
    
    if rejected_count > 0:
        print(f"⚠ Warning: {rejected_count} records failed validation")
    
    return df_valid, df_rejected


def apply_data_quality_rules(df):
    """
    Apply business-driven data quality transformations.
    
    Each transformation serves a specific analytics need:
    - Standardized nulls enable consistent filtering
    - Parsed dates enable time-series analysis
    - Duration parsing enables content length segmentation
    """
    log_section("Data Quality Enhancement", "-")
    
    df_clean = df
    
    # =================================================================
    # 1. NULL STANDARDIZATION
    # Business Need: Consistent handling for downstream analytics
    # =================================================================
    null_mappings = {
        "director": "Unknown",
        "cast_and_crew": "Not Available",
        "country": "Unknown",
        "rating": "UNRATED",
        "genre": "Uncategorized",
        "description": "No description available"
    }
    
    for column, default_value in null_mappings.items():
        df_clean = df_clean.withColumn(
            column,
            when(col(column).isNull() | (trim(col(column)) == ""), default_value)
            .otherwise(trim(col(column)))
        )
    
    print(f"✓ Standardized null values across {len(null_mappings)} columns")
    
    # =================================================================
    # 2. DATE PARSING
    # Business Need: Enable trend analysis and content freshness metrics
    # =================================================================
    df_clean = df_clean.withColumn(
        "date_added",
        coalesce(
            to_date(trim(col('date_added')), "MMMM d, yyyy"),
            to_date(trim(col('date_added')), "MMMM dd, yyyy"),
            lit(None).cast(DateType())
        )
    )
    print("✓ Parsed date_added with multiple format support")
    
    # =================================================================
    # 3. DURATION PARSING
    # Business Need: Content length analysis for UX optimization
    # =================================================================
    df_clean = df_clean \
        .withColumn("duration_value",
            when(col("duration").isNotNull() & (trim(col("duration")) != ""),
                regexp_extract(col("duration"), r"(\d+)", 1).cast("integer")
            ).otherwise(lit(None).cast("integer"))
        ) \
        .withColumn("duration_unit",
            when(lower(col("duration")).contains("min"), "minutes")
            .when(lower(col("duration")).contains("season"), "seasons")
            .otherwise("unknown")
        )
    print("✓ Extracted duration_value and duration_unit")
    
    # =================================================================
    # 4. DATA NORMALIZATION
    # Business Need: Consistent formatting for matching and grouping
    # =================================================================
    df_clean = df_clean \
        .withColumn("rating", upper(trim(col("rating")))) \
        .withColumn("content_type", trim(col("content_type")))
    
    print("✓ Normalized text fields for consistency")
    
    return df_clean


def enrich_with_business_features(df):
    """
    Add derived features that directly support business analytics.
    
    Each feature has a clear business use case - no feature bloat.
    """
    log_section("Business Feature Engineering", "-")
    
    df_enriched = df
    
    # =================================================================
    # TEMPORAL FEATURES - For trend analysis and seasonality
    # =================================================================
    df_enriched = df_enriched \
        .withColumn("added_year", year(col("date_added"))) \
        .withColumn("added_month", month(col("date_added"))) \
        .withColumn("content_age_years",
            when(col("release_year").isNotNull(), lit(2026) - col("release_year"))
            .otherwise(lit(None))
        )
    
    print("✓ Created temporal features: added_year, added_month, content_age_years")
    
    # =================================================================
    # MULTI-VALUE FIELD PARSING - Primary category extraction
    # =================================================================
    df_enriched = df_enriched \
        .withColumn("primary_genre",
            when(col("genre") != "Uncategorized",
                trim(split(col("genre"), ",")[0])
            ).otherwise("Uncategorized")
        ) \
        .withColumn("primary_country",
            when(col("country") != "Unknown",
                trim(split(col("country"), ",")[0])
            ).otherwise("Unknown")
        )
    
    print("✓ Extracted primary_genre and primary_country for simplified analysis")
    
    # =================================================================
    # DATA COMPLETENESS FLAGS - For quality segmentation
    # =================================================================
    df_enriched = df_enriched \
        .withColumn("has_director", col("director") != "Unknown") \
        .withColumn("has_cast", col("cast_and_crew") != "Not Available") \
        .withColumn("is_recent",
            when(col("content_age_years") <= 5, True).otherwise(False)
        )
    
    print("✓ Added quality flags: has_director, has_cast, is_recent")
    
    # =================================================================
    # DATA QUALITY SCORE - For prioritization and filtering
    # =================================================================
    df_enriched = df_enriched.withColumn(
        "data_quality_score",
        (col("has_director").cast("int") +
         col("has_cast").cast("int") +
         col("duration_value").isNotNull().cast("int") +
         col("date_added").isNotNull().cast("int") +
         col("release_year").isNotNull().cast("int")) / 5.0
    )
    
    print("✓ Calculated data_quality_score (0.0 to 1.0)")
    
    return df_enriched


def add_audit_columns(df):
    """Add processing metadata for data lineage and troubleshooting"""
    return df.withColumn("processed_timestamp", current_timestamp())


# ============================================================================
# METRICS & MONITORING
# ============================================================================

def generate_quality_report(df_initial, df_final, df_rejected, duplicates_removed):
    """
    Generate comprehensive data quality report for monitoring dashboards.
    """
    log_section("Data Quality Report")
    
    initial_count = df_initial
    final_count = df_final
    rejected_count = df_rejected
    
    # Calculate success metrics
    success_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
    
    print(f"""
PROCESSING SUMMARY:
  Input Records:           {initial_count:,}
  Output Records:          {final_count:,}
  Rejected Records:        {rejected_count:,}
  Duplicates Removed:      {duplicates_removed:,}
  Success Rate:            {success_rate:.2f}%
    """)
    
    return {
        'initial_count': initial_count,
        'final_count': final_count,
        'rejected_count': rejected_count,
        'duplicates_removed': duplicates_removed,
        'success_rate': success_rate
    }


def display_sample_output(df):
    """Show sample of processed data for quick validation"""
    print("\nSAMPLE OUTPUT (First 5 Records):")
    print("-" * 80)
    df.select(
        "show_id", "title", "content_type", "primary_genre",
        "release_year", "data_quality_score"
    ).show(5, truncate=False)


# ============================================================================
# DATA PERSISTENCE
# ============================================================================

def write_to_silver_layer(df, path, partition_column="content_type"):
    """
    Write to Silver layer with optimized storage format.
    
    Partitioning Strategy:
    - By content_type (Movie/TV Show) for query performance
    - Parquet format with Snappy compression for storage efficiency
    """
    log_section("Writing to Silver Layer", "-")
    
    df.write \
        .partitionBy(partition_column) \
        .mode("overwrite") \
        .format("parquet") \
        .option("compression", "snappy") \
        .save(path)
    
    print(f"✓ Data written to: {path}")
    print(f"✓ Format: Parquet (Snappy compressed)")
    print(f"✓ Partitioned by: {partition_column}")


def write_rejected_records(df_rejected, path):
    """Write rejected records for data quality investigation"""
    if df_rejected.count() > 0:
        log_section("Writing Rejected Records", "-")
        df_rejected.write \
            .mode("overwrite") \
            .format("parquet") \
            .save(path)
        print(f"✓ Rejected records written to: {path}")
    else:
        print("\n✓ No rejected records to write")


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def main():
    """
    Main ETL orchestration - Bronze to Silver transformation.
    """
    # Initialize
    spark, job, args = initialize_job()
    
    # Configuration
    S3_BUCKET = args["S3_BUCKET"]
    RAW_PATH = f"s3://{S3_BUCKET}/raw/netflix_titles.csv"
    PROCESSED_PATH = f"s3://{S3_BUCKET}/processed/"
    REJECTED_PATH = f"s3://{S3_BUCKET}/rejected/"
    
    log_section("Netflix ETL Pipeline - Bronze to Silver")
    print(f"Source (Bronze):      {RAW_PATH}")
    print(f"Target (Silver):      {PROCESSED_PATH}")
    print(f"Rejected Records:     {REJECTED_PATH}")
    print(f"Execution Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: EXTRACT - Read from Bronze Layer
    # ========================================================================
    log_section("Step 1: Extract from Bronze Layer")
    schema = get_netflix_schema()
    
    df_raw = spark.read \
        .option("header", True) \
        .schema(schema) \
        .csv(RAW_PATH)
    
    initial_count = df_raw.count()
    print(f"✓ Loaded {initial_count:,} records from Bronze layer")
    
    # ========================================================================
    # STEP 2: VALIDATE - Separate valid and invalid records
    # ========================================================================
    df_valid, df_rejected = validate_and_separate_records(df_raw)
    
    # ========================================================================
    # STEP 3: TRANSFORM - Apply business rules
    # ========================================================================
    log_section("Step 2: Transform & Cleanse")
    
    # Column renaming for clarity
    df_processed = df_valid \
        .withColumnRenamed("type", "content_type") \
        .withColumnRenamed("listed_in", "genre") \
        .withColumnRenamed("cast", "cast_and_crew")
    
    # Deduplication
    before_dedup = df_processed.count()
    df_processed = df_processed.dropDuplicates(["show_id"])
    duplicates_removed = before_dedup - df_processed.count()
    print(f"✓ Removed {duplicates_removed:,} duplicate records")
    
    # Apply transformations
    df_processed = apply_data_quality_rules(df_processed)
    df_processed = enrich_with_business_features(df_processed)
    df_processed = add_audit_columns(df_processed)
    
    final_count = df_processed.count()
    
    # ========================================================================
    # STEP 4: QUALITY METRICS & VALIDATION
    # ========================================================================
    metrics = generate_quality_report(
        initial_count, 
        final_count, 
        df_rejected.count(),
        duplicates_removed
    )
    
    display_sample_output(df_processed)
    
    # ========================================================================
    # STEP 5: LOAD - Write to Silver Layer
    # ========================================================================
    write_to_silver_layer(df_processed, PROCESSED_PATH)
    write_rejected_records(df_rejected, REJECTED_PATH)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    log_section("ETL Pipeline Completed Successfully")
    print(f"Total Columns in Output: {len(df_processed.columns)}")
    print(f"Data Quality Score: {metrics['success_rate']:.2f}% records processed")
    print("✅ Job completed successfully")
    
    job.commit()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
