"""
******************************************************************************
 NETFLIX DATA PIPELINE - SILVER TO GOLD LAYER TRANSFORMATION (PySpark)
 
 Purpose: Create business-ready analytical tables for Streamlit dashboards
 Technology: AWS Glue (PySpark)
 Data Format: Parquet with Snappy compression
 
 Architecture:
   Silver Layer (processed/) -> Gold Layer (curated/)
 
 Author: Mohamed Khasim
 Created: 02-08-2026
 
 Business Value:
   - Pre-aggregated metrics for fast dashboard loading
   - Denormalized tables optimized for analytics
   - Cost-effective querying (vs. on-demand aggregation)
   
 Gold Tables Created:
   1. content_overview - Executive KPIs
   2. genre_analysis - Genre metrics
   3. geographic_distribution - Country-based analytics
   4. temporal_trends - Monthly addition trends
   5. rating_distribution - Rating category analysis
   6. quality_scorecard - Data quality monitoring
   7. top_producers - Director/producer insights
******************************************************************************
"""

import sys
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

# =============================================================================
# INITIALIZATION
# =============================================================================

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'SILVER_S3_PATH',      # s3://netflix-pipeline-khasim-2026/processed/
    'GOLD_S3_PATH',        # s3://netflix-pipeline-khasim-2026/curated/
    'DATABASE_NAME'        # netflix_processed_db
])

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Configuration
SILVER_PATH = args['SILVER_S3_PATH']
GOLD_PATH = args['GOLD_S3_PATH']
DATABASE = args['DATABASE_NAME']

# Set Spark configurations for optimization
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")

print(f"Starting Silver to Gold transformation at {datetime.now()}")
print(f"Silver Path: {SILVER_PATH}")
print(f"Gold Path: {GOLD_PATH}")
print(f"Database: {DATABASE}")


# =============================================================================
# STEP 1: READ SILVER LAYER DATA
# =============================================================================

def read_silver_data():
    """Read processed data from Silver layer"""
    try:
        print("Reading Silver layer data...")
        df = spark.read.parquet(SILVER_PATH)
        
        # Cache the dataframe since we'll use it multiple times
        df.cache()
        
        record_count = df.count()
        print(f"Successfully read {record_count:,} records from Silver layer")
        print(f"Partitions: {df.rdd.getNumPartitions()}")
        
        return df
    except Exception as e:
        print(f"Error reading Silver data: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 1: CONTENT OVERVIEW METRICS
# =============================================================================

def create_content_overview(silver_df):
    """
    Executive dashboard KPIs
    Business Use: Leadership, Product teams
    """
    print("\n" + "="*80)
    print("Creating Gold Table 1: Content Overview")
    print("="*80)
    
    try:
        overview = silver_df.agg(
            # Metadata
            F.current_date().alias('report_date'),
            
            # Overall metrics
            F.countDistinct('show_id').alias('total_content_count'),
            F.countDistinct(F.when(F.col('content_type') == 'Movie', F.col('show_id'))).alias('total_movies'),
            F.countDistinct(F.when(F.col('content_type') == 'TV Show', F.col('show_id'))).alias('total_tv_shows'),
            
            # Quality metrics
            F.round(F.avg('data_quality_score'), 3).alias('avg_quality_score'),
            F.sum(F.when(F.col('data_quality_score') >= 0.8, 1).otherwise(0)).alias('high_quality_content_count'),
            
            # Completeness metrics
            F.sum(F.when(F.col('has_director') == True, 1).otherwise(0)).alias('content_with_director'),
            F.sum(F.when(F.col('has_cast') == True, 1).otherwise(0)).alias('content_with_cast'),
            
            # Temporal metrics
            F.sum(F.when(F.col('is_recent') == True, 1).otherwise(0)).alias('recent_content_count'),
            F.min('date_added').alias('earliest_content_added'),
            F.max('date_added').alias('latest_content_added'),
            F.min('release_year').alias('oldest_release_year'),
            F.max('release_year').alias('newest_release_year'),
            
            # Average content age
            F.round(F.avg('content_age_years'), 1).alias('avg_content_age_years'),
            
            # Diversity metrics
            F.countDistinct('primary_country').alias('unique_countries'),
            F.countDistinct('primary_genre').alias('unique_genres')
        )
        
        # Calculate percentages
        overview = overview.withColumn(
            'movie_percentage',
            F.round((F.col('total_movies') * 100.0 / F.col('total_content_count')), 2)
        ).withColumn(
            'tv_show_percentage',
            F.round((F.col('total_tv_shows') * 100.0 / F.col('total_content_count')), 2)
        ).withColumn(
            'high_quality_percentage',
            F.round((F.col('high_quality_content_count') * 100.0 / F.col('total_content_count')), 2)
        ).withColumn(
            'director_completeness_pct',
            F.round((F.col('content_with_director') * 100.0 / F.col('total_content_count')), 2)
        )
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}content_overview/"
        overview.coalesce(1).write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Content Overview created successfully")
        print(f"  Output: {output_path}")
        overview.show(vertical=True, truncate=False)
        
        return overview
        
    except Exception as e:
        print(f"Error creating content_overview: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 2: GENRE ANALYSIS
# =============================================================================

def create_genre_analysis(silver_df):
    """
    Content strategy and acquisition planning metrics
    Business Use: Content teams, Marketing
    """
    print("\n" + "="*80)
    print("Creating Gold Table 2: Genre Analysis")
    print("="*80)
    
    try:
        # Filter and aggregate by genre
        genre_base = silver_df.filter(
            (F.col('primary_genre').isNotNull()) & 
            (F.col('primary_genre') != 'Uncategorized')
        ).groupBy('primary_genre', 'content_type').agg(
            # Volume
            F.countDistinct('show_id').alias('content_count'),
            
            # Quality
            F.avg('data_quality_score').alias('avg_quality_score'),
            
            # Temporal
            F.min('release_year').alias('earliest_release_year'),
            F.max('release_year').alias('latest_release_year'),
            F.avg('content_age_years').alias('avg_content_age_years'),
            
            # Recency
            F.sum(F.when(F.col('is_recent') == True, 1).otherwise(0)).alias('recent_content_count'),
            F.sum(F.when(F.col('added_year') >= 2020, 1).otherwise(0)).alias('added_since_2020'),
            
            # Duration
            F.avg('duration_value').alias('avg_duration_value'),
            
            # Completeness
            (F.avg(F.when(F.col('has_director'), 1.0).otherwise(0.0)) * 100).alias('director_completeness_pct'),
            (F.avg(F.when(F.col('has_cast'), 1.0).otherwise(0.0)) * 100).alias('cast_completeness_pct')
        ).filter(
            F.col('content_count') >= 5
        )
        
        # Calculate percentage within content type
        window_spec = Window.partitionBy('content_type')
        genre_analysis = genre_base.withColumn(
            'percentage_of_type',
            F.round((F.col('content_count') * 100.0 / F.sum('content_count').over(window_spec)), 2)
        ).withColumn(
            'avg_quality_score', F.round(F.col('avg_quality_score'), 3)
        ).withColumn(
            'avg_content_age_years', F.round(F.col('avg_content_age_years'), 1)
        ).withColumn(
            'avg_duration_value', F.round(F.col('avg_duration_value'), 1)
        ).withColumn(
            'director_completeness_pct', F.round(F.col('director_completeness_pct'), 2)
        ).withColumn(
            'cast_completeness_pct', F.round(F.col('cast_completeness_pct'), 2)
        ).orderBy(F.desc('content_count'))
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}genre_analysis/"
        genre_analysis.write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Genre Analysis created successfully")
        print(f"  Output: {output_path}")
        print(f"  Records: {genre_analysis.count():,}")
        genre_analysis.show(10, truncate=False)
        
        return genre_analysis
        
    except Exception as e:
        print(f"Error creating genre_analysis: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 3: GEOGRAPHIC DISTRIBUTION
# =============================================================================

def create_geographic_distribution(silver_df):
    """
    Regional content strategy and licensing decisions
    Business Use: International teams, Business development
    """
    print("\n" + "="*80)
    print("Creating Gold Table 3: Geographic Distribution")
    print("="*80)
    
    try:
        # Base aggregation by country
        base_agg = silver_df.filter(
            (F.col('primary_country').isNotNull()) & 
            (F.col('primary_country') != 'Unknown')
        ).groupBy('primary_country', 'content_type').agg(
            F.countDistinct('show_id').alias('content_count'),
            F.countDistinct(F.when(F.col('content_type') == 'Movie', F.col('show_id'))).alias('movie_count'),
            F.countDistinct(F.when(F.col('content_type') == 'TV Show', F.col('show_id'))).alias('tv_show_count'),
            F.avg('data_quality_score').alias('avg_quality_score'),
            F.sum(F.when(F.col('added_year') == 2021, 1).otherwise(0)).alias('added_2021'),
            F.sum(F.when(F.col('added_year') == 2020, 1).otherwise(0)).alias('added_2020'),
            F.sum(F.when(F.col('added_year') == 2019, 1).otherwise(0)).alias('added_2019'),
            F.sum(F.when(F.col('is_recent') == True, 1).otherwise(0)).alias('recent_content_count'),
            F.avg('content_age_years').alias('avg_content_age_years')
        ).filter(
            F.col('content_count') >= 10
        )
        
        # Calculate total per country
        country_totals = base_agg.groupBy('primary_country').agg(
            F.sum('content_count').alias('total_country_content')
        )
        
        # Join and calculate percentages
        geo_dist = base_agg.join(country_totals, 'primary_country').withColumn(
            'percentage_of_country',
            F.round((F.col('content_count') * 100.0 / F.col('total_country_content')), 2)
        ).withColumn(
            'avg_quality_score', F.round(F.col('avg_quality_score'), 3)
        ).withColumn(
            'avg_content_age_years', F.round(F.col('avg_content_age_years'), 1)
        ).drop('total_country_content').orderBy(F.desc('content_count'))
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}geographic_distribution/"
        geo_dist.write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Geographic Distribution created successfully")
        print(f"  Output: {output_path}")
        print(f"  Records: {geo_dist.count():,}")
        geo_dist.show(10, truncate=False)
        
        return geo_dist
        
    except Exception as e:
        print(f"Error creating geographic_distribution: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 4: TEMPORAL TRENDS
# =============================================================================

def create_temporal_trends(silver_df):
    """
    Content acquisition trends and forecasting
    Business Use: Analytics teams, Finance
    """
    print("\n" + "="*80)
    print("Creating Gold Table 4: Temporal Trends")
    print("="*80)
    
    try:
        # Monthly aggregation
        monthly_base = silver_df.filter(
            F.col('date_added').isNotNull() &
            F.col('added_year').isNotNull() &
            F.col('added_month').isNotNull()
        ).groupBy('added_year', 'added_month', 'content_type').agg(
            F.countDistinct('show_id').alias('content_added_count'),
            F.avg('data_quality_score').alias('avg_quality_score'),
            F.avg('content_age_years').alias('avg_age_of_content_added')
        )
        
        # Add month names
        temporal = monthly_base.withColumn(
            'month_name',
            F.when(F.col('added_month') == 1, 'January')
            .when(F.col('added_month') == 2, 'February')
            .when(F.col('added_month') == 3, 'March')
            .when(F.col('added_month') == 4, 'April')
            .when(F.col('added_month') == 5, 'May')
            .when(F.col('added_month') == 6, 'June')
            .when(F.col('added_month') == 7, 'July')
            .when(F.col('added_month') == 8, 'August')
            .when(F.col('added_month') == 9, 'September')
            .when(F.col('added_month') == 10, 'October')
            .when(F.col('added_month') == 11, 'November')
            .when(F.col('added_month') == 12, 'December')
        )
        
        # Calculate cumulative count
        window_spec = Window.partitionBy('content_type').orderBy('added_year', 'added_month').rowsBetween(Window.unboundedPreceding, Window.currentRow)
        temporal = temporal.withColumn(
            'cumulative_content_count',
            F.sum('content_added_count').over(window_spec)
        ).withColumn(
            'avg_quality_score', F.round(F.col('avg_quality_score'), 3)
        ).withColumn(
            'avg_age_of_content_added', F.round(F.col('avg_age_of_content_added'), 1)
        ).orderBy(F.desc('added_year'), F.desc('added_month'), 'content_type')
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}temporal_trends/"
        temporal.write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Temporal Trends created successfully")
        print(f"  Output: {output_path}")
        print(f"  Records: {temporal.count():,}")
        temporal.show(10, truncate=False)
        
        return temporal
        
    except Exception as e:
        print(f"Error creating temporal_trends: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 5: RATING DISTRIBUTION
# =============================================================================

def create_rating_distribution(silver_df):
    """
    Content compliance and audience targeting
    Business Use: Compliance teams, Marketing
    """
    print("\n" + "="*80)
    print("Creating Gold Table 5: Rating Distribution")
    print("="*80)
    
    try:
        # Base aggregation by rating
        rating_base = silver_df.filter(
            F.col('rating').isNotNull()
        ).groupBy('rating', 'content_type').agg(
            F.countDistinct('show_id').alias('content_count'),
            F.avg('data_quality_score').alias('avg_quality_score'),
            F.sum(F.when(F.col('is_recent') == True, 1).otherwise(0)).alias('recent_content_count'),
            F.avg('content_age_years').alias('avg_content_age_years'),
            F.avg('duration_value').alias('avg_duration_value')
        )
        
        # Add rating categories
        rating_dist = rating_base.withColumn(
            'rating_category',
            F.when(F.col('rating').isin(['G', 'TV-Y', 'TV-G']), 'Kids & Family')
            .when(F.col('rating').isin(['PG', 'TV-PG', 'TV-Y7', 'TV-Y7-FV']), 'Older Kids & Teens')
            .when(F.col('rating').isin(['PG-13', 'TV-14']), 'Teens & Adults')
            .when(F.col('rating').isin(['R', 'TV-MA', 'NC-17']), 'Mature Audiences')
            .when(F.col('rating').isin(['NR', 'UNRATED', 'UR']), 'Unrated')
            .otherwise('Other')
        )
        
        # Calculate percentages
        window_spec = Window.partitionBy('content_type')
        rating_dist = rating_dist.withColumn(
            'percentage_of_type',
            F.round((F.col('content_count') * 100.0 / F.sum('content_count').over(window_spec)), 2)
        ).withColumn(
            'avg_quality_score', F.round(F.col('avg_quality_score'), 3)
        ).withColumn(
            'avg_content_age_years', F.round(F.col('avg_content_age_years'), 1)
        ).withColumn(
            'avg_duration_value', F.round(F.col('avg_duration_value'), 1)
        ).orderBy(F.desc('content_count'))
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}rating_distribution/"
        rating_dist.write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Rating Distribution created successfully")
        print(f"  Output: {output_path}")
        print(f"  Records: {rating_dist.count():,}")
        rating_dist.show(10, truncate=False)
        
        return rating_dist
        
    except Exception as e:
        print(f"Error creating rating_distribution: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 6: CONTENT QUALITY SCORECARD
# =============================================================================

def create_quality_scorecard(silver_df):
    """
    Data quality monitoring and content enrichment prioritization
    Business Use: Data engineering teams, Content operations
    """
    print("\n" + "="*80)
    print("Creating Gold Table 6: Quality Scorecard")
    print("="*80)
    
    try:
        # Define quality tiers
        df_with_tiers = silver_df.withColumn(
            'quality_tier',
            F.when(F.col('data_quality_score') >= 0.9, 'Excellent (0.9-1.0)')
            .when(F.col('data_quality_score') >= 0.7, 'Good (0.7-0.89)')
            .when(F.col('data_quality_score') >= 0.5, 'Fair (0.5-0.69)')
            .otherwise('Poor (<0.5)')
        )
        
        # Get sample titles (top 5 per tier)
        window_spec = Window.partitionBy('content_type', 'quality_tier').orderBy('title')
        ranked_titles = df_with_tiers.withColumn(
            'rn', F.row_number().over(window_spec)
        ).filter(F.col('rn') <= 5)
        
        sample_titles = ranked_titles.groupBy('content_type', 'quality_tier').agg(
            F.collect_list('title').alias('sample_titles')
        )
        
        # Quality base aggregation
        quality_base = df_with_tiers.groupBy('content_type', 'quality_tier').agg(
            F.countDistinct('show_id').alias('content_count'),
            F.avg('data_quality_score').alias('avg_quality_score'),
            (F.avg(F.when(F.col('has_director'), 1.0).otherwise(0.0)) * 100).alias('has_director_pct'),
            (F.avg(F.when(F.col('has_cast'), 1.0).otherwise(0.0)) * 100).alias('has_cast_pct'),
            F.sum(F.when(F.col('duration_value').isNotNull(), 1).otherwise(0)).alias('has_duration_count'),
            F.sum(F.when(F.col('date_added').isNotNull(), 1).otherwise(0)).alias('has_date_added_count'),
            F.sum(F.when(F.col('release_year').isNotNull(), 1).otherwise(0)).alias('has_release_year_count')
        )
        
        # Join with sample titles
        quality = quality_base.join(sample_titles, ['content_type', 'quality_tier'], 'left')
        
        # Calculate percentages
        window_spec = Window.partitionBy('content_type')
        quality = quality.withColumn(
            'percentage_of_type',
            F.round((F.col('content_count') * 100.0 / F.sum('content_count').over(window_spec)), 2)
        ).withColumn(
            'avg_quality_score', F.round(F.col('avg_quality_score'), 3)
        ).withColumn(
            'has_director_pct', F.round(F.col('has_director_pct'), 2)
        ).withColumn(
            'has_cast_pct', F.round(F.col('has_cast_pct'), 2)
        ).orderBy('content_type', F.desc('avg_quality_score'))
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}quality_scorecard/"
        quality.write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Quality Scorecard created successfully")
        print(f"  Output: {output_path}")
        print(f"  Records: {quality.count():,}")
        quality.show(10, truncate=False)
        
        return quality
        
    except Exception as e:
        print(f"Error creating quality_scorecard: {str(e)}")
        raise


# =============================================================================
# GOLD TABLE 7: TOP CONTENT PRODUCERS
# =============================================================================

def create_top_producers(silver_df):
    """
    Partnership opportunities and content acquisition strategy
    Business Use: Business development, Content acquisition teams
    """
    print("\n" + "="*80)
    print("Creating Gold Table 7: Top Producers")
    print("="*80)
    
    try:
        # Filter for directors (exclude collaborative credits)
        director_stats = silver_df.filter(
            (F.col('director').isNotNull()) &
            (F.col('director') != 'Unknown') &
            (~F.col('director').contains(','))  # Exclude collaborative credits
        ).groupBy('director', 'content_type').agg(
            F.countDistinct('show_id').alias('content_count'),
            F.avg('data_quality_score').alias('avg_quality_score'),
            F.collect_set('primary_genre').alias('genres_worked_in'),
            F.min('release_year').alias('first_release_year'),
            F.max('release_year').alias('latest_release_year'),
            F.sum(F.when(F.col('is_recent') == True, 1).otherwise(0)).alias('recent_works_count')
        ).filter(
            F.col('content_count') >= 2
        )
        
        # Calculate years active
        director_stats = director_stats.withColumn(
            'years_active',
            F.col('latest_release_year') - F.col('first_release_year')
        ).withColumn(
            'avg_quality_score', F.round(F.col('avg_quality_score'), 3)
        )
        
        # Rank by volume
        window_spec = Window.partitionBy('content_type').orderBy(F.desc('content_count'), F.desc('avg_quality_score'))
        top_producers = director_stats.withColumn(
            'rank_by_volume',
            F.row_number().over(window_spec)
        ).filter(
            F.col('rank_by_volume') <= 100
        ).orderBy('content_type', 'rank_by_volume')
        
        # Write to Gold layer
        output_path = f"{GOLD_PATH}top_producers/"
        top_producers.write.mode('overwrite').parquet(output_path)
        
        print(f"✓ Top Producers created successfully")
        print(f"  Output: {output_path}")
        print(f"  Records: {top_producers.count():,}")
        top_producers.show(10, truncate=False)
        
        return top_producers
        
    except Exception as e:
        print(f"Error creating top_producers: {str(e)}")
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main ETL orchestration"""
    try:
        print("\n" + "="*80)
        print("NETFLIX SILVER TO GOLD ETL PIPELINE")
        print("="*80)
        print(f"Start Time: {datetime.now()}")
        
        # Step 1: Read Silver data
        silver_df = read_silver_data()
        
        # Step 2: Create all Gold tables
        tables_created = []
        
        try:
            overview = create_content_overview(silver_df)
            tables_created.append('content_overview')
        except Exception as e:
            print(f"Failed to create content_overview: {str(e)}")
        
        try:
            genre = create_genre_analysis(silver_df)
            tables_created.append('genre_analysis')
        except Exception as e:
            print(f"Failed to create genre_analysis: {str(e)}")
        
        try:
            geo = create_geographic_distribution(silver_df)
            tables_created.append('geographic_distribution')
        except Exception as e:
            print(f"Failed to create geographic_distribution: {str(e)}")
        
        try:
            temporal = create_temporal_trends(silver_df)
            tables_created.append('temporal_trends')
        except Exception as e:
            print(f"Failed to create temporal_trends: {str(e)}")
        
        try:
            rating = create_rating_distribution(silver_df)
            tables_created.append('rating_distribution')
        except Exception as e:
            print(f"Failed to create rating_distribution: {str(e)}")
        
        try:
            quality = create_quality_scorecard(silver_df)
            tables_created.append('quality_scorecard')
        except Exception as e:
            print(f"Failed to create quality_scorecard: {str(e)}")
        
        try:
            producers = create_top_producers(silver_df)
            tables_created.append('top_producers')
        except Exception as e:
            print(f"Failed to create top_producers: {str(e)}")
        
        # Step 3: Summary
        print("\n" + "="*80)
        print("ETL PIPELINE SUMMARY")
        print("="*80)
        print(f"End Time: {datetime.now()}")
        print(f"Total Tables Created: {len(tables_created)}/7")
        print(f"Successfully Created: {', '.join(tables_created)}")
        
        if len(tables_created) < 7:
            print("\n⚠ WARNING: Some tables failed to create. Check logs above.")
        else:
            print("\n✓ All Gold tables created successfully!")
        
        # Unpersist cached dataframe
        silver_df.unpersist()
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Run MSCK REPAIR TABLE on Gold tables in Athena")
        print("2. Verify data quality with sample queries")
        print("3. Update Streamlit dashboards to query Gold tables")
        print("4. Set up AWS Glue Triggers for automated updates")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in main execution: {str(e)}")
        raise
    finally:
        # Commit the job
        job.commit()
        print(f"\nGlue Job completed at {datetime.now()}")


if __name__ == "__main__":
    main()
