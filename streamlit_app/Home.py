"""
Netflix Analytics Dashboard - Home Page
"""
import streamlit as st
import pandas as pd
from utils.data_loader import DataLoader
from config import *



# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #e7d7d7;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #e7d7d7;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F5F5F1;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="main-header" style="text-align:center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg" width="250">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="main-header">Lakehouse Analytics Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Data Pipeline - Bronze → Silver → Gold</div>', unsafe_allow_html=True)
st.markdown("---")

# Introduction
st.markdown("""
### 🎯 Project Overview
This interactive dashboard visualizes insights from the **Netflix Content Analytics Pipeline**, 
a production-grade data engineering project built with AWS services.

**Architecture:** Medallion (Bronze → Silver → Gold)  
**Data Source:** Netflix catalog (8,807+ titles)  
**Tech Stack:** AWS S3, Glue, Athena, PySpark, SQL, Streamlit
""")

# Load data
@st.cache_resource
def get_data_loader():
    return DataLoader()

loader = get_data_loader()

with st.spinner("Loading data from AWS Athena..."):
    try:
        overview_df = loader.load_content_overview()
        
        if not overview_df.empty:
            st.success("✅ Successfully connected to AWS Data Lake!")
            
            # Key Metrics
            st.markdown("---")
            st.markdown("### 📊 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Content",
                    value=f"{overview_df['total_content_count'].iloc[0]:,}",
                    delta="8.8K+ titles"
                )
            
            with col2:
                st.metric(
                    label="Movies",
                    value=f"{overview_df['total_movies'].iloc[0]:,}",
                    delta=f"{overview_df['movie_percentage'].iloc[0]:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="TV Shows",
                    value=f"{overview_df['total_tv_shows'].iloc[0]:,}",
                    delta=f"{overview_df['tv_show_percentage'].iloc[0]:.1f}%"
                )
            
            with col4:
                st.metric(
                    label="Data Quality",
                    value=f"{overview_df['avg_quality_score'].iloc[0]:.0%}",
                    delta="High quality"
                )
            
            # Additional Stats
            st.markdown("---")
            st.markdown("### 🌍 Content Diversity")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**{overview_df['unique_countries'].iloc[0]}** Countries")
            
            with col2:
                st.info(f"**{overview_df['unique_genres'].iloc[0]}** Genres")
            
            with col3:
                st.info(f"**{overview_df['avg_content_age_years'].iloc[0]:.1f}** Years Avg Age")
        
        else:
            st.warning("⚠️ No data loaded. Check AWS configuration.")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("💡 Make sure AWS credentials are configured correctly.")

# Navigation Guide
st.markdown("---")
st.markdown("### 🧭 Navigation Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📊 Analytics Pages:**
    - **Content Overview** - High-level statistics and KPIs
    - **Genre Analysis** - Genre distribution and trends
    - **Geographic Insights** - Country-wise content analysis
    - **Rating Analysis** - Content rating distribution
    """)

with col2:
    st.markdown("""
    **📈 Advanced Analytics:**
    - **Temporal Trends** - Time-series content addition patterns
    - **Quality Scorecard** - Data quality metrics
    - **Top Producers** - Director and producer insights
    """)

# Architecture Diagram
st.markdown("---")
st.markdown("### 🏗️ Data Pipeline Architecture")

st.code("""
📦 BRONZE LAYER (Raw Data)
   └─ S3: raw/ → Glue Catalog: netflix_raw_db.raw

         ⬇️ AWS Glue Job (PySpark) Transforming Bronze → Silver

📊 SILVER LAYER (Processed Data)
   └─ S3: processed/ → Glue Catalog: netflix_processed_db.netflix_silver_processed

         ⬇️ AWS Glue Job (PySpark) Transforming Silver → Gold

🏆 GOLD LAYER (Business Analytics)
   └─ S3: curated/ → Glue Catalog: netflix_curated_db.*
   
         ⬇️ Streamlit Dashboard (Current) Querying Gold Layer via Athena
         
📈 You are here!
""", language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Netflix Analytics Pipeline</strong></p>
    <p>Built by <i><b>Mohamed Khasim | Data Engineer </i> | 2026</b></p>
    <p>Tech Stack: <b>AWS S3, Glue, Athena, PySpark, Streamlit</b></p>
</div>
""", unsafe_allow_html=True)
