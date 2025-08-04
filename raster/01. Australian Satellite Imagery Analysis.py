# Databricks notebook source
# MAGIC %md
# MAGIC # Australian Satellite Imagery Analysis with Databricks
# MAGIC
# MAGIC This notebook demonstrates how to process, analyze, and visualize Australian satellite imagery using Databricks.
# MAGIC
# MAGIC ## What we'll cover:
# MAGIC 1. Loading satellite imagery (Landsat/Sentinel-2)
# MAGIC 2. Basic raster processing and analysis
# MAGIC 3. Vegetation index calculations (NDVI, EVI)
# MAGIC 4. Time series analysis
# MAGIC 5. Interactive visualizations
# MAGIC 6. Change detection
# MAGIC 7. Export results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Installation

# COMMAND ----------

# MAGIC %pip install "numpy==1.26.4" rasterio xarray matplotlib plotly folium scikit-image pandas seaborn
# MAGIC %restart_python

# COMMAND ----------

import os
import pandas as pd
# Handle NumPy compatibility
try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")
    raise
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import warnings
warnings.filterwarnings('ignore')

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, expr, when, isnan, isnull, avg, count, sum as spark_sum,
    min as spark_min, max as spark_max, stddev, collect_list
)
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

# Geospatial imports
try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    import xarray as xr
    print("✅ All geospatial libraries loaded successfully!")
except ImportError as e:
    print(f"⚠️ Some libraries not available: {e}")

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Sources
# MAGIC
# MAGIC ### Sample Data Locations
# MAGIC - **Sydney Basin**: Urban growth analysis
# MAGIC - **Murray River**: Agricultural monitoring
# MAGIC - **Kakadu National Park**: Environmental monitoring
# MAGIC - **Perth Region**: Coastal change detection

# COMMAND ----------

class AustraliaSatelliteAnalyzer:
    """
    Australian Satellite Imagery Analysis Tool for Databricks
    Optimized for hackathon use with real data examples
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.crs = "EPSG:4326"  # WGS84
        
        # Australian regions of interest
        self.regions = {
            'sydney': {
                'name': 'Sydney Basin',
                'bbox': [150.5, -34.2, 151.5, -33.2],
                'description': 'Urban growth and development analysis'
            },
            'murray_river': {
                'name': 'Murray River Valley',
                'bbox': [140.5, -35.5, 145.0, -33.5],
                'description': 'Agricultural and water resource monitoring'
            },
            'kakadu': {
                'name': 'Kakadu National Park',
                'bbox': [132.0, -13.0, 133.5, -12.0],
                'description': 'Environmental and vegetation monitoring'
            },
            'perth': {
                'name': 'Perth Metropolitan',
                'bbox': [115.5, -32.5, 116.5, -31.5],
                'description': 'Coastal and urban development'
            }
        }
        
        # Band information for Landsat 8/9
        self.landsat_bands = {
            'coastal': {'band_num': 1, 'wavelength': '0.43-0.45μm', 'description': 'Coastal/Aerosol'},
            'blue': {'band_num': 2, 'wavelength': '0.45-0.51μm', 'description': 'Blue'},
            'green': {'band_num': 3, 'wavelength': '0.53-0.59μm', 'description': 'Green'},
            'red': {'band_num': 4, 'wavelength': '0.64-0.67μm', 'description': 'Red'},
            'nir': {'band_num': 5, 'wavelength': '0.85-0.88μm', 'description': 'Near-Infrared'},
            'swir1': {'band_num': 6, 'wavelength': '1.57-1.65μm', 'description': 'Short-wave Infrared 1'},
            'swir2': {'band_num': 7, 'wavelength': '2.11-2.29μm', 'description': 'Short-wave Infrared 2'}
        }
        
        print(f"🛰️ Australian Satellite Analyzer initialized")
        print(f"📍 Available regions: {list(self.regions.keys())}")
    
    def create_sample_landsat_data(self, region='sydney', size=(100, 100), n_timesteps=12):
        """
        Create realistic sample Landsat data for demonstration
        
        Args:
            region (str): Region name from self.regions
            size (tuple): Image dimensions (height, width)
            n_timesteps (int): Number of time steps
        """
        print(f"🔄 Creating sample Landsat data for {self.regions[region]['name']}...")
        
        np.random.seed(42)  # For reproducible results
        
        # Get region bounds
        bbox = self.regions[region]['bbox']
        lon_min, lat_min, lon_max, lat_max = bbox
        
        # Create coordinate arrays
        lons = np.linspace(lon_min, lon_max, size[1])
        lats = np.linspace(lat_max, lat_min, size[0])  # Note: lat decreases with y
        
        # Create time array (monthly data for one year)
        dates = pd.date_range('2023-01-01', periods=n_timesteps, freq='M')
        
        # Initialize data arrays
        data = {}
        
        # Simulate realistic reflectance values for each band
        for band_name, band_info in self.landsat_bands.items():
            if band_name in ['coastal', 'blue', 'green', 'red']:
                # Visible bands: lower reflectance, some seasonal variation
                base_values = np.random.normal(0.08, 0.02, size)
                seasonal_factor = np.sin(np.arange(n_timesteps) * 2 * np.pi / 12) * 0.02
            elif band_name == 'nir':
                # NIR: higher for vegetation, seasonal variation
                base_values = np.random.normal(0.25, 0.05, size)
                seasonal_factor = np.sin(np.arange(n_timesteps) * 2 * np.pi / 12) * 0.08
            else:  # SWIR bands
                # SWIR: medium values, less seasonal variation
                base_values = np.random.normal(0.15, 0.03, size)
                seasonal_factor = np.sin(np.arange(n_timesteps) * 2 * np.pi / 12) * 0.03
            
            # Add spatial patterns (simulate land cover)
            y_indices, x_indices = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
            
            # Create land cover patterns
            if region == 'sydney':
                # Urban center with vegetation around
                urban_mask = ((y_indices - size[0]//2)**2 + (x_indices - size[1]//2)**2) < (min(size)//4)**2
                base_values[urban_mask] *= 0.5  # Lower reflectance in urban areas
                if band_name == 'nir':
                    base_values[~urban_mask] *= 1.5  # Higher NIR in vegetated areas
            
            # Add temporal dimension
            time_series = np.zeros((n_timesteps, size[0], size[1]))
            for t in range(n_timesteps):
                time_series[t] = np.clip(base_values + seasonal_factor[t] + 
                                       np.random.normal(0, 0.01, size), 0, 1)
            
            data[band_name] = time_series
        
        # Create xarray Dataset
        dataset = xr.Dataset(
            data_vars={
                band: (['time', 'y', 'x'], data[band]) 
                for band in self.landsat_bands.keys()
            },
            coords={
                'time': dates,
                'y': lats,
                'x': lons
            },
            attrs={
                'title': f'Sample Landsat Data - {self.regions[region]["name"]}',
                'region': region,
                'crs': self.crs,
                'description': self.regions[region]['description']
            }
        )
        
        print(f"✅ Created dataset with shape: {dataset.dims}")
        print(f"📅 Time range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"🌍 Spatial extent: {lon_min:.2f}°E to {lon_max:.2f}°E, {lat_min:.2f}°N to {lat_max:.2f}°N")
        
        return dataset
    
    def calculate_vegetation_indices(self, dataset):
        """
        Calculate vegetation indices from satellite bands
        
        Args:
            dataset: xarray Dataset with satellite bands
        """
        print("🌱 Calculating vegetation indices...")
        
        # NDVI (Normalized Difference Vegetation Index)
        dataset['ndvi'] = (dataset['nir'] - dataset['red']) / (dataset['nir'] + dataset['red'])
        
        # EVI (Enhanced Vegetation Index)
        dataset['evi'] = 2.5 * ((dataset['nir'] - dataset['red']) / 
                               (dataset['nir'] + 6 * dataset['red'] - 7.5 * dataset['blue'] + 1))
        
        # NDWI (Normalized Difference Water Index)
        dataset['ndwi'] = (dataset['green'] - dataset['nir']) / (dataset['green'] + dataset['nir'])
        
        # NDBI (Normalized Difference Built-up Index)
        dataset['ndbi'] = (dataset['swir1'] - dataset['nir']) / (dataset['swir1'] + dataset['nir'])
        
        print("✅ Calculated indices: NDVI, EVI, NDWI, NDBI")
        return dataset
    
    def convert_to_dataframe(self, dataset, sample_points=None):
        """
        Convert xarray dataset to pandas DataFrame for Spark processing
        
        Args:
            dataset: xarray Dataset
            sample_points: Number of points to sample (None for all)
        """
        print("🔄 Converting to DataFrame for Spark processing...")
        
        # Stack the dataset to create a flat structure
        stacked = dataset.stack(point=('y', 'x'))
        
        # Convert to DataFrame
        df_data = []
        
        for time_idx, time_val in enumerate(dataset.time.values):
            for point_idx in range(len(stacked.point)):
                y_val = float(stacked.y.values[point_idx])
                x_val = float(stacked.x.values[point_idx])
                
                row = {
                    'time': pd.Timestamp(time_val),
                    'latitude': y_val,
                    'longitude': x_val,
                    'pixel_id': f"{y_val:.4f}_{x_val:.4f}"
                }
                
                # Add all bands and indices
                for var in dataset.data_vars:
                    value = float(stacked[var].values[time_idx, point_idx])
                    if not np.isnan(value):
                        row[var] = value
                
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sample if requested
        if sample_points and len(df) > sample_points:
            df = df.sample(n=sample_points, random_state=42)
            print(f"🎯 Sampled {sample_points} points from {len(df_data)} total")
        
        print(f"✅ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Analyzer and Create Sample Data

# COMMAND ----------

import numpy as np
import xarray as xr


# Initialize the analyzer
analyzer = AustraliaSatelliteAnalyzer(spark)

# Create sample data for Sydney region
sydney_data = analyzer.create_sample_landsat_data(region='sydney', size=(50, 50), n_timesteps=12)

# Calculate vegetation indices
sydney_data = analyzer.calculate_vegetation_indices(sydney_data)

# Display dataset information
print(f"\n📊 Dataset Summary:")
print(f"Dimensions: {dict(sydney_data.dims)}")
print(f"Variables: {list(sydney_data.data_vars.keys())}")
print(f"Coordinates: {list(sydney_data.coords.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization 1: True Color Composite

# COMMAND ----------

def create_true_color_composite(dataset, time_index=0, enhance=True):
    """Create a true color RGB composite"""
    
    # Get RGB bands for the specified time
    red = dataset['red'].isel(time=time_index)
    green = dataset['green'].isel(time=time_index)
    blue = dataset['blue'].isel(time=time_index)
    
    # Stack into RGB array
    rgb = np.dstack([red.values, green.values, blue.values])
    
    if enhance:
        # Apply contrast enhancement (2% linear stretch)
        rgb_enhanced = np.zeros_like(rgb)
        for i in range(3):
            band = rgb[:, :, i]
            p2, p98 = np.percentile(band[~np.isnan(band)], [2, 98])
            rgb_enhanced[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        rgb = rgb_enhanced
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True color composite
    ax1.imshow(rgb, extent=[
        dataset.x.min(), dataset.x.max(),
        dataset.y.min(), dataset.y.max()
    ])
    ax1.set_title(f'True Color Composite\n{pd.Timestamp(dataset.time.values[time_index]).strftime("%Y-%m-%d")}')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # NDVI for comparison
    ndvi = dataset['ndvi'].isel(time=time_index)
    im = ax2.imshow(ndvi.values, cmap='RdYlGn', vmin=-0.2, vmax=0.8,
                    extent=[dataset.x.min(), dataset.x.max(),
                           dataset.y.min(), dataset.y.max()])
    ax2.set_title('NDVI (Vegetation Index)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Add colorbar for NDVI
    plt.colorbar(im, ax=ax2, label='NDVI')
    
    plt.tight_layout()
    plt.show()

# Create visualization for the latest time step
create_true_color_composite(sydney_data, time_index=-1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization 2: Multi-Index Comparison

# COMMAND ----------

def plot_vegetation_indices(dataset, time_index=0):
    """Plot multiple vegetation indices side by side"""
    
    indices = ['ndvi', 'evi', 'ndwi', 'ndbi']
    titles = ['NDVI (Vegetation)', 'EVI (Enhanced Vegetation)', 
              'NDWI (Water)', 'NDBI (Built-up)']
    cmaps = ['RdYlGn', 'RdYlGn', 'Blues', 'Reds']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (index, title, cmap) in enumerate(zip(indices, titles, cmaps)):
        data = dataset[index].isel(time=time_index)
        
        im = axes[i].imshow(data.values, cmap=cmap, 
                           extent=[dataset.x.min(), dataset.x.max(),
                                  dataset.y.min(), dataset.y.max()])
        axes[i].set_title(title)
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label=index.upper())
    
    plt.suptitle(f'Satellite Indices Comparison - {pd.Timestamp(dataset.time.values[time_index]).strftime("%Y-%m-%d")}', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Plot indices for the latest time step
plot_vegetation_indices(sydney_data, time_index=-1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Series Analysis with Spark

# COMMAND ----------

# Convert to DataFrame and create Spark DataFrame
df_sydney = analyzer.convert_to_dataframe(sydney_data, sample_points=5000)
spark_df = spark.createDataFrame(df_sydney)

# Cache for better performance
#spark_df.cache()

print(f"📊 Spark DataFrame created with {spark_df.count()} rows")
spark_df.printSchema()

# COMMAND ----------

# Calculate temporal statistics using Spark
print("📈 Calculating temporal statistics...")

# Monthly averages for vegetation indices
monthly_stats = (spark_df
                .groupBy(
                    spark_df.time.cast("date").alias("date")
                )
                .agg(
                    avg("ndvi").alias("avg_ndvi"),
                    avg("evi").alias("avg_evi"),
                    avg("ndwi").alias("avg_ndwi"),
                    avg("ndbi").alias("avg_ndbi"),
                    count("*").alias("pixel_count"),
                    stddev("ndvi").alias("std_ndvi")
                )
                .orderBy("date")
                .toPandas())

print("✅ Monthly statistics calculated")
print(monthly_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Time Series Visualization

# COMMAND ----------

def create_interactive_time_series(monthly_stats):
    """Create interactive time series plots using Plotly"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('NDVI (Vegetation)', 'EVI (Enhanced Vegetation)', 
                       'NDWI (Water)', 'NDBI (Built-up)'),
        vertical_spacing=0.1
    )
    
    # NDVI
    fig.add_trace(
        go.Scatter(x=monthly_stats['date'], y=monthly_stats['avg_ndvi'],
                  mode='lines+markers', name='NDVI',
                  line=dict(color='green', width=3)),
        row=1, col=1
    )
    
    # EVI
    fig.add_trace(
        go.Scatter(x=monthly_stats['date'], y=monthly_stats['avg_evi'],
                  mode='lines+markers', name='EVI',
                  line=dict(color='darkgreen', width=3)),
        row=1, col=2
    )
    
    # NDWI
    fig.add_trace(
        go.Scatter(x=monthly_stats['date'], y=monthly_stats['avg_ndwi'],
                  mode='lines+markers', name='NDWI',
                  line=dict(color='blue', width=3)),
        row=2, col=1
    )
    
    # NDBI
    fig.add_trace(
        go.Scatter(x=monthly_stats['date'], y=monthly_stats['avg_ndbi'],
                  mode='lines+markers', name='NDBI',
                  line=dict(color='red', width=3)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Vegetation Indices Time Series - Sydney Basin',
        height=600,
        showlegend=False
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Date")
    
    # Update y-axis labels
    fig.update_yaxes(title_text="NDVI", row=1, col=1)
    fig.update_yaxes(title_text="EVI", row=1, col=2)
    fig.update_yaxes(title_text="NDWI", row=2, col=1)
    fig.update_yaxes(title_text="NDBI", row=2, col=2)
    
    fig.show()
    
    return fig

# Create interactive plot
time_series_fig = create_interactive_time_series(monthly_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spatial Analysis: Change Detection

# COMMAND ----------

def perform_change_detection(dataset, start_idx=0, end_idx=-1):
    """Perform simple change detection between two time periods"""
    
    print(f"🔍 Performing change detection...")
    print(f"Comparing {pd.Timestamp(dataset.time.values[start_idx]).strftime('%Y-%m-%d')} " +
          f"to {pd.Timestamp(dataset.time.values[end_idx]).strftime('%Y-%m-%d')}")
    
    # Calculate NDVI change
    ndvi_start = dataset['ndvi'].isel(time=start_idx)
    ndvi_end = dataset['ndvi'].isel(time=end_idx)
    ndvi_change = ndvi_end - ndvi_start
    
    # Calculate NDBI change (urbanization indicator)
    ndbi_start = dataset['ndbi'].isel(time=start_idx)
    ndbi_end = dataset['ndbi'].isel(time=end_idx)
    ndbi_change = ndbi_end - ndbi_start
    
    # Create change detection plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # NDVI change
    im1 = axes[0,0].imshow(ndvi_change.values, cmap='RdYlGn', vmin=-0.3, vmax=0.3,
                          extent=[dataset.x.min(), dataset.x.max(),
                                 dataset.y.min(), dataset.y.max()])
    axes[0,0].set_title('NDVI Change (Vegetation)')
    axes[0,0].set_xlabel('Longitude')
    axes[0,0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0,0], label='NDVI Change')
    
    # NDBI change
    im2 = axes[0,1].imshow(ndbi_change.values, cmap='Reds', vmin=-0.2, vmax=0.2,
                          extent=[dataset.x.min(), dataset.x.max(),
                                 dataset.y.min(), dataset.y.max()])
    axes[0,1].set_title('NDBI Change (Urbanization)')
    axes[0,1].set_xlabel('Longitude')
    axes[0,1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[0,1], label='NDBI Change')
    
    # Change magnitude
    change_magnitude = np.sqrt(ndvi_change.values**2 + ndbi_change.values**2)
    im3 = axes[1,0].imshow(change_magnitude, cmap='viridis',
                          extent=[dataset.x.min(), dataset.x.max(),
                                 dataset.y.min(), dataset.y.max()])
    axes[1,0].set_title('Change Magnitude')
    axes[1,0].set_xlabel('Longitude')
    axes[1,0].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1,0], label='Change Magnitude')
    
    # Histogram of changes
    axes[1,1].hist(ndvi_change.values.flatten(), bins=50, alpha=0.7, 
                   label='NDVI Change', color='green')
    axes[1,1].hist(ndbi_change.values.flatten(), bins=50, alpha=0.7, 
                   label='NDBI Change', color='red')
    axes[1,1].set_xlabel('Change Value')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Changes')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    ndvi_loss = np.sum(ndvi_change.values < -0.1)
    ndvi_gain = np.sum(ndvi_change.values > 0.1)
    urban_expansion = np.sum(ndbi_change.values > 0.05)
    
    print(f"📊 Change Detection Results:")
    print(f"  🌱 Vegetation loss pixels: {ndvi_loss}")
    print(f"  🌿 Vegetation gain pixels: {ndvi_gain}")
    print(f"  🏗️ Urban expansion pixels: {urban_expansion}")
    
    return ndvi_change, ndbi_change

# Perform change detection
ndvi_change, ndbi_change = perform_change_detection(sydney_data, start_idx=0, end_idx=-1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Map Visualization

# COMMAND ----------

def create_interactive_map(dataset, df_sample, index='ndvi', time_index=-1):
    """Create an interactive map using Folium"""
    
    print(f"🗺️ Creating interactive map for {index.upper()}...")
    
    # Get the center of the region
    center_lat = float(dataset.y.mean())
    center_lon = float(dataset.x.mean())
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Sample data for performance
    filtered_df = df_sample[df_sample['time'] == df_sample['time'].iloc[time_index]]
    df_map = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
    
    # Color mapping for the index
    if index == 'ndvi':
        colormap = 'RdYlGn'
        vmin, vmax = -0.2, 0.8
    elif index == 'ndwi':
        colormap = 'Blues'
        vmin, vmax = -0.5, 0.5
    elif index == 'ndbi':
        colormap = 'Reds'
        vmin, vmax = -0.3, 0.3
    else:
        colormap = 'viridis'
        vmin, vmax = df_map[index].quantile([0.02, 0.98])
    
    # Add points to map
    for _, row in df_map.iterrows():
        value = row[index]
        
        # Normalize value for color
        normalized_value = (value - vmin) / (vmax - vmin)
        
        # Get color
        if index == 'ndvi':
            if value > 0.5:
                color = 'green'
            elif value > 0.2:
                color = 'yellow'
            else:
                color = 'red'
        elif index == 'ndwi':
            color = 'blue' if value > 0 else 'brown'
        elif index == 'ndbi':
            color = 'red' if value > 0 else 'green'
        else:
            color = 'blue'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"{index.upper()}: {value:.3f}<br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{index.upper()} Values</b></p>
    <p><span style="color:green;">●</span> High (> {vmax*0.7:.2f})</p>
    <p><span style="color:yellow;">●</span> Medium</p>
    <p><span style="color:red;">●</span> Low (< {vmin*0.7:.2f})</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Create interactive map for NDVI
interactive_map = create_interactive_map(sydney_data, df_sydney, index='ndvi', time_index=-1)
interactive_map

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Results and Summary

# COMMAND ----------

def export_analysis_results(dataset, monthly_stats, output_path="/tmp/satellite_analysis/"):
    """Export analysis results to various formats"""
    
    print(f"💾 Exporting results to {output_path}...")
    
    # Create output directory
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # Export monthly statistics
    monthly_stats.to_csv(f"{output_path}/monthly_statistics.csv", index=False)
    
    # Export latest NDVI as GeoTIFF (simplified)
    latest_ndvi = dataset['ndvi'].isel(time=-1)
    
    # Create summary statistics
    summary = {
        'region': dataset.attrs.get('region', 'unknown'),
        'time_range': f"{dataset.time.values[0]} to {dataset.time.values[-1]}",
        'spatial_extent': f"{dataset.x.min().values:.4f}°E to {dataset.x.max().values:.4f}°E, " +
                         f"{dataset.y.min().values:.4f}°N to {dataset.y.max().values:.4f}°N",
        'mean_ndvi_latest': float(latest_ndvi.mean()),
        'std_ndvi_latest': float(latest_ndvi.std()),
        'vegetation_percentage': float((latest_ndvi > 0.3).sum() / latest_ndvi.size * 100),
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save summary
    import json
    with open(f"{output_path}/analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Export completed!")
    print(f"📁 Files saved to: {output_path}")
    
    return summary

# Export results
summary = export_analysis_results(sydney_data, monthly_stats)

# Display summary
print("\n📋 Analysis Summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Analysis: Multi-Region Comparison

# COMMAND ----------

def compare_regions():
    """Compare satellite data across multiple Australian regions"""
    
    print("🌏 Comparing multiple Australian regions...")
    
    regions_to_compare = ['sydney', 'perth', 'kakadu']
    region_summaries = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, region in enumerate(regions_to_compare):
        print(f"📍 Processing {analyzer.regions[region]['name']}...")
        
        # Create sample data
        region_data = analyzer.create_sample_landsat_data(region=region, size=(30, 30), n_timesteps=6)
        region_data = analyzer.calculate_vegetation_indices(region_data)
        
        # Calculate latest NDVI and NDBI
        latest_ndvi = region_data['ndvi'].isel(time=-1)
        latest_ndbi = region_data['ndbi'].isel(time=-1)
        
        # Plot NDVI
        im1 = axes[0, i].imshow(latest_ndvi.values, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0, i].set_title(f'{analyzer.regions[region]["name"]}\nNDVI (Vegetation)')
        axes[0, i].set_xlabel('Longitude Index')
        axes[0, i].set_ylabel('Latitude Index')
        
        # Plot NDBI
        im2 = axes[1, i].imshow(latest_ndbi.values, cmap='Reds', vmin=-0.3, vmax=0.3)
        axes[1, i].set_title(f'NDBI (Built-up)')
        axes[1, i].set_xlabel('Longitude Index')
        axes[1, i].set_ylabel('Latitude Index')
        
        # Calculate summary statistics
        region_summaries[region] = {
            'mean_ndvi': float(latest_ndvi.mean()),
            'mean_ndbi': float(latest_ndbi.mean()),
            'vegetation_cover': float((latest_ndvi > 0.3).sum() / latest_ndvi.size * 100),
            'urban_cover': float((latest_ndbi > 0.1).sum() / latest_ndbi.size * 100)
        }
    
    plt.tight_layout()
    plt.show()
    
    # Create comparison table
    comparison_df = pd.DataFrame(region_summaries).T
    comparison_df.index = [analyzer.regions[r]['name'] for r in comparison_df.index]
    
    print("\n📊 Regional Comparison Summary:")
    print(comparison_df.round(3))
    
    return comparison_df

# Compare regions
regional_comparison = compare_regions()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Sources for Real Implementation
# MAGIC
# MAGIC ### 🛰️ **Satellite Data Sources**
# MAGIC
# MAGIC 1. **Digital Earth Australia (DEA)**
# MAGIC    - URL: https://www.dea.ga.gov.au/
# MAGIC    - Products: Landsat, Sentinel-2, SAR
# MAGIC    - API: DEA OWS, STAC API
# MAGIC
# MAGIC 2. **USGS Earth Explorer**
# MAGIC    - URL: https://earthexplorer.usgs.gov/
# MAGIC    - Products: Landsat, MODIS, ASTER
# MAGIC    - Free registration required
# MAGIC
# MAGIC 3. **Copernicus Open Access Hub**
# MAGIC    - URL: https://scihub.copernicus.eu/
# MAGIC    - Products: Sentinel-1, Sentinel-2, Sentinel-3
# MAGIC    - Free access with registration
# MAGIC
# MAGIC 4. **Google Earth Engine**
# MAGIC    - URL: https://earthengine.google.com/
# MAGIC    - Massive catalog, cloud processing
# MAGIC    - Academic access available
# MAGIC
# MAGIC ### 📊 **Sample Data URLs**
# MAGIC ```python
# MAGIC # Example URLs for real data access
# MAGIC dea_landsat_url = "https://ows.dea.ga.gov.au/"
# MAGIC dea_stac_url = "https://explorer.dea.ga.gov.au/stac"
# MAGIC usgs_api_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
# MAGIC ```
# MAGIC
# MAGIC ### 🔧 **Implementation Notes**
# MAGIC - Replace sample data generation with real data loading
# MAGIC - Use `rasterio`, `xarray`, or `rioxarray` for GeoTIFF loading
# MAGIC - Implement proper cloud masking for Landsat/Sentinel-2
# MAGIC - Add coordinate reference system handling
# MAGIC - Consider memory management for large datasets
