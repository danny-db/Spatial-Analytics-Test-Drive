import os
from databricks import sql
from databricks.sdk.core import Config
import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
import numpy as np

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# Databricks config
cfg = Config()

# Query the SQL warehouse with Service Principal credentials
def sql_query_with_service_principal(query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

# Query the SQL warehouse with the user credentials
def sql_query_with_user_token(query: str, user_token: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
        access_token=user_token
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

# Streamlit app configuration
st.set_page_config(
    page_title="VicMap Road Infrastructure Visualization",
    page_icon="🛣️",
    layout="wide"
)

@st.cache_data
def query_road_infra_data(table_name: str, limit: int, user_token: str = None) -> pd.DataFrame:
    """Query road infrastructure data from Databricks table using ST_X and ST_Y functions"""
    query = f"""
    SELECT st_x(st_geomfromtext(geom_4326)) AS longitude, st_y(st_geomfromtext(geom_4326)) AS latitude, * 
    FROM {table_name} 
    WHERE geom_4326 IS NOT NULL
    LIMIT {limit}
    """
    
    if user_token:
        return sql_query_with_user_token(query, user_token)
    else:
        return sql_query_with_service_principal(query)

@st.cache_data
def query_road_type_stats(table_name: str, user_token: str = None) -> pd.DataFrame:
    """Query road type statistics"""
    query = f"""
    SELECT 
        FTYPE_CODE,
        COUNT(*) as feature_count,
        COUNT(DISTINCT h3) as h3_cells
    FROM {table_name}
    WHERE FTYPE_CODE IS NOT NULL
    AND geom_4326 IS NOT NULL
    GROUP BY FTYPE_CODE
    ORDER BY feature_count DESC
    LIMIT 20
    """
    
    if user_token:
        return sql_query_with_user_token(query, user_token)
    else:
        return sql_query_with_service_principal(query)

@st.cache_data
def query_h3_aggregation(table_name: str, user_token: str = None) -> pd.DataFrame:
    """Query H3 cell aggregation data"""
    query = f"""
    SELECT 
        h3,
        COUNT(*) as feature_count,
        AVG(st_x(st_geomfromtext(geom_4326))) as avg_longitude,
        AVG(st_y(st_geomfromtext(geom_4326))) as avg_latitude,
        COLLECT_SET(FTYPE_CODE) as feature_types
    FROM {table_name}
    WHERE h3 IS NOT NULL
    AND geom_4326 IS NOT NULL
    GROUP BY h3
    HAVING COUNT(*) > 1
    ORDER BY feature_count DESC
    LIMIT 100
    """
    
    if user_token:
        return sql_query_with_user_token(query, user_token)
    else:
        return sql_query_with_service_principal(query)

# Main app interface
st.title("🛣️ VicMap Road Infrastructure Visualization")
st.markdown("Interactive visualization of Victorian road infrastructure data using Leafmap")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Table configuration
table_name = st.sidebar.text_input(
    "Table Name:",
    value="vrstdptrainingcat01.dannywong.silver_road_infra",
    placeholder="catalog.schema.table",
    help="Enter the Unity Catalog table name"
)

# Data limit slider
data_limit = st.sidebar.slider(
    "Number of records to visualize:",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100
)

# Authentication method selection
auth_method = st.sidebar.radio(
    "Authentication Method:",
    ["User Token", "Service Principal"],
    help="Choose between user token or service principal authentication"
)

# Map configuration
st.sidebar.subheader("Map Configuration")
map_style = st.sidebar.selectbox(
    "Base Map Style:",
    ["OpenStreetMap", "CartoDB.Positron", "CartoDB.DarkMatter", "Stamen.Terrain"]
)

# Layer options
st.sidebar.subheader("Layer Options")
show_road_points = st.sidebar.checkbox("Show Road Infrastructure Points", value=True)
show_road_heatmap = st.sidebar.checkbox("Show Infrastructure Heatmap", value=True)
show_h3_aggregation = st.sidebar.checkbox("Show H3 Cell Aggregation", value=False)
show_feature_types = st.sidebar.checkbox("Show Feature Type Distribution", value=True)

# Feature type filter
feature_type_filter = st.sidebar.text_input(
    "Filter by Feature Type (optional):",
    placeholder="e.g., road_end, int_nosignal",
    help="Enter a specific FTYPE_CODE to filter results"
)

try:
    # Extract user access token from the request headers
    user_token = st.context.headers.get('X-Forwarded-Access-Token')
    
    # Determine which authentication method to use
    use_user_token = auth_method == "User Token" and user_token is not None
    
    # Query data based on authentication method
    with st.spinner("Fetching data from Databricks..."):
        if use_user_token:
            df = query_road_infra_data(table_name, data_limit, user_token)
            if show_feature_types:
                type_stats_df = query_road_type_stats(table_name, user_token)
            if show_h3_aggregation:
                h3_df = query_h3_aggregation(table_name, user_token)
        else:
            df = query_road_infra_data(table_name, data_limit)
            if show_feature_types:
                type_stats_df = query_road_type_stats(table_name)
            if show_h3_aggregation:
                h3_df = query_h3_aggregation(table_name)
    
    if not df.empty:
        # Apply feature type filter if specified
        if feature_type_filter:
            df = df[df['FTYPE_CODE'].str.contains(feature_type_filter, case=False, na=False)]
            if df.empty:
                st.warning(f"No records found matching feature type: {feature_type_filter}")
        
        # Display authentication status
        auth_status = "🔑 User Token" if use_user_token else "🤖 Service Principal"
        st.sidebar.success(f"Connected via: {auth_status}")
        
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Feature Types", df['FTYPE_CODE'].nunique())
        with col3:
            st.metric("Unique UFI Count", df['UFI'].nunique())
        with col4:
            st.metric("Unique H3 Cells", df['h3'].nunique())
        
        # Create Leafmap visualization
        st.subheader("Interactive Map")
        
        # Initialize map centered on Victoria
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = leafmap.Map(
            center=[center_lat, center_lon], 
            zoom=10,
            layers_control=True,
            draw_control=False,
            measure_control=False,
            fullscreen_control=True
        )
        
        # Add base map
        if map_style != "OpenStreetMap":
            m.add_basemap(map_style)
        
        # Add road infrastructure points layer
        if show_road_points and not df.empty:
            #df_sample = df.head(50)  # Limit for testing
            
            for idx, row in df.iterrows():
                try:
                    if pd.notna(row['longitude']) and pd.notna(row['latitude']):
                        popup_text = f"Type: {row['FTYPE_CODE']}<br>UFI: {row['UFI']}"
                        
                        m.add_marker(
                            location=[float(row['latitude']), float(row['longitude'])],
                            popup=popup_text,
                            tooltip=str(row['FTYPE_CODE'])
                        )
                except Exception as e:
                    st.write(f"Error adding marker for row {idx}: {str(e)}")
                    continue
            # # Color code by feature type
            # feature_types = df['FTYPE_CODE'].unique()
            # colors = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 'lightred', 
            #          'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
            #          'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
            
            # for i, ftype in enumerate(feature_types[:len(colors)]):
            #     ftype_data = df[df['FTYPE_CODE'] == ftype].copy()
            #     if not ftype_data.empty:
            #         # Simplified popup with string conversion
            #         ftype_data['popup'] = ftype_data.apply(
            #             lambda row: f"Type: {str(row['FTYPE_CODE'])}<br>UFI: {str(row['UFI'])}<br>PFI: {str(row['PFI'])}", 
            #             axis=1
            #         )
                    
            #         m.add_points_from_xy(
            #             ftype_data,
            #             x="longitude",
            #             y="latitude",
            #             popup="popup",
            #             layer_name=f"Road Features - {ftype}",
            #             color=colors[i % len(colors)],
            #             icon="road",
            #             spin=False
            #         )
        
        # Add heatmap layer
        if show_road_heatmap and not df.empty:
            # Create heatmap data
            heat_data = [[row['latitude'], row['longitude'], 1] 
                        for _, row in df.iterrows()]
            
            m.add_heatmap(
                heat_data,
                layer_name="Infrastructure Density Heatmap",
                radius=10,
                blur=8,
                min_opacity=0.3
            )
        
        # Add H3 aggregation layer
        if show_h3_aggregation and 'h3_df' in locals() and not h3_df.empty:
            for _, row in h3_df.head(50).iterrows():
                popup_text = f"""
                H3 Cell: {row['h3']}<br>
                Feature Count: {row['feature_count']}<br>
                Feature Types: {str(row['feature_types'])[:100]}...
                """
                
                # Size marker based on feature count
                marker_size = min(max(row['feature_count'] / 10, 5), 20)
                
                m.add_marker(
                    location=[row['avg_latitude'], row['avg_longitude']],
                    popup=popup_text,
                    tooltip=f"H3: {row['h3']} ({row['feature_count']} features)",
                    icon=leafmap.plugins.BeautifyIcon(
                        icon="th",
                        iconShape="circle",
                        borderColor="purple",
                        backgroundColor="lightpurple"
                    )
                )
        
        # Display the map
        m.to_streamlit(width=None, height=600, add_layer_control=True)
        
        # Feature type analysis
        if show_feature_types and 'type_stats_df' in locals() and not type_stats_df.empty:
            st.subheader("Feature Type Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Feature Type Distribution**")
                st.bar_chart(type_stats_df.set_index('FTYPE_CODE')['feature_count'])
            
            with col2:
                st.write("**H3 Cell Coverage by Feature Type**")
                st.bar_chart(type_stats_df.set_index('FTYPE_CODE')['h3_cells'])
            
            # Feature type statistics table
            st.subheader("Feature Type Statistics")
            st.dataframe(
                type_stats_df,
                column_config={
                    "FTYPE_CODE": "Feature Type",
                    "feature_count": st.column_config.NumberColumn("Feature Count", format="%d"),
                    "h3_cells": st.column_config.NumberColumn("H3 Cells", format="%d"),
                },
                use_container_width=True
            )
        
        # Geographic distribution analysis
        st.subheader("Geographic Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Latitude Distribution**")
            lat_hist = pd.cut(df['latitude'], bins=20).value_counts().sort_index()
            lat_hist.index = lat_hist.index.astype(str)
            st.bar_chart(lat_hist)
        
        with col2:
            st.write("**Longitude Distribution**")
            lon_hist = pd.cut(df['longitude'], bins=20).value_counts().sort_index()
            lon_hist.index = lon_hist.index.astype(str)
            st.bar_chart(lon_hist)
        
        # Show raw data option
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data Sample")
            st.dataframe(df.head(100), use_container_width=True)
            
    else:
        st.warning("No data found in the specified table or matching the filter criteria.")
        
except Exception as e:
    st.error(f"Error connecting to Databricks or querying data: {str(e)}")
    st.info("Please check your configuration and permissions.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Leafmap, and Databricks - VicMap Road Infrastructure Data")
