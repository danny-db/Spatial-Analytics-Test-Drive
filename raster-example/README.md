# If security is a concern...

1. Install all dependencies for `geopandas` (==1.1.1) and `rasterio` (==1.4.3)libraries as `.whl` files.

    Each Python wheel follows a naming convention: {dist}-{version}(-{build})?-{python.version}-{os_platform}.whl
    ```
    python -m pip download --only-binary :all: --dest . --no-cache <package_name>
    ```

2. Upload `.whl` files to Databricks workspace using the Databricks UI.

# Notebook Examples

* [raster-example.ipynb](./raster-example.ipynb)

# Reference Docs

* https://www.databricks.com/notebooks/geopandas-notebook.html

* https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
