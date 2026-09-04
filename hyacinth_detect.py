"""
Water Hyacinth & Hypoxia Detection - Initial Data Pipeline
"""

import ee
import geemap
import matplotlib.pyplot as plt
import urllib.request
from ee_utils import EE_PROJECT_ID

# ---- STEP 1: Authenticate & Initialize ----
ee.Authenticate()
ee.Initialize(project=EE_PROJECT_ID)

# ---- STEP 2: Area of Interest ----
#aoi = ee.Geometry.Point([93.85, 24.55]).buffer(3000)  # Loktak Lake, Manipur
#aoi = ee.Geometry.Point([73.7749, 18.5486]).buffer(2000)  # Pashan Lake, Pune
aoi = ee.Geometry.Point([73.8567, 18.5204]).buffer(3000)  # Mula-Mutha River, Pune
# ---- STEP 3: Pull Sentinel-2 imagery ----
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2025-01-01', '2025-03-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .sort('CLOUDY_PIXEL_PERCENTAGE')
)

image = collection.first().clip(aoi)
print("Image date:", image.date().format().getInfo())
print("Cloud %:", image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo())

# ---- STEP 4: Compute Indices ----
ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

# ---- STEP 5: Save thumbnail images ----
rgb_thumb_url = image.getThumbURL({
    'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000,
    'region': aoi, 'dimensions': 512
})
ndvi_thumb_url = ndvi.getThumbURL({
    'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green'],
    'region': aoi, 'dimensions': 512
})
ndwi_thumb_url = ndwi.getThumbURL({
    'min': -1, 'max': 1, 'palette': ['brown', 'white', 'blue'],
    'region': aoi, 'dimensions': 512
})

urllib.request.urlretrieve(rgb_thumb_url, 'true_color.png')
urllib.request.urlretrieve(ndvi_thumb_url, 'ndvi_output.png')
urllib.request.urlretrieve(ndwi_thumb_url, 'ndwi_output.png')

print("Saved: true_color.png, ndvi_output.png, ndwi_output.png")

# ---- STEP 6: Side-by-side comparison image ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, path, title in zip(
    axes,
    ['true_color.png', 'ndvi_output.png', 'ndwi_output.png'],
    ['True Color', 'NDVI (Vegetation/Hyacinth)', 'NDWI (Water)']
):
    img = plt.imread(path)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('comparison_output.png', dpi=150)
plt.show()

print("Done. comparison_output.png is ready for your slide.")