#!/usr/bin/env python
"""Runtime validation of the complete pipeline."""
import sys, os
sys.path.insert(0, r'D:\Project - I')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import ee
try:
    ee.Initialize(project='project-i-506007')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='project-i-506007')

from ee_utils import run_full_pipeline

print("Starting pipeline...")
result = run_full_pipeline(
    lat=18.52, lon=73.85,
    start_date='2025-02-10', end_date='2025-02-24',
    max_cloud=5,
    labelled_points=None,
)
print("Pipeline completed")
print("Keys:", list(result.keys()))

# Debug: check classification result for errors
classification_result = result.get('classification', {})
print("classification keys:", list(classification_result.keys()))
print("classification error:", classification_result.get('error'))
print("classified_image:", classification_result.get('classified_image'))
print("thumbnail_url:", classification_result.get('thumbnail_url'))

# Raw area EE objects
print("hyacinth_area_ha:", result.get('hyacinth_area_ha'))
print("water_area_ha:", result.get('water_area_ha'))
print("hhri_mean:", result.get('hhri_mean'))

print("SUCCESS - no NameError/TypeError/AttributeError")
