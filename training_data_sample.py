"""
Template usage example for Phase 1 ground-truth collection.

This file is a template and usage example only. It is not scientific ground truth.
The actual validation dataset must be compiled separately and must be site-agnostic,
not tailored to any single lake or region.

Phase 1 classes:
- 0 = Water
- 1 = Water Hyacinth
- 2 = Other Vegetation

Lotus-like vegetation is treated as class 2 in the Phase 1 model.

Required schema for a real ground-truth dataset:
site,region,country,class,label,lat,lon,date,source,sample_type,sector,notes

The project is intended to work across multiple water bodies and regions in the Indian
subcontinent, not just one named lake.
"""

# Example structure only — not scientific ground truth.
# Replace with your real data file and keep it outside code if needed.
TEMPLATE_GROUND_TRUTH = [
    {
        'site': 'example_water_body',
        'region': 'example_region',
        'country': 'India',
        'class': 0,
        'label': 'Water',
        'lat': 0.0,
        'lon': 0.0,
        'date': '2025-02-15',
        'source': 'Sentinel-2 visual interpretation',
        'sample_type': 'point',
        'sector': 'north',
        'notes': 'Template example only; not real data.'
    },
    {
        'site': 'example_water_body',
        'region': 'example_region',
        'country': 'India',
        'class': 1,
        'label': 'Water Hyacinth',
        'lat': 0.0,
        'lon': 0.0,
        'date': '2025-02-15',
        'source': 'Sentinel-2 visual interpretation',
        'sample_type': 'point',
        'sector': 'center',
        'notes': 'Template example only; not real data.'
    },
    {
        'site': 'example_water_body',
        'region': 'example_region',
        'country': 'India',
        'class': 2,
        'label': 'Other Vegetation',
        'lat': 0.0,
        'lon': 0.0,
        'date': '2025-02-15',
        'source': 'Sentinel-2 visual interpretation',
        'sample_type': 'point',
        'sector': 'south',
        'notes': 'Template example only; not real data.'
    }
]


if __name__ == '__main__':
    print('Template file only. Do not treat this as scientific ground truth.')
