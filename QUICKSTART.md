# Quick Start Guide - Water Hyacinth Detection System

## Overview

This project is designed to be site-agnostic and scalable across multiple water bodies and regions in the Indian subcontinent. It is not tied to a single water body or city.

Phase 1 final ontology:
- 0 = Water
- 1 = Water Hyacinth
- 2 = Other Vegetation

Lotus-like vegetation is treated as class 2 for the current Phase 1 model.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Authenticate Earth Engine

```bash
earthengine authenticate
```

## 3. Set your project ID

Open `ee_utils.py` and ensure your Earth Engine project is configured correctly.

## 4. Launch the app

```bash
python -m streamlit run app.py
```

## 5. Use the app

The app can be used for:
- arbitrary water-body discovery by city/place
- analysis of any selected water body
- prototype Dynamic World training for demo operation
- scientific validation using site/region/sector-based splits with human-labeled ground truth

## Prototype mode vs scientific validation mode

### Prototype / demo mode
- Uses Dynamic World labels as a proxy training source
- Useful for app demonstration and testing
- Must be labeled clearly as "Prototype / proxy evaluation — not scientific validation"
- Not an authoritative ground-truth dataset

### Scientific validation mode
- Uses the master CSV ground-truth dataset
- Requires train/validation splits by site, region, or sector
- Trains Random Forest only on the scientific training set
- Evaluates strictly on the hold-out validation set
- Reports metrics as scientific validation results only

## 6. Ground-truth data requirements

The repository includes a template at:

```text
data/ground_truth_template.csv
```

Required columns:
- site
- region
- country
- class
- label
- lat
- lon
- date
- source
- sample_type
- sector
- notes

Important:
- The template contains no fabricated coordinates.
- `training_data_sample.py` is a template/example only and is not scientific ground truth.

## 7. Validation design

Use a spatially independent validation design whenever possible:
- TRAIN_SITES / VALIDATION_SITES
- TRAIN_REGIONS / VALIDATION_REGIONS
- TRAIN_SECTORS / VALIDATION_SECTORS

The validation split must be configured before model fitting. Do not mix training and validation labels.

## 8. Example validation configuration

```python
VALIDATION_CONFIG = {
    'site': None,
    'region': None,
    'country': None,
    'train_sites': ['site_a', 'site_b'],
    'validation_sites': ['site_c'],
    'train_regions': [],
    'validation_regions': [],
    'train_sectors': ['north', 'center'],
    'validation_sectors': ['south', 'east'],
    'classifier': 'RandomForest',
    'features': ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']
}
```

## 9. Validation rules

- training data must include classes 0, 1, and 2
- validation data must not contain unknown classes
- training and validation sectors must not overlap
- training and validation sites must not overlap
- training and validation regions must not overlap
- duplicate coordinates and duplicate sample IDs are rejected
- validation data must not be used to fit the classifier
- no site-specific logic should be required in the model pipeline

## 10. Scientific caution

Accuracy claims should be made only after real validation data is available.
Dynamic World labels are a prototype training source and do not replace human-labeled ground truth.
The repository template is not enough to claim model performance.

## Phase 2: Hypoxia Risk Assessment & HHRI

The system includes Phase 2 capabilities:
- Chlorophyll-a proxy (Gitelson red-edge formulation)
- Turbidity proxy (Red/NIR band ratio)
- Surface temperature estimate (with documented Sentinel-2 limitations)
- Hypoxia risk model (categorical LOW/MODERATE/HIGH using HHRI index)
- Configurable HHRI weights and thresholds

### Scientific Distinction for Phase 2:
- Proxy variables are NOT direct in-situ measurements
- Dissolved oxygen is NOT measured directly by satellites
- HHRI weights are configurable, but require empirical calibration for scientific validation
- Hypoxia risk is a proxy/rule-based estimate, not a causally validated model

## 11. Current Project Scope

### Phase 1:
- NDVI, NDWI, NDVI Texture, Distance to Shore
- 3-class ontology: 0=Water, 1=Water Hyacinth, 2=Other Vegetation (Lotus-like)
- Random Forest classifier with spatial hold-out validation

### Phase 2:
- Chlorophyll-a proxy, Turbidity proxy, Temperature estimate
- Hypoxia Risk Model using HHRI composite index
- Phase 1 hyacinth output directly feeds Phase 2 risk assessment
