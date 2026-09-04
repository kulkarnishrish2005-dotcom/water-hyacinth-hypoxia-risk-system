# Water Hyacinth Detection System (Phase 1)

## Overview

This system uses Google Earth Engine (GEE) and Sentinel-2 satellite imagery to detect water hyacinth in inland water bodies across the Indian subcontinent. It is designed to be site-agnostic and extensible to many water bodies rather than tied to any one lake or region.

Phase 1 final ontology:
- 0 = Water
- 1 = Water Hyacinth
- 2 = Other Vegetation

Lotus-like vegetation is treated as class 2 for the current Phase 1 model.

## Two modes in Phase 1

### Mode A — Prototype / demo mode
This mode remains available for running the app without manually collected labels. It uses Dynamic World labels as a proxy training source.

Important:
- Dynamic World labels are used as a prototype training source only.
- They are not treated as authoritative species-level ground truth.
- The prototype accuracy is a proxy estimate only and must not be described as scientific validation.

### Mode B — Scientific validation mode
This is the authoritative Phase 1 evaluation path.

It uses human-labeled ground-truth CSV data and spatial hold-out splits such as:
- TRAIN_SITES / VALIDATION_SITES
- TRAIN_REGIONS / VALIDATION_REGIONS
- TRAIN_SECTORS / VALIDATION_SECTORS

This mode is the only one used for scientific accuracy reporting.

## Current Phase 1 implementation

### ✅ Core features
- Water/land mask using JRC Global Surface Water + NDWI
- NDVI and NDWI calculation
- NDVI texture via local neighborhood standard deviation
- Distance-to-shore from land mask
- Random Forest classifier using:
  - NDVI
  - NDWI
  - ndvi_texture
  - distance_to_shore
- Spatially-aware validation framework based on site/region/sector metadata
- Multi-site ground-truth schema support
- Prototype Dynamic World workflow retained for demo operation

### ✅ Validation framework
The repository supports generic validation splits such as:
- TRAIN_SITES / VALIDATION_SITES
- TRAIN_REGIONS / VALIDATION_REGIONS
- TRAIN_SECTORS / VALIDATION_SECTORS

This is intentionally not tied to Loktak, Vembanad, Pune, or any other individual lake.

## Project structure

```text
D:\Project - I\
├── ee_utils.py
├── app.py
├── hyacinth_detect.py
├── training_data_sample.py
├── data/
│   └── ground_truth_template.csv
├── README.md
├── QUICKSTART.md
├── requirements.txt
└── generated output images
```

## Ground-truth schema

The real validation dataset should follow the schema below:

site,region,country,class,label,lat,lon,date,source,sample_type,sector,notes

For example, the field values should be real and specific to the water body and region, but the system itself does not assume any one site.

## Training data and validation guidance

### Important
- The file `training_data_sample.py` is a template and usage example only.
- It is not scientific ground truth.
- Real validation data must be collected separately and kept as a dataset file.

### Required Phase 1 classes
- 0 = Water
- 1 = Water Hyacinth
- 2 = Other Vegetation

Do not use a 4-class ontology in this phase.

## Validation requirements

The validation workflow must obey these rules:
- training and validation must be spatially independent whenever possible
- validation data must never be used for model fitting
- class labels must be checked before modeling
- sector overlaps are rejected
- site overlaps are rejected
- region overlaps are rejected
- empty training/validation sets are rejected
- unknown classes are rejected
- duplicate coordinates and duplicate sample IDs are rejected

## Scientific use policy

The project must not claim scientific accuracy without real validation data from independent water bodies or sectors.

Dynamic World labels are used as a prototype training source and are not treated as authoritative species-level ground truth.

The repository’s template/example data is not enough for final validation and must not be treated as ground truth.

## Known limitations

- Validation requires real, labeled data from actual sites
- Spatial autocorrelation can remain even after site or sector splitting
- Mixed pixels can reduce label purity at Sentinel-2 resolution
- Temporal mismatch can occur if the imagery date and label date differ
- Class imbalance can distort observed accuracy
- Label quality and manual interpretation error remain important
- The system includes Phase 1 and Phase 2 prototype features
- A stronger experiment should eventually use cross-site and cross-region generalization
- **Phase 2 limitations**: Proxy variables are NOT direct in-situ measurements
- Dissolved oxygen is NOT measured directly by satellites
- HHRI weights are configurable but require empirical calibration for scientific validation
- Sentinel-2 lacks thermal bands; temperature estimates require Landsat integration

## Next steps

1. Create a real ground-truth CSV for the first validation site(s)
2. Assign train and validation sectors/sites
3. Run the model on those splits
4. Review confusion matrix, overall accuracy, and per-class metrics
5. Repeat with additional sites to test transferability across India

### Random Forest Parameters
- Trees: 50
- Seed: 42
- Split: 70/30 train/test (for prototype mode)

## Next Steps (Future Phases)

### Phase 1 Enhancements:
- [ ] Automate training data collection (active learning)
- [ ] Wire temporal NDVI difference into main pipeline
- [ ] Cross-site and cross-region validation generalization

### Phase 2 Enhancements:
- [ ] Empirically calibrate HHRI weights using independent hypoxia measurements
- [ ] Integrate Landsat thermal data for actual surface temperature
- [ ] Build supervised hypoxia classifier with labeled dissolved oxygen data
- [ ] Implement temporal/seasonal hypoxia risk tracking
- [ ] Cross-validate Phase 1 hyacinth features contribution to hypoxia prediction
- [ ] Multi-date time series analysis
- [ ] Export classified rasters and risk maps (GeoTIFF)
- [ ] Area statistics and spread-rate metrics
- [ ] Integration with field validation data

### Scientific Integrity:
- Phase 2 proxy variables are explicitly documented as estimated, not measured
- HHRI weights are provisional configurable parameters, not empirically validated
- No causation claimed from satellite correlation alone
- Dissolved oxygen proxy (DO_proxy) is a placeholder, not a measured value

## Credits

- **Sentinel-2**: Copernicus Sentinel data (ESA)
- **JRC GSW**: EC Joint Research Centre
- **Earth Engine**: Google Earth Engine
- **Geopy**: OpenStreetMap contributors (Nominatim)

## License

This is an academic project for educational purposes.

---

**Author**: B.Tech Project Team  
**Date**: August 2026  
**Phase**: Phase 1 Complete + Phase 2 Prototype
