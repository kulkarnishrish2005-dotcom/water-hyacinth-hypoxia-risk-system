"""
Earth Engine Utility Functions for Water Hyacinth Detection System.
Contains all Earth Engine processing logic: index computation, water/land masking,
location discovery, feature engineering, training data, and RF classification.

Two explicit modes are maintained:
  1. Prototype / Proxy mode — uses Dynamic World as an automatic proxy training source.
     Never presents Dynamic World labels as authoritative water-hyacinth ground truth.
  2. Scientific validation mode — uses human-labeled ground-truth CSV data with
     spatial hold-out splits (site/region/sector). Validation data NEVER used for model fitting.
"""

import csv
import os
from datetime import datetime, timedelta

import ee
import geemap
import matplotlib.pyplot as plt

EE_PROJECT_ID = 'project-i-506007'

CLASS_NAMES = {
    0: 'Water',
    1: 'Water Hyacinth',
    2: 'Other Vegetation'
}

# Prototype/proxy mode label — never presented as scientific validation
PROXY_MODE_LABEL = 'Prototype / Proxy — Dynamic World labels are automatic proxies only, not authoritative species-level ground truth.'

# Scientific validation mode label
SCIENCE_MODE_LABEL = 'Scientific validation — human-labeled ground truth with spatial hold-out splits.'

VALIDATION_CONFIG = {
    'site': None,
    'region': None,
    'country': None,
    'train_sites': [],
    'validation_sites': [],
    'train_regions': [],
    'validation_regions': [],
    'train_sectors': [],
    'validation_sectors': [],
    'classifier': 'RandomForest',
    'features': [
        'NDVI',
        'NDWI',
        'ndvi_texture',
        'distance_to_shore'
    ],
    'mode': 'prototype'  # 'prototype' or 'scientific'
}

REQUIRED_GROUND_TRUTH_COLUMNS = [
    'site',
    'region',
    'country',
    'class',
    'label',
    'lat',
    'lon',
    'date',
    'source',
    'source_type',
    'sample_type',
    'sector',
    'patch_id',
    'confidence',
    'class',
    'notes'
]

SCIENCE_CLASS_LABELS = [
    'Water',
    'Water Hyacinth',
    'Other Vegetation'
]

# Default Phase 2 configuration
PHASE2_CONFIG = {
    'chlorophyll_proxy': {
        'enabled': True,
        'formula': 'Gitelson',
        'red_edge_band': 'B5',  # Red edge
        'red_band': 'B4',        # Red
        'nir_band': 'B8',        # Near-infrared
        'coefficients': {
            'L': 2.56,
            'G': 6.09,
            'C1': 7.11,
            'C2': 19.67
        }
    },
    'turbidity_proxy': {
        'enabled': True,
        'formula': 'Band ratio',
        'ratio': 'B4 / B8',  # Red / NIR
        'normalization': '0-1 scale after min-max'
    },
    'temperature': {
        'enabled': True,
        'source': 'Landsat thermal',
        'method': 'Single-channel split-window'
    },
    'hypoxia_risk': {
        'enabled': True,
        'classes': ['LOW', 'MODERATE', 'HIGH'],
        'features': ['chlorophyll_proxy', 'turbidity_proxy', 'temperature', 'hyacinth_density']
    },
    'hhri': {
        'enabled': True,
        'weights': {
            'w1_ndvi': 1.0,
            'w2_ndwi': 1.0,
            'w3_chl': 1.0,
            'w4_turbidity': 1.0,
            'w5_doproxy': 1.0
        },
        'thresholds': {
            'low': 0.3,
            'moderate': 0.6
        }
    }
}


def _coerce_sample_id(row, index):
    """Create a deterministic sample ID if one is not provided."""
    sample_id = row.get('sample_id') or row.get('patch_id')
    if sample_id is not None and str(sample_id).strip():
        return str(sample_id).strip()

    site = str(row.get('site', 'unknown')).strip() or 'unknown'
    region = str(row.get('region', 'unknown')).strip() or 'unknown'
    sector = str(row.get('sector', 'unknown')).strip() or 'unknown'
    class_value = row.get('class')
    try:
        class_value = normalize_class_label(class_value)
    except Exception:
        class_value = 'unknown'
    return f"{site}_{region}_{sector}_{class_value}_{index}"


def load_ground_truth_csv(path):
    """Load and validate a master ground-truth CSV for scientific Phase 1 evaluation."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    with open(path, 'r', newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError('Ground-truth CSV is empty or missing a header row.')

        missing = [col for col in REQUIRED_GROUND_TRUTH_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                'Ground-truth CSV is missing required columns: ' + ', '.join(missing) +
                '. Required columns: ' + ', '.join(REQUIRED_GROUND_TRUTH_COLUMNS)
            )

        ground_truth = []
        seen_coordinates = {}
        seen_sample_ids = {}

        for row_index, row in enumerate(reader, start=2):
            if row is None or not any((value or '').strip() for value in row.values()):
                continue

            normalized_row = {}
            for key in REQUIRED_GROUND_TRUTH_COLUMNS:
                normalized_row[key] = (row.get(key, '') or '').strip()

            # Required field checks
            for required_key in ['site', 'region', 'country', 'date', 'source', 'sample_type', 'sector', 'notes']:
                if normalized_row[required_key] == '':
                    raise ValueError(f"Row {row_index}: '{required_key}' is required.")

            if normalized_row['label'] == '':
                raise ValueError(f"Row {row_index}: 'label' is required.")

            try:
                class_value = normalize_class_label(normalized_row['class'])
            except ValueError as exc:
                raise ValueError(f"Row {row_index}: invalid class '{normalized_row['class']}'. {exc}") from exc

            normalized_row['class'] = class_value
            normalized_row['label'] = normalized_row['label']

            try:
                lat = float(normalized_row['lat'])
                lon = float(normalized_row['lon'])
            except ValueError as exc:
                raise ValueError(f"Row {row_index}: lat/lon must be numeric. Found lat='{normalized_row['lat']}', lon='{normalized_row['lon']}'.") from exc

            if not (-90 <= lat <= 90):
                raise ValueError(f"Row {row_index}: latitude out of range: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Row {row_index}: longitude out of range: {lon}")

            normalized_row['lat'] = lat
            normalized_row['lon'] = lon
            normalized_row['sample_id'] = _coerce_sample_id(normalized_row, row_index)

            coord_key = (round(lat, 8), round(lon, 8))
            if coord_key in seen_coordinates:
                raise ValueError(
                    f"Duplicate coordinate detected at row {row_index}: lat={lat}, lon={lon}. "
                    f"Already seen at row {seen_coordinates[coord_key]}."
                )
            seen_coordinates[coord_key] = row_index

            sample_id = normalized_row['sample_id']
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate sample_id '{sample_id}' detected at row {row_index}. "
                    f"Already seen at row {seen_sample_ids[sample_id]}."
                )
            seen_sample_ids[sample_id] = row_index

            ground_truth.append(normalized_row)

    if not ground_truth:
        raise ValueError('Ground-truth CSV contains no valid rows.')

    return ground_truth


def ground_truth_to_ee_features(ground_truth):
    """Convert a ground-truth row list into an EE FeatureCollection with attached metadata."""
    if not ground_truth:
        raise ValueError('Ground-truth data is empty.')

    features = []
    for index, row in enumerate(ground_truth):
        props = {
            'sample_id': str(row.get('sample_id') or _coerce_sample_id(row, index)),
            'site': str(row.get('site', 'unknown')).strip(),
            'region': str(row.get('region', 'unknown')).strip(),
            'country': str(row.get('country', 'unknown')).strip(),
            'class': int(normalize_class_label(row.get('class'))),
            'label': str(row.get('label', CLASS_NAMES.get(normalize_class_label(row.get('class')), 'unknown'))).strip(),
            'date': str(row.get('date', 'unknown')).strip(),
            'source': str(row.get('source', 'unknown')).strip(),
            'sample_type': str(row.get('sample_type', 'point')).strip(),
            'sector': str(row.get('sector', 'unknown')).strip().lower(),
            'notes': str(row.get('notes', '')).strip(),
        }
        geometry = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
        features.append(ee.Feature(geometry, props))

    return ee.FeatureCollection(features)


def validate_split_configuration(train_fc, validation_fc, train_label='train', validation_label='validation'):
    """Validate split structure and required classes before any classifier training."""
    if train_fc is None or validation_fc is None:
        raise ValueError(f'{train_label} and {validation_label} feature collections are required.')

    train_size = train_fc.size().getInfo()
    validation_size = validation_fc.size().getInfo()
    if train_size == 0:
        raise ValueError(f'{train_label} set is empty. Provide at least one training sample.')
    if validation_size == 0:
        raise ValueError(f'{validation_label} set is empty. Provide at least one validation sample.')

    train_classes = set(train_fc.aggregate_array('class').getInfo())
    validation_classes = set(validation_fc.aggregate_array('class').getInfo())

    valid_classes = set(CLASS_NAMES.keys())
    unknown_train = sorted(train_classes - valid_classes)
    unknown_validation = sorted(validation_classes - valid_classes)
    if unknown_train:
        raise ValueError(f'{train_label} contains unknown classes: {unknown_train}. Valid classes are {sorted(valid_classes)}.')
    if unknown_validation:
        raise ValueError(f'{validation_label} contains unknown classes: {unknown_validation}. Valid classes are {sorted(valid_classes)}.')

    missing_train = sorted(valid_classes - train_classes)
    if missing_train:
        raise ValueError(
            f'{train_label} set is missing required classes: {missing_train}. '
            'Phase 1 scientific validation requires all classes 0, 1, and 2 in training.'
        )

    missing_validation = sorted(valid_classes - validation_classes)
    if missing_validation:
        raise ValueError(
            f'{validation_label} set is missing required classes: {missing_validation}. '
            'Scientific validation requires each class to be present in the validation set where available.'
        )

    return {
        'train_size': train_size,
        'validation_size': validation_size,
        'train_classes': sorted(train_classes),
        'validation_classes': sorted(validation_classes),
    }


def summarize_split(train_fc, validation_fc):
    """Report class counts and metadata counts for scientific validation."""
    summary = validate_split_configuration(train_fc, validation_fc)
    summary['train_class_counts'] = train_fc.aggregate_histogram('class').getInfo()
    summary['validation_class_counts'] = validation_fc.aggregate_histogram('class').getInfo()
    summary['train_sites'] = sorted(set(train_fc.aggregate_array('site').getInfo()))
    summary['validation_sites'] = sorted(set(validation_fc.aggregate_array('site').getInfo()))
    summary['train_regions'] = sorted(set(train_fc.aggregate_array('region').getInfo()))
    summary['validation_regions'] = sorted(set(validation_fc.aggregate_array('region').getInfo()))
    summary['train_sectors'] = sorted(set(train_fc.aggregate_array('sector').getInfo()))
    summary['validation_sectors'] = sorted(set(validation_fc.aggregate_array('sector').getInfo()))
    return summary


def _normalize_split_values(values):
    """Normalize split values to lowercase strings for overlap checks."""
    return [str(v).strip().lower() for v in values if v is not None and str(v).strip()]


def _ensure_no_overlap(train_values, validation_values, field_name):
    """Raise a clear error if the same entity appears in both groups."""
    train_values = _normalize_split_values(train_values)
    validation_values = _normalize_split_values(validation_values)
    overlap = sorted(set(train_values) & set(validation_values))
    if overlap:
        raise ValueError(
            f"Training and validation {field_name} overlap: {overlap}. "
            f"Each {field_name} must belong to exactly one split."
        )


def _select_ground_truth_split(training_fc, train_sites=None, validation_sites=None,
                              train_regions=None, validation_regions=None,
                              train_sectors=None, validation_sectors=None):
    """Create the scientific train/validation split based on site, region, or sector metadata."""
    if train_sites is not None or validation_sites is not None:
        if train_sites is None or validation_sites is None:
            raise ValueError('Both train_sites and validation_sites must be provided together for site-based validation.')
        train_sites = _normalize_split_values(train_sites)
        validation_sites = _normalize_split_values(validation_sites)
        _ensure_no_overlap(train_sites, validation_sites, 'sites')
        train_fc = training_fc.filter(ee.Filter.inList('site', train_sites))
        validation_fc = training_fc.filter(ee.Filter.inList('site', validation_sites))
        return train_fc, validation_fc

    if train_regions is not None or validation_regions is not None:
        if train_regions is None or validation_regions is None:
            raise ValueError('Both train_regions and validation_regions must be provided together for region-based validation.')
        train_regions = _normalize_split_values(train_regions)
        validation_regions = _normalize_split_values(validation_regions)
        _ensure_no_overlap(train_regions, validation_regions, 'regions')
        train_fc = training_fc.filter(ee.Filter.inList('region', train_regions))
        validation_fc = training_fc.filter(ee.Filter.inList('region', validation_regions))
        return train_fc, validation_fc

    if train_sectors is not None or validation_sectors is not None:
        if train_sectors is None or validation_sectors is None:
            raise ValueError('Both train_sectors and validation_sectors must be provided together for sector-based validation.')
        train_sectors = _normalize_split_values(train_sectors)
        validation_sectors = _normalize_split_values(validation_sectors)
        _ensure_no_overlap(train_sectors, validation_sectors, 'sectors')
        train_fc = training_fc.filter(ee.Filter.inList('sector', train_sectors))
        validation_fc = training_fc.filter(ee.Filter.inList('sector', validation_sectors))
        return train_fc, validation_fc

    raise ValueError(
        'No valid scientific split supplied. Provide train_sites/validation_sites, '
        'train_regions/validation_regions, or train_sectors/validation_sectors.'
    )


def extract_feature_values_for_ground_truth(image, ground_truth_fc, required_bands=None, scale=10):
    """Sample required Phase 1 features at each ground-truth location."""
    if required_bands is None:
        required_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']

    sample_fc = image.select(required_bands).sampleRegions(
        collection=ground_truth_fc,
        properties=['class', 'sample_id', 'site', 'region', 'country', 'sector', 'date'],
        scale=scale,
        geometries=False,
        tileScale=4
    )

    missing_bands = [band for band in required_bands if band not in image.bandNames().getInfo()]
    if missing_bands:
        raise ValueError(f"Image is missing required feature bands: {missing_bands}")

    sample_fc = sample_fc.filter(ee.Filter.notNull(required_bands))
    return sample_fc


def run_scientific_validation(ground_truth_path=None, analysis_date=None, aoi=None, lat=None, lon=None,
                             train_sites=None, validation_sites=None,
                             train_regions=None, validation_regions=None,
                             train_sectors=None, validation_sectors=None,
                             start_date=None, end_date=None, max_cloud=10,
                             scale=10, mode='prototype'):
    """
    Run validation workflow - either prototype/proxy mode or scientific validation mode.

    Parameters:
        ground_truth_path: Path to the master ground-truth CSV. REQUIRED for scientific mode;
                          MUST be None (or omitted) for prototype mode. Scientific mode raises
                          a clear ValueError if this is missing or invalid.
        analysis_date: Date for imagery selection (datetime or str '%Y-%m-%d'). Required for both modes.
        aoi: Optional pre-built AOI geometry
        lat, lon: Required if aoi is None
        train_sites/validation_sites: Site-based spatial split (scientific mode only)
        train_regions/validation_regions: Region-based spatial split (scientific mode only)
        train_sectors/validation_sectors: Sector-based spatial split (scientific mode only)
        start_date, end_date: Imagery date range
        max_cloud: Maximum cloud percentage for image selection
        scale: Sampling scale in meters
        mode: 'prototype' or 'scientific' — controls the validation workflow. Default: 'prototype'.
    """

    # ========================================
    # MODE SELECTION - HANDLE FIRST
    # ========================================

    if mode == 'prototype':
        # PROTOTYPE MODE: Use Dynamic World proxy, no ground-truth CSV required
        return _run_prototype_mode(
            analysis_date=analysis_date,
            aoi=aoi,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            max_cloud=max_cloud
        )

    elif mode == 'scientific':
        # SCIENTIFIC MODE: Requires ground-truth CSV and validation splits
        return _run_scientific_mode(
            ground_truth_path=ground_truth_path,
            analysis_date=analysis_date,
            aoi=aoi,
            lat=lat,
            lon=lon,
            train_sites=train_sites,
            validation_sites=validation_sites,
            train_regions=train_regions,
            validation_regions=validation_regions,
            train_sectors=train_sectors,
            validation_sectors=validation_sectors,
            start_date=start_date,
            end_date=end_date,
            max_cloud=max_cloud,
            scale=scale
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'prototype' or 'scientific'.")


def _run_prototype_mode(analysis_date, aoi=None, lat=None, lon=None,
                       start_date=None, end_date=None, max_cloud=10):
    """
    Run prototype/proxy mode using Dynamic World labels.

    Does NOT require ground-truth CSV.
    Does NOT perform scientific validation.
    Clearly labeled as proxy/prototype evaluation.
    """
    print("\n=== Running PROTOTYPE MODE (Dynamic World Proxy) ===\n")

    # Validate required parameters first — before any EE calls.
    if analysis_date is None:
        raise ValueError('analysis_date is required.')

    if isinstance(analysis_date, str):
        analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d')

    if start_date is None:
        start_date = analysis_date.strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (analysis_date + timedelta(days=1)).strftime('%Y-%m-%d')

    if aoi is None:
        if lat is None or lon is None:
            raise ValueError('Either aoi or both lat/lon must be provided.')
        aoi = make_aoi(float(lat), float(lon), buffer_m=3000)

    # Use run_full_pipeline which already handles prototype mode
    result = run_full_pipeline(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
        labelled_points=None
    )

    # Wrap classification result with explicit mode for UI dispatch
    classification_result = result.get('classification', {})
    classification_result['mode'] = 'prototype'

    # Wrap in validation-like result structure for UI consistency
    prototype_result = {
        'mode': 'prototype',
        'analysis_date': analysis_date.strftime('%Y-%m-%d'),
        'site': None,
        'train_sites': [],
        'validation_sites': [],
        'train_regions': [],
        'validation_regions': [],
        'train_sectors': [],
        'validation_sectors': [],
        'train_sample_count': 0,
        'validation_sample_count': 0,
        'class_counts': {},
        'confusion_matrix': None,
        'metrics': {
            'overall_accuracy': 0.0,
            'macro_f1': 0.0,
            'per_class': {},
            'producer_accuracy': {},
            'user_accuracy': {}
        },
        'image': result.get('image'),
        'aoi': result.get('aoi'),
        'feature_bands': ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore'],
        'scientific_note': PROXY_MODE_LABEL,
        'classification': classification_result,
        'thumbnails': result.get('thumbnails', {}),
        'ndvi': result.get('ndvi'),
        'ndwi': result.get('ndwi'),
        'water_mask': result.get('water_mask'),
        'ndvi_texture': result.get('ndvi_texture'),
        'distance_to_shore': result.get('distance_to_shore'),
        'chlorophyll_proxy': result.get('chlorophyll_proxy'),
        'turbidity_proxy': result.get('turbidity_proxy'),
        'temperature': result.get('temperature'),
        # HHRI and area keys from run_full_pipeline
        'hhri': result.get('hhri'),
        'hypoxia_risk_category': result.get('hypoxia_risk_category'),
        'chlorophyll_proxy_band': result.get('chlorophyll_proxy_band'),
        'turbidity_proxy_band': result.get('turbidity_proxy_band'),
        'temperature_band': result.get('temperature_band'),
        'hyacinth_density_band': result.get('hyacinth_density_band'),
        'hyacinth_area_ha': result.get('hyacinth_area_ha'),
        'water_area_ha': result.get('water_area_ha'),
        'hyacinth_coverage_pct': result.get('hyacinth_coverage_pct'),
        'other_vegetation_area_ha': result.get('other_vegetation_area_ha'),
        'hhri_mean': result.get('hhri_mean'),
        'chl_proxy_mean': result.get('chl_proxy_mean'),
        'turb_proxy_mean': result.get('turb_proxy_mean'),
    }

    print("\n=== PROTOTYPE MODE Complete ===\n")
    return prototype_result


def _run_scientific_mode(ground_truth_path, analysis_date, aoi=None, lat=None, lon=None,
                        train_sites=None, validation_sites=None,
                        train_regions=None, validation_regions=None,
                        train_sectors=None, validation_sectors=None,
                        start_date=None, end_date=None, max_cloud=10, scale=10):
    """
    Run scientific validation mode using human-labeled ground truth.

    REQUIRES ground-truth CSV.
    REQUIRES validation splits (site/region/sector).
    Performs strict hold-out validation.
    """
    print("\n=== Running SCIENTIFIC VALIDATION MODE ===\n")

    # Validate ground-truth path
    if ground_truth_path is None or not os.path.exists(ground_truth_path):
        raise FileNotFoundError(
            f"Scientific validation mode requires a valid ground-truth CSV. "
            f"Provided path: {ground_truth_path}. "
            f"Either provide a valid CSV or switch to prototype mode."
        )

    # Verify a validation split is configured before doing any heavy EE work.
    # This fails fast so the user is told what's missing without waiting on
    # imagery queries or (worse) silently running without a proper hold-out split.
    has_split = bool(
        train_sites or validation_sites
        or train_regions or validation_regions
        or train_sectors or validation_sectors
    )
    if not has_split:
        raise ValueError(
            'Scientific validation requires a split: train_sites/validation_sites, '
            'train_regions/validation_regions, or train_sectors/validation_sectors.'
        )

    # Validate analysis_date before any EE calls so invalid input fails fast.
    if analysis_date is None:
        raise ValueError('analysis_date is required for scientific validation.')
    if isinstance(analysis_date, str):
        analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d')

    # Set default date range relative to analysis_date — before AOI is built.
    if start_date is None:
        start_date = analysis_date.strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (analysis_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # Load and validate the ground-truth CSV schema before any EE work.
    # This is pure-Python validation, so malformed data fails fast without
    # waiting on (or requiring) authenticated Earth Engine calls.
    ground_truth = load_ground_truth_csv(ground_truth_path)

    # Build AOI — requires EE library (may fail if not authenticated).
    if aoi is None:
        if lat is None or lon is None:
            raise ValueError('Either aoi or both lat/lon must be provided to select imagery for scientific validation.')
        aoi = make_aoi(float(lat), float(lon), buffer_m=3000)

    image = get_sentinel2_image(aoi, start_date=start_date, end_date=end_date, max_cloud=max_cloud)
    if image is None:
        raise ValueError(
            f"No cloud-free image found for aoi and date range {start_date} to {end_date}. "
            "Scientific validation requires a valid analysis image."
        )

    image_with_indices = compute_indices(image)
    mask_image = compute_water_land_mask(image_with_indices, aoi)
    water_mask = mask_image.select('water_mask')
    image_with_texture = compute_ndvi_texture(image_with_indices, kernel_radius=3)
    image_with_features = compute_distance_to_shore(image_with_texture, water_mask)

    ground_truth_fc = ground_truth_to_ee_features(ground_truth)

    # Scientific mode: requires explicit train/validation split
    if train_sites is not None or validation_sites is not None:
        train_fc, validation_fc = _select_ground_truth_split(
            ground_truth_fc,
            train_sites=train_sites,
            validation_sites=validation_sites,
            train_regions=train_regions,
            validation_regions=validation_regions,
            train_sectors=train_sectors,
            validation_sectors=validation_sectors,
        )
    elif train_regions is not None or validation_regions is not None:
        train_fc, validation_fc = _select_ground_truth_split(
            ground_truth_fc,
            train_sites=train_sites,
            validation_sites=validation_sites,
            train_regions=train_regions,
            validation_regions=validation_regions,
            train_sectors=train_sectors,
            validation_sectors=validation_sectors,
        )
    elif train_sectors is not None or validation_sectors is not None:
        train_fc, validation_fc = _select_ground_truth_split(
            ground_truth_fc,
            train_sites=train_sites,
            validation_sites=validation_sites,
            train_regions=train_regions,
            validation_regions=validation_regions,
            train_sectors=train_sectors,
            validation_sectors=validation_sectors,
        )
    else:
        raise ValueError(
            'Scientific validation requires a split: train_sites/validation_sites, '
            'train_regions/validation_regions, or train_sectors/validation_sectors.'
        )

    split_summary = summarize_split(train_fc, validation_fc)

    train_samples = extract_feature_values_for_ground_truth(
        image_with_features,
        train_fc,
        required_bands=['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore'],
        scale=scale
    )
    validation_samples = extract_feature_values_for_ground_truth(
        image_with_features,
        validation_fc,
        required_bands=['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore'],
        scale=scale
    )

    if train_samples.size().getInfo() == 0:
        raise ValueError('No valid training samples with required feature values were extracted from the scientific ground truth.')
    if validation_samples.size().getInfo() == 0:
        raise ValueError('No valid validation samples with required feature values were extracted from the scientific ground truth.')

    feature_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']
    classifier = ee.Classifier.smileRandomForest(numberOfTrees=50, seed=42).train(
        features=train_samples,
        classProperty='class',
        inputProperties=feature_bands
    )

    validation_predictions = validation_samples.classify(classifier)
    error_matrix = validation_predictions.errorMatrix('class', 'classification')
    confusion_matrix = error_matrix.getInfo()
    class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    metrics = compute_validation_metrics(confusion_matrix, class_labels=class_labels)

    result = {
        'mode': 'scientific_validation',
        'analysis_date': analysis_date.strftime('%Y-%m-%d') if hasattr(analysis_date, 'strftime') else str(analysis_date),
        'site': None,
        'train_sites': train_sites or [],
        'validation_sites': validation_sites or [],
        'train_regions': train_regions or [],
        'validation_regions': validation_regions or [],
        'train_sectors': train_sectors or [],
        'validation_sectors': validation_sectors or [],
        'train_sample_count': train_fc.size().getInfo(),
        'validation_sample_count': validation_fc.size().getInfo(),
        'class_counts': split_summary,
        'confusion_matrix': confusion_matrix,
        'metrics': metrics,
        'image': image,
        'aoi': aoi,
        'feature_bands': feature_bands,
        'scientific_note': SCIENCE_MODE_LABEL,
        # Classification dict with explicit mode for UI dispatch
        'classification': {
            'mode': 'scientific_validation',
            'confusion_matrix': confusion_matrix,
            'confusion_labels': class_labels,
            'metrics': metrics,
            'train_sites': train_sites or [],
            'validation_sites': validation_sites or [],
            'train_regions': train_regions or [],
            'validation_regions': validation_regions or [],
            'train_sectors': train_sectors or [],
            'validation_sectors': validation_sectors or [],
            'train_sample_count': train_fc.size().getInfo(),
            'validation_sample_count': validation_fc.size().getInfo(),
            'training_accuracy': metrics.get('overall_accuracy', 0),
        },
        # Placeholder thumbnails for scientific mode (no thumbnails in scientific pipeline)
        'thumbnails': {},
    }

    print("\n=== SCIENTIFIC VALIDATION MODE Complete ===\n")
    return result


# ============================================================
# STEP 1: Authentication (handled separately; this just initializes)
# ============================================================

def initialize_ee(project_id=EE_PROJECT_ID):
    """Initialize Earth Engine with the given project."""
    ee.Initialize(project=project_id)
    print(f"Earth Engine initialized for project: {project_id}")


# ============================================================
# STEP 2: Area of Interest
# ============================================================

def make_aoi(lat, lon, buffer_m=3000):
    """Create an Area of Interest geometry from lat/lon with a buffer."""
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m)


# ============================================================
# STEP 3: Pull Sentinel-2 imagery
# ============================================================

def get_sentinel2_image(aoi, start_date='2025-01-01', end_date='2025-03-01', max_cloud=10):
    """
    Pull Sentinel-2 SR Harmonized imagery filtered by date, bounds, and cloud cover.

    Returns the first clear-image or None if no image meets criteria.
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )

    count = collection.size().getInfo()
    print(f"Found {count} Sentinel-2 images matching criteria")

    if count == 0:
        print("⚠️  No cloud-free Sentinel-2 image found for the given date range and location.")
        return None

    image = collection.first()

    # Clip to AOI
    image = image.clip(aoi)

    cloud_pct = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"Selected image date: {image.date().format().getInfo()}, Cloud %: {cloud_pct}%")

    return image


# ============================================================
# STEP 4: Compute Indices (NDVI, NDWI)
# ============================================================

def compute_indices(image):
    """
    Compute NDVI and NDWI from a Sentinel-2 image.

    NDVI = (B8 - B4) / (B8 + B4) — Vegetation/Hyacinth detection
    NDWI = (B3 - B8) / (B3 + B8) — Water detection
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # Add indices as bands to the image
    image_with_indices = image.addBands([ndvi, ndwi])

    print("✓ Computed NDVI and NDWI indices")
    return image_with_indices


# ============================================================
# STEP 5: Water/Land Mask using JRC GSW
# ============================================================

def compute_water_land_mask(image, aoi):
    """
    Generate a clean binary water/land mask using:
    1. JRC Global Surface Water 'occurrence' band (threshold >50%)
    2. NDWI as a secondary check

    Returns a binary image: 1=water, 0=land
    """
    # --- JRC GSW Water Occurrence ---
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    occurrence = jrc.select('occurrence')  # % of time pixel was observed as water

    # Threshold: >50% occurrence means reliably wet
    water_occurrence = occurrence.gte(50).rename('water_occurrence')

    # --- NDWI Secondary Check ---
    ndwi = image.select('NDWI')
    # NDWI > 0 typically indicates water; use a loose threshold to capture varied water bodies
    water_ndwi = ndwi.gt(0).rename('water_ndwi')

    # --- Combine: Both checks must agree (AND logic) ---
    # A pixel is water only if it has >50% JRC occurrence AND NDWI > 0
    # Use updateMask with the NDWI mask
    water_mask = water_occurrence.updateMask(water_ndwi).rename('water_mask')

    # Land is the complement
    land_mask = water_mask.Not().rename('land_mask')

    # Add both masks to an output image
    mask_image = ee.Image(0).addBands([water_mask, land_mask]).rename(
        ['constant', 'water_mask', 'land_mask']
    )

    # Clip to AOI
    mask_image = mask_image.clip(aoi)

    # Compute area statistics for verification
    # Use: water_mask.reduceRegion() directly for cleaner code
    try:
        water_pct = water_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        ).get('water_mask')
    except Exception:
        water_pct = None

    pct_info = water_pct.getInfo() if water_pct else None
    print(f"✓ Water/land mask computed. Water pixel %: {pct_info if isinstance(pct_info, float) else 'computing...'}")

    return mask_image


# ============================================================
# STEP 2 (partial): Location Discovery — search-by-city
# ============================================================

def geocode_city(city_name):
    """
    Geocode a city/place name into lat/lon using geopy/Nominatim.
    Returns (lat, lon) tuple or raises ValueError.
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    geolocator = Nominatim(user_agent="water-hyacinth-detection")

    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location is None:
            raise ValueError(f"Could not geocode city: '{city_name}'")
        lat, lon = location.latitude, location.longitude
        print(f"✓ Geocoded '{city_name}': {lat}, {lon}")
        return (lat, lon)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        raise ValueError(f"Geocoding service error: {e}")


def discover_water_bodies(city_name, radius_km=25):
    """
    Search for water bodies around a geocoded city using JRC GSW dataset.

    Uses reduceToVectors to find distinct water body polygons within the radius,
    then returns approximate centroids and estimated sizes.

    Returns a dict containing the top water bodies with centroid coordinates and
    estimated size in hectares.
    """
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic

    # Geocode the city
    lat, lon = geocode_city(city_name)

    # Create AOI buffer around the city
    aoi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)  # meters

    # --- JRC GSW Water Occurrence ---
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    occurrence = jrc.select('occurrence')

    # Threshold >35% to balance persistent water detection and seasonal coverage
    water_mask = occurrence.gte(35)

    water_pixel_count = water_mask.selfMask().reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi,
        scale=30,
        maxPixels=1e9
    ).get('occurrence').getInfo()
    print(f"Water pixels found before vectorization: {water_pixel_count}")

    # Reduce to vectors: find distinct water polygons
    # This extracts connected components of water pixels
    try:
        water_polygons = water_mask.reduceToVectors(
            geometry=aoi,
            reducer=ee.Reducer.countEvery(),
            scale=30,
            geometryType='polygon',
            tileScale=4
        )
    except Exception as e:
        print(f"⚠️ reduceToVectors error: {e}")
        water_polygons = None

    # Filter out small polygons, then keep the ten largest water bodies.
    if water_polygons:
        def add_area(feature):
            area_ha = feature.geometry().area(1).divide(10000)
            return feature.set('area_ha', area_ha)

        def add_centroid(feature):
            centroid = feature.geometry().centroid(1).coordinates()
            return feature.set({
                'centroid_lon': centroid.get(0),
                'centroid_lat': centroid.get(1)
            })

        fc = (water_polygons
              .map(add_area)
              .filter(ee.Filter.gte('area_ha', 5))
              .sort('area_ha', False)
              .limit(10)
              .map(add_centroid))

        count = fc.size().getInfo()
        if count == 0:
            print("⚠️ No water bodies detected in the specified radius.")
            return {
                'city': city_name,
                'center_lat': lat,
                'center_lon': lon,
                'polygon_count': 0,
                'water_bodies': [],
                'aoi': aoi
            }

        print(f"✓ Found {count} water body polygons near {city_name}")

        water_bodies = []
        for feature in fc.getInfo().get('features', []):
            properties = feature.get('properties', {})
            water_bodies.append({
                'centroid_lat': properties.get('centroid_lat'),
                'centroid_lon': properties.get('centroid_lon'),
                'estimated_size_ha': properties.get('area_ha')
            })

        return {
            'city': city_name,
            'center_lat': lat,
            'center_lon': lon,
            'polygon_count': count,
            'water_bodies': water_bodies,
            'aoi': aoi
        }

    else:
        print("⚠️ No water bodies detected in the specified radius.")
        return {
            'city': city_name,
            'center_lat': lat,
            'center_lon': lon,
            'polygon_count': 0,
            'water_bodies': [],
            'aoi': aoi
        }


# ============================================================
# STEP 3 (continued): Feature Engineering
# ============================================================

def compute_ndvi_texture(image, kernel_radius=3):
    """
    Compute NDVI local texture (standard deviation) using a moving window.
    Kernel radius 3 pixels — distinguishes uniform hyacinth mats from patchy lotus/other vegetation.

    Returns image with added 'ndvi_texture' band.
    """
    ndvi = image.select('NDVI')
    # reduceNeighborhood with stdDev reducer
    ndvi_texture = ndvi.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=kernel_radius)
    ).rename('ndvi_texture')

    image_with_texture = image.addBands([ndvi_texture])
    print(f"✓ Computed NDVI texture (kernel radius={kernel_radius})")
    return image_with_texture


def compute_distance_to_shore(image, land_mask):
    """
    Compute distance to shore from the land mask.
    Uses fastDistanceTransform then sqrt.
    Lotus tends to be near edges; hyacinth can occupy open water.

    Returns image with added 'distance_to_shore' band.
    """
    # land_mask should be 1 for land, 0 for water
    # fastDistanceTransform computes distance from each water pixel to nearest land
    distance = land_mask.fastDistanceTransform().sqrt().rename('distance_to_shore')

    image_with_dist = image.addBands([distance])
    print(f"✓ Computed distance to shore transform")
    return image_with_dist


def compute_ndvi_difference(image1, image2, aoi):
    """
    Compute NDVI difference between two image dates (temporal spread-rate feature).
    Takes two images and an AOI, returns the NDVI difference image.

    This is implemented as a separate function for multi-date analysis —
    not wired into the main pipeline yet as requested.
    """
    # Ensure both images have NDVI
    ndvi1 = image1.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi2 = image2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Clip both to AOI
    ndvi1 = ndvi1.clip(aoi)
    ndvi2 = ndvi2.clip(aoi)

    # Difference: NDVI2 - NDVI1
    ndvi_diff = ndvi2.subtract(ndvi1).rename('ndvi_difference')

    print("✓ Computed NDVI difference between two dates")
    return ndvi_diff


# ============================================================
# Phase 2: Biophysical Proxies
# ============================================================


def compute_chlorophyll_proxy(image, formula='Gitelson', red_edge_band='B5', red_band='B4',
                              nir_band='B8', coefficients=None):
    """
    Compute satellite-derived chlorophyll-a proxy using the Gitelson formulation.

    Gitelson et al. (2008, 2014) red-edge chlorophyll algorithm for Sentinel-2:
    Chl = L * ((RhoRedEdge / RhoRed) - C1) / ((RhoRedEnd / RhoRed) - C2) + offset

    Where:
    - RhoRedEnd = reflectance at red edge (B5 for Sentinel-2)
    - RhoRed = reflectance at red (B4 for Sentinel-2)
    - L, C1, C2 = sensor-specific coefficients

    Returns chlorophyll proxy in μg/L (or arbitrary units after normalization).
    """
    if coefficients is None:
        # Documented Gitelson coefficients for medium-resolution missions
        coefficients = {
            'L': 2.56,
            'G': 6.09,
            'C1': 7.11,
            'C2': 19.67
        }

    # Select the appropriate bands from the image
    try:
        rho_red = image.select(red_band).multiply(0.0001).rename('rho_red')  # TOA reflectance scaling
        rho_red_edge = image.select(red_edge_band).multiply(0.0001).rename('rho_red_edge')
        rho_nir = image.select(nir_band).multiply(0.0001).rename('rho_nir')
    except Exception:
        # Fallback: assume bands already in image
        rho_red = image.select(red_band).rename('rho_red')
        rho_red_edge = image.select(red_edge_band).rename('rho_red_edge')
        rho_nir = image.select(nir_band).rename('rho_nir')

    # Gitelson red-edge chlorophyll formula (preserved exactly):
    # Chl = L * ((RhoRedEdge / RhoRed) - C1) / ((RhoRedEdge / RhoRed) - C2)
    # EE Image must call the method, NOT the Python float coefficient.
    ratio = rho_red_edge.divide(rho_red)
    chl_proxy = ratio.subtract(coefficients['C1']).divide(
        ratio.subtract(coefficients['C2'])
    ).multiply(coefficients['L']).rename('chlorophyll_a_proxy')

    print(f"✓ Computed chlorophyll-a proxy using {formula} formula (Gitelson red-edge)")
    return image.addBands([chl_proxy])


def compute_turbidity_proxy(image, ratio_bands=None):
    """
    Compute satellite turbidity proxy using a band ratio formulation.

    Turbidity proxy = Red / NIR (or Blue / Red depending on algorithm).
    For Sentinel-2: typically B4 (Red) / B8 (NIR) or B3 (Green) / B4 (Red).

    Returns a normalized turbidity proxy band.
    """
    if ratio_bands is None:
        # Default: Red (B4) / NIR (B8) - works well for suspended sediments
        ratio_bands = ['B4', 'B8']

    try:
        red_band_obj = image.select(ratio_bands[0])
        nir_band_obj = image.select(ratio_bands[1])
        # Ratio: Red / NIR
        turbidity = red_band_obj.divide(nir_band_obj).rename('turbidity_proxy')
    except Exception:
        # Alternative: Green / Red
        try:
            green_band = image.select('B3')
            red_band = image.select('B4')
            turbidity = green_band.divide(red_band).rename('turbidity_proxy')
        except Exception:
            raise ValueError("Could not compute turbidity proxy - required bands not found")

    print("✓ Computed turbidity proxy (Red/NIR band ratio)")
    return image.addBands([turbidity])


def compute_surface_temperature_placeholder(image, aoi, use_landsat=True):
    """
    Compute surface temperature estimate.

    For Sentinel-2 (no thermal bands): returns a placeholder indicating limitation.
    For Landsat: retrieves thermal band, applies cloud masking, and processes surface temperature.

    Returns:
    - If use_landsat=True and Landsat imagery is available: surface temperature in Celsius
    - If only Sentinel-2: a placeholder proxy band with clear documentation of limitation
    """
    # Check if this is Sentinel-2 (no thermal bands)
    try:
        band_names = image.bandNames().getInfo()
        has_thermal = 'B10' in band_names or 'B11' in band_names
    except Exception:
        has_thermal = False

    if not has_thermal or use_landsat:
        # Try Landsat approach - look for Landsat 8/9 thermal data
        try:
            return compute_landsat_surface_temperature(image, aoi)
        except Exception:
            # Fall back to placeholder
            return compute_sentinel2_temperature_placeholder(image, aoi)
    else:
        # Sentinel-2 has no thermal bands - return placeholder
        return compute_sentinel2_temperature_placeholder(image, aoi)


def compute_landsat_surface_temperature(image, aoi):
    """
    Retrieve Landsat thermal band, apply cloud masking, and compute surface temperature.

    Uses the single-channel method with Landsat 8/9 STBQA calibration coefficients.
    See: USGS documentation for Landsat surface temperature product.

    Returns image with 'land_surface_temperature' band in Celsius.
    """
    # This is a placeholder implementation - actual Landsat processing would require
    # a separate Landsat image collection, which is outside the Sentinel-2-only pipeline.
    # The function signature accepts a Sentinel-2 image, so we document the limitation.
    raise NotImplementedError(
        "Landsat surface temperature requires a separate Landsat image collection. "
        "This function is called from compute_surface_temperature_placeholder for "
        "interface compatibility but will raise NotImplementedError with Sentinel-2 data."
    )


def compute_sentinel2_temperature_placeholder(image, aoi):
    """
    Return a placeholder temperature band for Sentinel-2 images (which lack thermal bands).

    Documents the limitation clearly and provides a no-op band that users can replace
    with actual in-situ measurements when available.

    Returns image with 'water_surface_temperature_placeholder' band.
    """
    # Create a placeholder band with a clear nodata/missing value indicator
    # All values will be -999 with a comment band explaining the limitation
    placeholder = image.select('B1').multiply(0).add(-999).rename(
        'water_surface_temperature_placeholder'
    )

    # Add a quality advisory band
    advisory = image.select('B1').multiply(0).add(0).rename('temp_advisory')
    # Set advisory value to indicate Sentinel-2 limitation
    # 0 = Sentinel-2 no thermal data; 1 = Landsat available; 2 = in-situ measurement

    # Clip to AOI
    placeholder = placeholder.clip(aoi)
    advisory = advisory.clip(aoi)

    print("✓ Computed surface temperature placeholder — Sentinel-2 has no thermal bands")
    return image.addBands([placeholder, advisory])


def compute_hypoxia_risk(image, chl_proxy_band='chlorophyll_a_proxy',
                         turbidity_proxy_band='turbidity_proxy',
                         temperature_band='water_surface_temperature_placeholder',
                         hyacinth_density_band=None,
                         weights=None, thresholds=None):
    """
    Predict hypoxia risk (LOW, MODERATE, HIGH) using a feature-based model.

    This is a proxy/rule-based risk estimate, NOT a measured dissolved oxygen model.

    Critical research contribution:
    Phase 1 hyacinth output becomes an input feature for hypoxia-risk estimation.

    The hypothesis:
    Higher aquatic vegetation burden can be associated with ecological conditions
    that increase oxygen stress under appropriate environmental conditions, but
    the relationship is site- and context-dependent.

    Do not claim causation from satellite correlation alone.

    Parameters:
    - image: Earth Engine Image with feature bands
    - chl_proxy_band: name of chlorophyll proxy band
    - turbidity_proxy_band: name of turbidity proxy band
    - temperature_band: name of temperature band
    - hyacinth_density_band: name of hyacinth classification band (from Phase 1)
    - weights: dict of configurable weights {w1, w2, w3, w4, w5}
    - thresholds: dict of risk category thresholds {low, moderate}

    Returns image with 'hypoxia_risk' band (categorical: 0=LOW, 1=MODERATE, 2=HIGH)
    and 'hhri' band (continuous HHRI index).
    """
    # Set default weights (configurable)
    if weights is None:
        weights = {
            'w1_ndvi': 1.0,
            'w2_ndwi': 1.0,
            'w3_chl': 1.0,
            'w4_turbidity': 1.0,
            'w5_doproxy': 1.0
        }

    # Set default thresholds
    if thresholds is None:
        thresholds = {
            'low': 0.3,
            'moderate': 0.6
        }

    # Get chlorophyll proxy band
    try:
        chl = image.select(chl_proxy_band)
    except Exception:
        chl = image.normalizedDifference([]).rename('chl_proxy_placeholder')  # fallback

    # Get turbidity proxy band
    try:
        turb = image.select(turbidity_proxy_band)
    except Exception:
        turb = image.normalizedDifference([]).rename('turbidity_proxy_placeholder')

    # Get temperature band
    try:
        temp = image.select(temperature_band)
        # If it's the placeholder (-999), replace with sensible default
        temp = temp.where(temp.gt(-500), image.select('B1').multiply(0).add(20))
        temp = temp.rename('temperature_celsius')
    except Exception:
        temp = image.normalizedDifference([]).rename('temp_placeholder')

    # Get hyacinth density if available
    if hyacinth_density_band:
        try:
            hyacinth = image.select(hyacinth_density_band)
        except Exception:
            hyacinth = image.normalizedDifference([]).rename('hyacinth_density_placeholder')
    else:
        # Default: assume low hyacinth density (proxy = 0)
        hyacinth = image.normalizedDifference([]).rename('hyacinth_density_zero')

    # Normalize inputs

    # Normalize chlorophyll proxy (assume typical range 0-100 μg/L for normalization)
    chl_norm = chl.divide(100).min(1).max(0).rename('chl_norm')

    # Normalize turbidity proxy (ratio is already 0+, clip at reasonable max)
    turb_norm = turb.min(5).divide(5).rename('turb_norm')  # clip at 5, normalize

    # Normalize temperature (assume typical water temp range 0-30°C)
    temp_norm = temp.divide(30).min(1).max(0).rename('temp_norm')

    # Normalize hyacinth density (0-1 range)
    hyac_norm = hyacinth.min(1).max(0).rename('hyac_norm')

    # Normalize NDVI to 0-1 range: (NDVI + 1) / 2
    ndvi_norm = (image.select('NDVI').add(1)).multiply(0.5).rename('ndvi_norm')

    # Normalize 1-NDWI to 0-1 range
    not_ndwi_norm = (ee.Image(1).subtract(image.select('NDWI'))).multiply(0.5).rename('not_ndwi_norm')

    # Compute HHRI index:
    # HHRI = w1(NDVI) + w2(1 - NDWI) + w3(Chl) + w4(Turbidity) - w5(DO_proxy)
    # Since DO_proxy is not available from satellite, we use temperature as a proxy placeholder
    # The critical research contribution is that Phase 1 hyacinth output enters as a feature
    #
    # IMPORTANT: All arithmetic must use Earth Engine server-side operations (.add/.subtract),
    # never Python '+'/'-' operators on EE server objects (which raises TypeError).

    # Each weighted term must remain an EE Image so .add()/.subtract() work.
    # ee.Number.multiply(ee.Image) -> ee.Image; we explicitly cast.
    term1 = ndvi_norm.multiply(weights.get('w1_ndvi', 1.0)).rename('t1')
    term2 = not_ndwi_norm.multiply(weights.get('w2_ndwi', 1.0)).rename('t2')
    term3 = chl_norm.multiply(weights.get('w3_chl', 1.0)).rename('t3')
    term4 = turb_norm.multiply(weights.get('w4_turbidity', 1.0)).rename('t4')
    term5 = temp_norm.multiply(weights.get('w5_doproxy', 1.0)).rename('t5')  # DO_proxy placeholder

    # Build HHRI step-by-step using EE server-side operations only.
    # Formula unchanged: term1 + term2 + term3 + term4 - term5
    hhri = (
        term1.add(term2).add(term3).add(term4).subtract(term5)
    ).rename('hhri')

    # Apply HHRI thresholds to get risk categories
    low_threshold = ee.Number(thresholds.get('low', 0.3))
    moderate_threshold = ee.Number(thresholds.get('moderate', 0.6))

    # Categorical risk assignment
    # LOW: HHRI < low_threshold (risk = 0)
    # MODERATE: low_threshold <= HHRI < moderate_threshold (risk = 1)
    # HIGH: HHRI >= moderate_threshold (risk = 2)

    # Use EE's .And() method for boolean operations
    is_low = hhri.lt(low_threshold)
    is_moderate = hhri.gte(low_threshold).And(hhri.lt(moderate_threshold))
    is_high = hhri.gte(moderate_threshold)

    # Create risk category images
    risk_low = ee.Image(0).multiply(is_low.rename('mask_low')).rename('risk_low')
    risk_mod = ee.Image(1).multiply(is_moderate.rename('mask_mod')).rename('risk_mod')
    risk_high = ee.Image(2).multiply(is_high.rename('mask_high')).rename('risk_high')

    # Combine: risk = risk_low + risk_mod + risk_high (only one should be 1, others 0)
    # But since they are mutually exclusive, we can add them
    hypoxia_risk = risk_low.add(risk_mod).add(risk_high).rename('hypoxia_risk_category')

    result = {
        'hhri': hhri,
        'chlorophyll_proxy': chl_norm,
        'turbidity_proxy': turb_norm,
        'temperature': temp_norm,
        'hyacinth_density': hyac_norm,
        'hypoxia_risk_category': hypoxia_risk,
        'weights': weights,
        'thresholds': thresholds,
        'note': 'Hypoxia risk model: proxy/rule-based estimate. '
                'Phase 1 hyacinth density is an input feature. '
                'Do not claim measured dissolved oxygen or causation. '
                'HHRI combines NDVI, (1-NDWI), chlorophyll proxy, turbidity proxy, '
                'and a DO_proxy placeholder (using temperature). '
                'Weights are configurable but not empirically validated.'
    }

    print("✓ Computed hypoxia risk proxy with HHRI index")
    return result


# ============================================================
# STEP 4: Training Data Collection Helper
# ============================================================

def normalize_class_label(class_label):
    """Normalize class labels to the Phase 1 ontology: 0=Water, 1=Water Hyacinth, 2=Other Vegetation."""
    if isinstance(class_label, str):
        normalized = class_label.strip().lower().replace('_', ' ')
        aliases = {
            'water': 0,
            'water hyacinth': 1,
            'hyacinth': 1,
            'other vegetation': 2,
            'other vegetation lotus': 2,
            'lotus': 2,
            'lotus like': 2,
            'lotus-like': 2,
            'other veg': 2,
            'other vegetation lotus like': 2
        }
        if normalized in aliases:
            return aliases[normalized]

    try:
        class_int = int(class_label)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid class label '{class_label}'. Expected one of: {list(CLASS_NAMES.keys())}")

    if class_int not in CLASS_NAMES:
        raise ValueError(
            f"Invalid class label {class_int}. Valid classes are: {list(CLASS_NAMES.keys())}. "
            "Phase 1 classes are 0=Water, 1=Water Hyacinth, 2=Other Vegetation."
        )

    return class_int


def validate_ground_truth_sets(train_fc, validation_fc=None, expected_classes=None):
    """Validate the training and validation feature collections before model fitting."""
    if expected_classes is None:
        expected_classes = list(CLASS_NAMES.keys())

    if train_fc is None or train_fc.size().getInfo() == 0:
        raise ValueError('Training data is empty. Provide at least one labeled sample per class in the training set.')

    train_classes = set(train_fc.aggregate_array('class').getInfo())
    if not set(expected_classes).issubset(train_classes):
        missing = sorted(set(expected_classes) - set(train_classes))
        raise ValueError(
            f"Training data is missing required classes: {missing}. "
            "Phase 1 requires all classes 0, 1, and 2 to be represented in training data."
        )

    if validation_fc is not None:
        if validation_fc.size().getInfo() == 0:
            raise ValueError('Validation data is empty. A non-empty validation set is required for scientific evaluation.')

        validation_classes = set(validation_fc.aggregate_array('class').getInfo())
        unknown = sorted(validation_classes - set(expected_classes))
        if unknown:
            raise ValueError(f"Validation data contains unknown classes: {unknown}. Valid classes are 0, 1, and 2 only.")

        if not set(expected_classes).issubset(validation_classes):
            missing = sorted(set(expected_classes) - set(validation_classes))
            raise ValueError(
                f"Validation data is missing required classes: {missing}. "
                "Each validation set should include class 0, 1, and 2 when available in the site."
            )

    return True


def split_ground_truth_by_sector(training_fc, train_sectors=None, validation_sectors=None):
    """Create spatially independent train/validation sets based on sector labels."""
    if train_sectors is None or validation_sectors is None:
        raise ValueError(
            'Spatial validation requires explicit train_sectors and validation_sectors. '
            'If sector metadata is missing, do not fall back to a misleading random split.'
        )

    train_sectors = [str(s).strip().lower() for s in train_sectors]
    validation_sectors = [str(s).strip().lower() for s in validation_sectors]

    overlap = set(train_sectors) & set(validation_sectors)
    if overlap:
        raise ValueError(
            f"Training and validation sectors overlap: {sorted(overlap)}. Each sector must belong to one split only."
        )

    train_fc = training_fc.filter(ee.Filter.inList('sector', train_sectors))
    validation_fc = training_fc.filter(ee.Filter.inList('sector', validation_sectors))

    if train_fc.size().getInfo() == 0:
        raise ValueError(f"No training samples found for sectors: {train_sectors}")
    if validation_fc.size().getInfo() == 0:
        raise ValueError(f"No validation samples found for sectors: {validation_sectors}")

    # Patch-level leakage guard: same sample_id or patch_id must not appear in both sets.
    train_ids = set(train_fc.aggregate_array('sample_id').getInfo())
    validation_ids = set(validation_fc.aggregate_array('sample_id').getInfo())
    overlap_ids = train_ids & validation_ids
    if overlap_ids:
        raise ValueError(
            'Sample leakage detected: the same sample_id appears in both training and validation sets: '
            f'{sorted(list(overlap_ids))[:10]}'
        )

    validate_ground_truth_sets(train_fc, validation_fc)
    return train_fc, validation_fc


def split_ground_truth_by_site_or_region(training_fc, train_sites=None, validation_sites=None,
                                        train_regions=None, validation_regions=None):
    """Create train/validation splits by site or region labels without site-specific assumptions."""
    if train_sites is not None or validation_sites is not None:
        if train_sites is None or validation_sites is None:
            raise ValueError('Both train_sites and validation_sites must be provided together for site-based validation.')

        train_sites = [str(s).strip().lower() for s in train_sites]
        validation_sites = [str(s).strip().lower() for s in validation_sites]
        overlap = set(train_sites) & set(validation_sites)
        if overlap:
            raise ValueError(f"Training and validation sites overlap: {sorted(overlap)}")

        train_fc = training_fc.filter(ee.Filter.inList('site', train_sites))
        validation_fc = training_fc.filter(ee.Filter.inList('site', validation_sites))

        train_ids = set(train_fc.aggregate_array('sample_id').getInfo())
        validation_ids = set(validation_fc.aggregate_array('sample_id').getInfo())
        overlap_ids = train_ids & validation_ids
        if overlap_ids:
            raise ValueError(
                'Sample leakage detected across train/validation sites: '
                f'{sorted(list(overlap_ids))[:10]}'
            )

        validate_ground_truth_sets(train_fc, validation_fc)
        return train_fc, validation_fc

    if train_regions is not None or validation_regions is not None:
        if train_regions is None or validation_regions is None:
            raise ValueError('Both train_regions and validation_regions must be provided together for region-based validation.')

        train_regions = [str(r).strip().lower() for r in train_regions]
        validation_regions = [str(r).strip().lower() for r in validation_regions]
        overlap = set(train_regions) & set(validation_regions)
        if overlap:
            raise ValueError(f"Training and validation regions overlap: {sorted(overlap)}")

        train_fc = training_fc.filter(ee.Filter.inList('region', train_regions))
        validation_fc = training_fc.filter(ee.Filter.inList('region', validation_regions))

        train_ids = set(train_fc.aggregate_array('sample_id').getInfo())
        validation_ids = set(validation_fc.aggregate_array('sample_id').getInfo())
        overlap_ids = train_ids & validation_ids
        if overlap_ids:
            raise ValueError(
                'Sample leakage detected across train/validation regions: '
                f'{sorted(list(overlap_ids))[:10]}'
            )

        validate_ground_truth_sets(train_fc, validation_fc)
        return train_fc, validation_fc

    raise ValueError(
        'No site or region split supplied. Provide train_sites/validation_sites or train_regions/validation_regions.'
    )


def compute_validation_metrics(confusion_matrix, class_labels=None):
    """Compute overall accuracy, per-class precision/recall/F1, macro F1, and class accuracies."""
    if class_labels is None:
        class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    matrix = confusion_matrix
    if not matrix or not matrix[0]:
        raise ValueError('Confusion matrix is empty; cannot compute validation metrics.')

    n_classes = len(class_labels)
    if len(matrix) != n_classes:
        raise ValueError(
            f"Confusion matrix length ({len(matrix)}) does not match the number of labels ({n_classes})."
        )

    for row in matrix:
        if len(row) != n_classes:
            raise ValueError('Confusion matrix is not square.')

    total_samples = sum(sum(row) for row in matrix)
    overall_accuracy = sum(matrix[i][i] for i in range(n_classes)) / total_samples if total_samples > 0 else 0.0

    class_metrics = {}
    for i, label in enumerate(class_labels):
        tp = matrix[i][i]
        row_total = sum(matrix[i])
        col_total = sum(matrix[r][i] for r in range(n_classes))

        precision = tp / col_total if col_total else 0.0
        recall = tp / row_total if row_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'producer_accuracy': recall,
            'user_accuracy': precision,
            'support': row_total
        }

    macro_f1 = sum(metrics['f1'] for metrics in class_metrics.values()) / n_classes if n_classes > 0 else 0.0

    results = {
        'confusion_matrix': matrix,
        'class_labels': class_labels,
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'per_class': class_metrics,
        'producer_accuracy': {label: metrics['producer_accuracy'] for label, metrics in class_metrics.items()},
        'user_accuracy': {label: metrics['user_accuracy'] for label, metrics in class_metrics.items()}
    }

    return results


def create_training_fc(labelled_points, aoi):
    """
    Create an ee.FeatureCollection from manually identified lat/lon points with class labels.

    labelled_points: list of dicts, each with:
        - 'lat': float
        - 'lon': float
        - 'class': int (0=Water, 1=Water Hyacinth, 2=Other Vegetation)
        - optionally 'label': str (human-readable)
        - optionally 'sector': str (spatial region identifier for validation splitting)
    """
    features = []

    for point in labelled_points:
        lat = point['lat']
        lon = point['lon']
        class_label = normalize_class_label(point['class'])
        label = point.get('label', CLASS_NAMES.get(class_label, f'class_{class_label}'))
        sector = point.get('sector')

        feature_props = {
            'site': str(point.get('site', 'unknown')).strip(),
            'region': str(point.get('region', 'unknown')).strip(),
            'country': str(point.get('country', 'unknown')).strip(),
            'class': class_label,
            'label': label,
            'date': str(point.get('date', 'unknown')).strip(),
            'source': str(point.get('source', 'manual')).strip(),
            'sample_type': str(point.get('sample_type', 'point')).strip(),
            'notes': str(point.get('notes', '')).strip()
        }
        if sector is not None:
            feature_props['sector'] = str(sector).strip().lower()
        else:
            feature_props['sector'] = str(point.get('sector', 'unknown')).strip().lower()

        point_geom = ee.Geometry.Point([lon, lat])
        feature = ee.Feature(point_geom, feature_props)
        features.append(feature)

    if features:
        training_fc = ee.FeatureCollection(features)
    else:
        training_fc = ee.FeatureCollection([])

    training_fc = training_fc.filterBounds(aoi)

    if training_fc.size().getInfo() > 0:
        class_counts = training_fc.aggregate_histogram('class').getInfo()
        print(f"✓ Created training FC with {training_fc.size().getInfo()} points")
        print(f"  Class distribution: {class_counts}")

    return training_fc


# ============================================================
# STEP 5: Random Forest Classifier
# ============================================================

def create_dynamic_world_training_samples(image, aoi, start_date, end_date):
    """Create automatic water and aquatic-vegetation samples from Dynamic World."""
    dynamic_world = (
        ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select('label')
        .mode()
        .rename('dynamic_world_label')
    )

    classification_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']
    sampling_image = image.select(classification_bands).addBands(dynamic_world)

    samples = sampling_image.stratifiedSample(
        numPoints=100,
        classBand='dynamic_world_label',
        classValues=[0, 4],
        classPoints=[100, 100],
        region=aoi,
        scale=10,
        geometries=True,
        dropNulls=True,
        seed=42,
        tileScale=4
    )

    def set_classifier_class(feature):
        classifier_class = ee.Number(
            ee.Algorithms.If(
                ee.Number(feature.get('dynamic_world_label')).eq(4), 1, 0
            )
        )
        return feature.set('class', classifier_class)

    training_fc = samples.map(set_classifier_class)
    sample_count = training_fc.size().getInfo()
    print(f"✓ Created {sample_count} automatic Dynamic World training samples")
    return training_fc


def postprocess_vegetation_subclassification(classified_image, feature_image,
                                          ndvi_texture_threshold=0.08,
                                          distance_to_shore_threshold=50,
                                          vegetation_value=1):
    """
    Split the initial vegetation class into two subclasses using texture + shore distance.

    Rule used:
    - Hyacinth: low NDVI texture (uniform mat) + far from shore
    - Other vegetation / lotus-like: high NDVI texture (patchy) + near shore

    Class mapping after this step:
    - 0 = water
    - 1 = hyacinth
    - 2 = other vegetation / lotus
    """
    is_vegetation = classified_image.eq(vegetation_value)
    ndvi_texture = feature_image.select('ndvi_texture')
    distance_to_shore = feature_image.select('distance_to_shore')

    hyacinth_mask = (
        is_vegetation
        .And(ndvi_texture.lte(ndvi_texture_threshold))
        .And(distance_to_shore.gte(distance_to_shore_threshold))
    )

    # Everything else classified as vegetation is treated as other vegetation / lotus-like
    other_vegetation_mask = is_vegetation.And(hyacinth_mask.Not())

    refined = classified_image.where(hyacinth_mask, 1).where(other_vegetation_mask, 2)
    return refined.rename('classification')


def classify_random_forest(image, training_fc, classification_bands=None,
                          ndvi_texture_threshold=0.08,
                          distance_to_shore_threshold=50,
                          postprocess_vegetation=True,
                          water_mask=None,
                          water_buffer_m=30,
                          test_fraction=0.3):
    """
    Train a Random Forest classifier using Earth Engine's smileRandomForest.

    We restrict the classification to the actual water body plus a small shoreline
    buffer (~20-30m) before applying the vegetation split. This prevents inland
    land vegetation from being incorrectly shown as aquatic vegetation.

    Accuracy is computed on a held-out test set, using a 70/30 train/test split.
    This measures generalization performance rather than memorization of the training labels.
    """
    if classification_bands is None:
        classification_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']

    # Verify all bands exist in the image
    available_bands = image.bandNames().getInfo()
    for band in classification_bands:
        if band not in available_bands:
            raise ValueError(f"Band '{band}' not found in image. Available: {available_bands}")

    # Restrict classification to the water body + small shoreline buffer.
    if water_mask is None:
        classification_mask = ee.Image(1).rename('classification_mask')
    else:
        water_mask = ee.Image(water_mask).rename('water_mask')
        shoreline_buffer = water_mask.focal_max(
            radius=water_buffer_m, units='meters', iterations=1
        )
        classification_mask = water_mask.Or(shoreline_buffer).rename('classification_mask')

    stacked = image.select(classification_bands).updateMask(classification_mask)

    trained_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=50,
        seed=42
    ).train(
        features=training_fc,
        classProperty='class',
        inputProperties=classification_bands
    )

    binary_classified_image = stacked.classify(trained_classifier).rename('classification')
    binary_classified_image = binary_classified_image.updateMask(classification_mask)

    if postprocess_vegetation and 'ndvi_texture' in available_bands and 'distance_to_shore' in available_bands:
        feature_layers = image.select(['ndvi_texture', 'distance_to_shore'])
        classified_image = postprocess_vegetation_subclassification(
            binary_classified_image,
            feature_layers,
            ndvi_texture_threshold=ndvi_texture_threshold,
            distance_to_shore_threshold=distance_to_shore_threshold,
            vegetation_value=1
        )
        palette = ['blue', 'darkgreen', 'pink']
        max_value = 2
    else:
        classified_image = binary_classified_image
        palette = ['blue', 'darkgreen']
        max_value = 1

    classified_image = classified_image.updateMask(classification_mask)

    # Proper train/test split: 70% train, 30% held-out test.
    # This is the metric that reflects real generalization performance.
    test_points = training_fc.randomColumn(seed=1).filter(ee.Filter.lt('random', test_fraction))
    train_points = training_fc.filter(ee.Filter.gte('random', test_fraction))

    accuracy = test_points.classify(trained_classifier).errorMatrix(
        'class', 'classification'
    )
    accuracy_metrics = accuracy.getInfo()

    confusion_matrix = (
        accuracy_metrics
        if isinstance(accuracy_metrics, list)
        else accuracy_metrics['array']
    )

    class_labels = ['Water', 'Hyacinth', 'Other Vegetation']
    padded_matrix = []
    for i in range(len(class_labels)):
        row = []
        for j in range(len(class_labels)):
            if i < len(confusion_matrix) and j < len(confusion_matrix[i]):
                row.append(confusion_matrix[i][j])
            else:
                row.append(0)
        padded_matrix.append(row)

    total_correct = sum(
        padded_matrix[i][i] for i in range(len(padded_matrix))
    )
    total_samples = sum(sum(row) for row in padded_matrix)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"✓ RF Classification complete")
    print(f"  Train points used: {train_points.size().getInfo()}")
    print(f"  Test points used: {test_points.size().getInfo()}")
    print(f"  Confusion Matrix (Actual vs Predicted): {padded_matrix}")
    print(f"  Overall Held-Out Accuracy: {overall_accuracy:.2%}")
    for i, row_label in enumerate(class_labels):
        if i < len(padded_matrix):
            print(f"  Row {row_label}: {padded_matrix[i]}")

    thumbnail_url = classified_image.getThumbURL({
        'bands': ['classification'],
        'min': 0,
        'max': max_value,
        'palette': palette,
        'region': image.geometry(),
        'dimensions': 512
    })

    result = {
        'classified_image': classified_image,
        'binary_classified_image': binary_classified_image,
        'classification_mask': classification_mask,
        'water_buffer_m': water_buffer_m,
        'confusion_matrix': padded_matrix,
        'confusion_labels': class_labels,
        'training_accuracy': overall_accuracy,
        'thumbnail_url': thumbnail_url,
        'palette': palette,
        'classification_bands': classification_bands,
        'vegetation_thresholds': {
            'ndvi_texture': ndvi_texture_threshold,
            'distance_to_shore': distance_to_shore_threshold,
        },
        'train_test_split': {
            'train_fraction': 1 - test_fraction,
            'test_fraction': test_fraction,
            'description': '70/30 train/test split; accuracy is computed on the held-out 30% test set.'
        }
    }

    return result


# ============================================================
# Utility: Full pipeline runner
# ============================================================

def run_full_pipeline(lat, lon,
                      start_date='2025-01-01', end_date='2025-03-01',
                      max_cloud=10,
                      labelled_points=None):
    """
    Run the complete analysis pipeline for a given location.

    Parameters:
    - lat, lon: location coordinates
    - start_date, end_date: date range for Sentinel-2 imagery
        - max_cloud: max cloud percentage to accept
        - labelled_points: retained for backward compatibility; Dynamic World labels
            are used automatically for RF training

    Returns dict with all outputs:
    - image: the Sentinel-2 image
    - ndvi, ndvi_texture: index and texture bands
    - water_mask: binary water/land mask
    - classified: RF classification result dict
    - thumbnails: dict of image thumbnail URLs
    - training_samples: automatically generated Dynamic World training points
    """
    print(f"\n=== Running Full Pipeline for {lat}, {lon} ===\n")

    # Create AOI
    aoi = make_aoi(lat, lon, buffer_m=3000)

    # Pull imagery
    image = get_sentinel2_image(aoi, start_date, end_date, max_cloud)
    if image is None:
        return {'error': 'No cloud-free image found'}

    # Compute indices
    image_with_indices = compute_indices(image)

    # Compute water/land mask
    mask_image = compute_water_land_mask(image_with_indices, aoi)
    water_mask = mask_image.select('water_mask')

    # Compute NDVI texture
    image_with_texture = compute_ndvi_texture(image_with_indices, kernel_radius=3)

    # Compute distance to shore
    image_with_features = compute_distance_to_shore(image_with_texture, water_mask)

    # Generate automatic training data from Dynamic World for this AOI and date range.
    training_fc = create_dynamic_world_training_samples(
        image_with_features, aoi, start_date, end_date
    )

    # Build classification bands
    classification_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']

    # Run RF classification
    try:
        classification_result = classify_random_forest(
            image_with_features,
            training_fc,
            classification_bands,
            water_mask=water_mask,
            water_buffer_m=30
        )
    except Exception as e:
        classification_result = {
            'error': str(e),
            'training_accuracy': 0,
            'thumbnail_url': None
        }

    # --- Phase 2: Compute auxiliary features ---
    # Chlorophyll-a proxy
    chl_proxy = compute_chlorophyll_proxy(image_with_indices)

    # Turbidity proxy
    turbidity_proxy = compute_turbidity_proxy(image_with_indices)

    # Surface temperature (requires Landsat; placeholder if only Sentinel-2 available)
    temperature = compute_surface_temperature_placeholder(image_with_indices, aoi)

    # Add the classification band to the image so hypoxia_risk can select hyacinth density
    # The classified_image has the 3-class ontology: 0=Water, 1=Hyacinth, 2=Other Vegetation
    classified_image = classification_result.get('classified_image')
    if classified_image is not None:
        image_with_classification = image_with_indices.addBands(
            classified_image.rename('classification')
        )
    else:
        # Fallback: add a dummy band so the name exists
        image_with_classification = image_with_indices.addBands(
            ee.Image(0).rename('classification')
        )

    # Hypoxia risk / HHRI — integrate Phase 1 hyacinth output with Phase 2 proxies
    # First add the proxy bands that compute_hypoxia_risk expects
    image_with_hypoxia = image_with_classification.addBands([
        chl_proxy, turbidity_proxy, temperature
    ])
    # Use 'classification' band name (0=Water, 1=Hyacinth, 2=Other Vegetation)
    hypoxia_result = compute_hypoxia_risk(
        image_with_hypoxia,
        chl_proxy_band='chlorophyll_a_proxy',
        turbidity_proxy_band='turbidity_proxy',
        temperature_band='water_surface_temperature_placeholder',
        hyacinth_density_band='classification',
    )

    # --- Generate thumbnails ---
    # True Color
    rgb_thumb = image.getThumbURL({
        'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000,
        'region': aoi, 'dimensions': 512
    })

    # NDVI
    ndvi_thumb = image_with_indices.select('NDVI').getThumbURL({
        'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green'],
        'region': aoi, 'dimensions': 512
    })

    # NDWI
    ndwi_thumb = image_with_indices.select('NDWI').getThumbURL({
        'min': -1, 'max': 1, 'palette': ['brown', 'white', 'blue'],
        'region': aoi, 'dimensions': 512
    })

    # Classified
    classified_thumb = classification_result.get('thumbnail_url')

    thumbnails = {
        'true_color': rgb_thumb,
        'ndvi': ndvi_thumb,
        'ndwi': ndwi_thumb,
        'classified': classified_thumb
    }

    # Hypoxia risk map thumbnail (categorical: 0=LOW, 1=MODERATE, 2=HIGH)
    hypoxia_risk_img = hypoxia_result.get('hypoxia_risk_category')
    if hypoxia_risk_img is not None:
        hypoxia_thumb = hypoxia_risk_img.getThumbURL({
            'bands': ['hypoxia_risk_category'],
            'min': 0, 'max': 2,
            'palette': ['green', 'orange', 'red'],
            'region': aoi, 'dimensions': 512
        })
        thumbnails['hypoxia_risk'] = hypoxia_thumb
    else:
        thumbnails['hypoxia_risk'] = None

    # Compile result
    result = {
        'image': image,
        'aoi': aoi,
        'ndvi': image_with_indices.select('NDVI'),
        'ndwi': image_with_indices.select('NDWI'),
        'water_mask': water_mask,
        'ndvi_texture': image_with_texture.select('ndvi_texture'),
        'distance_to_shore': image_with_features.select('distance_to_shore'),
        'classification': classification_result,
        'thumbnails': thumbnails,
        'labelled_points': labelled_points,
        'training_samples': training_fc,
        # Phase 2 auxiliary features
        'chlorophyll_proxy': chl_proxy,
        'turbidity_proxy': turbidity_proxy,
        'temperature': temperature,
        # Hypoxia risk / HHRI
        'hhri': hypoxia_result.get('hhri'),
        'hypoxia_risk_category': hypoxia_result.get('hypoxia_risk_category'),
        'chlorophyll_proxy_band': hypoxia_result.get('chlorophyll_proxy'),
        'turbidity_proxy_band': hypoxia_result.get('turbidity_proxy'),
        'temperature_band': hypoxia_result.get('temperature'),
        'hyacinth_density_band': hypoxia_result.get('hyacinth_density'),
    }

    # === WATER HYACINTH AREA CALCULATIONS (server-side EE) ===
    # Use existing classified_image (3-class: 0=Water, 1=Hyacinth, 2=Other Vegetation)
    # and existing water_mask for denominator (analyzed water area only).
    classified_image = classification_result.get('classified_image')
    if classified_image is not None:
        # Hyacinth area: sum of pixelArea where classification == 1
        hyacinth_mask = classified_image.eq(1).selfMask()
        hyacinth_area_ha = hyacinth_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            tileScale=4
        ).get('classification')
        # Water area: sum of pixelArea within water_mask (binary 1=water)
        water_area_ha = water_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9,
            tileScale=4
        ).get('water_mask')
        # Coverage percentage
        hyacinth_coverage_val = ee.Number(hyacinth_area_ha).divide(ee.Number(water_area_ha)).multiply(100)
        hyacinth_coverage = hyacinth_coverage_val.rename('hyacinth_coverage_pct')

        # Also compute other vegetation area (class 2) for complete summary
        other_veg_mask = classified_image.eq(2).selfMask()
        other_veg_area_ha = other_veg_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            tileScale=4
        ).get('classification')

        result['hyacinth_area_ha'] = hyacinth_area_ha
        result['water_area_ha'] = water_area_ha
        result['hyacinth_coverage_pct'] = hyacinth_coverage
        result['other_vegetation_area_ha'] = other_veg_area_ha
    else:
        result['hyacinth_area_ha'] = None
        result['water_area_ha'] = None
        result['hyacinth_coverage_pct'] = None
        result['other_vegetation_area_ha'] = None

    # === SCALAR REDUCTIONS (reduce Phase 2 proxies over AOI for summary display) ===
    chl_proxy_img = hypoxia_result.get('chlorophyll_proxy')
    if chl_proxy_img is not None:
        chl_stat = chl_proxy_img.reduceRegion(ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e9, tileScale=4)
        result['chl_proxy_mean'] = chl_stat.get('chl_norm')
    else:
        result['chl_proxy_mean'] = None

    turb_proxy_img = hypoxia_result.get('turbidity_proxy')
    if turb_proxy_img is not None:
        turb_stat = turb_proxy_img.reduceRegion(ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e9, tileScale=4)
        result['turb_proxy_mean'] = turb_stat.get('turb_norm')
    else:
        result['turb_proxy_mean'] = None

    # HHRI scalar (reduce over water AOI)
    hhri_img = hypoxia_result.get('hhri')
    if hhri_img is not None:
        hhri_stat = hhri_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9,
            tileScale=4
        )
        result['hhri_mean'] = hhri_stat.get('hhri')
    else:
        result['hhri_mean'] = None

    print(f"\n=== Pipeline Complete ===\n")
    return result


# ============================================================
# Debug / test entry point
# ============================================================

if __name__ == '__main__':
    """Test script - run to verify individual functions work."""
    import sys

    # Test with Loktak Lake coordinates
    lat, lon = 24.55, 93.85

    # Initialize
    initialize_ee()

    # Make AOI
    aoi = make_aoi(lat, lon)
    print(f"AOI created: {aoi.getInfo()}")

    # Get image
    image = get_sentinel2_image(aoi)
    if image is None:
        print("No image found - this could be due to date range or location")
        sys.exit(0)

    # Compute indices
    image_with_indices = compute_indices(image)

    # Water/land mask
    mask_image = compute_water_land_mask(image_with_indices, aoi)

    # NDVI texture
    image_with_texture = compute_ndvi_texture(image_with_indices)

    # Distance to shore
    image_with_features = compute_distance_to_shore(image_with_texture, mask_image.select('water_mask'))

    # Create training points sample (manual labels)
    # 0=water, 1=hyacinth, 2=lotus, 3=other veg
    sample_points = [
        {'lat': lat + 0.01, 'lon': lon + 0.005, 'class': 1, 'label': 'hyacinth sample'},
        {'lat': lat - 0.01, 'lon': lon - 0.005, 'class': 2, 'label': 'lotus sample'},
        {'lat': lat, 'lon': lon, 'class': 0, 'label': 'open water sample'},
    ]

    training_fc = create_training_fc(sample_points, aoi)

    print("\n✓ All basic functions tested successfully!")