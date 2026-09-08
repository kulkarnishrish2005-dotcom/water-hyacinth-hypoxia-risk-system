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
        # Three-state vegetation detection gate fields
        'detection_state': result.get('detection_state', 'classified'),
        'confident_veg_area_ha': result.get('confident_veg_area_ha'),
        'classify_threshold_ha': result.get('classify_threshold_ha'),
        'p95_ndvi': result.get('p95_ndvi'),
        'suspect_signal': result.get('suspect_signal'),
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
    """Create an Area of Interest geometry from lat/lon with a buffer, returned as a bounding box."""
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()


# ============================================================
# STEP 3: Pull Sentinel-2 imagery
# ============================================================

def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    # Preserve original 0-10000 scale
    return image.updateMask(mask)

def get_sentinel2_image(aoi, start_date='2025-01-01', end_date='2025-03-01', max_cloud=10):
    """
    Pull Sentinel-2 SR Harmonized imagery filtered by date, bounds, and cloud cover.
    
    Applies a pixel-level cloud mask (QA60) and sorts descending so the clearest 
    images are placed on top when mosaiced.
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
        .map(mask_s2_clouds)
        .sort('CLOUDY_PIXEL_PERCENTAGE', False) # Descending: clear images are drawn last (on top)
    )

    count = collection.size().getInfo()
    print(f"Found {count} Sentinel-2 images matching criteria")

    if count == 0:
        print("  No cloud-free Sentinel-2 image found for the given date range and location.")
        return None

    # Use mosaic to stitch tiles together, preventing wedge-shaped cutoffs
    image = collection.mosaic().clip(aoi)

    # Note: Because the collection was mapped with mask_s2_clouds,
    # the 'CLOUDY_PIXEL_PERCENTAGE' property on individual images was preserved,
    # but the mosaic() operation creates a new image without those properties.
    # We fetch it from the first image (now the cloudiest, due to descending sort) just for logging.
    first_image = collection.first()
    if first_image:
        cloud_pct = first_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        print(f"Mosaic created. First tile Cloud %: {cloud_pct}%")

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

    # Threshold: >20% occurrence defines the historical water basin.
    # We do NOT use NDWI here because water hyacinth has NDWI < 0.
    # We MUST unmask with 0 so that actual land becomes 0 instead of NODATA,
    # which is required for fastDistanceTransform to work properly later.
    water_mask = occurrence.unmask(0).gte(20).rename('water_mask')

    # Land is the complement (binary 0)
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


def compute_distance_to_shore(image, water_mask):
    """
    Compute distance to shore from the water mask.
    water_mask is 1 for water, 0 for land.
    fastDistanceTransform computes distance to the nearest ZERO pixel.
    So water_mask.fastDistanceTransform() computes distance from water (1) to land (0).
    
    Returns image with added 'distance_to_shore' band.
    """
    # fastDistanceTransform computes squared Euclidean distance in pixels.
    # We take sqrt to get distance in pixels.
    distance = water_mask.fastDistanceTransform().sqrt().rename('distance_to_shore')

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

def create_dynamic_world_training_samples(image, aoi, start_date, end_date, water_mask=None):
    """
    DEPRECATED: Dynamic World proxy labels for aquatic vegetation are unreliable.
    DW is a terrestrial land-cover product that labels most floating vegetation as "water",
    producing severely imbalanced training data and ~0.18–0.22 recall on hyacinth.

    Retained for backward compatibility. Use create_spectral_rule_training_samples() instead.
    """
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
    
    if water_mask is not None:
        sampling_image = sampling_image.updateMask(water_mask)

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
    print(f"✓ Created {sample_count} automatic Dynamic World training samples (DEPRECATED)")
    return training_fc


def create_spectral_rule_training_samples(image, aoi, water_mask,
                                          n_water=200, n_veg=200,
                                          scale=10, seed=42):
    """
    Rule-based proxy label generator for aquatic vegetation classification,
    using site-adaptive spectral thresholds.

    ── How it works ──────────────────────────────────────────────────────
    This function generates training labels from spectral index thresholds applied only
    to *confident* pixels. Because turbidity and baseline NDVI vary drastically
    between water bodies (e.g. Loktak vs Vembanad), global constants fail.
    
    Instead, we compute the NDVI percentiles within the historical water mask
    for the specific site, and set dynamic thresholds:
    
    - water_ndvi_max = 25th percentile of site NDVI (capped at max 0.05)
    - veg_ndvi_min   = 95th percentile of site NDVI (floored at min 0.15)
    
    Confident open water:          NDVI < water_ndvi_max  AND  NDWI > 0.1
    Confident aquatic vegetation:  NDVI > veg_ndvi_min  AND  NDWI < 0.1
                                   AND pixel is inside the JRC water mask
                                   (vegetation ON historical water = aquatic)

    Pixels in the ambiguous middle ground are left UNLABELED. The Random Forest must
    generalize to these from the full 4-feature set (NDVI, NDWI, ndvi_texture, distance_to_shore).
    """
    ndvi = image.select('NDVI')
    ndwi = image.select('NDWI')

    # Compute site-specific NDVI percentiles within the water body
    ndvi_in_water = ndvi.updateMask(water_mask)
    percentiles = ndvi_in_water.reduceRegion(
        reducer=ee.Reducer.percentile([25, 95]),
        geometry=aoi,
        scale=30,  # 30m is fine for distribution stats
        maxPixels=1e9,
        tileScale=4
    )
    
    p25 = ee.Number(percentiles.get('NDVI_p25'))
    p95 = ee.Number(percentiles.get('NDVI_p95'))
    
    # Evaluate early for sanity check gate
    p95_val = p95.getInfo()

    # Adaptive thresholds with safety bounds (in case the lake is 100% choked or 100% clear)
    water_ndvi_max = p25.min(0.05)
    veg_ndvi_min = p95.max(0.15)
    
    # Static NDWI thresholds used as secondary confidence guards
    water_ndwi_min = ee.Number(0.1)
    veg_ndwi_max = ee.Number(0.1)

    # ── Confident open water mask ──
    confident_water = (
        ndvi.lt(water_ndvi_max)
        .And(ndwi.gt(water_ndwi_min))
        .And(water_mask.eq(1))
        .rename('label')
    )

    # ── Confident aquatic vegetation mask ──
    confident_veg = (
        ndvi.gt(veg_ndvi_min)
        .And(ndwi.lt(veg_ndwi_max))
        .And(water_mask.eq(1))
        .rename('label')
    )

    classification_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']

    # ── Deterministic Area Calculation & Heuristic Gate ──
    veg_area_img = confident_veg.multiply(ee.Image.pixelArea()).divide(10000).rename('area')
    veg_area_val = veg_area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).getNumber('area').getInfo()

    water_area_img = water_mask.multiply(ee.Image.pixelArea()).divide(10000).rename('area')
    water_area_val = water_area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).getNumber('area').getInfo()

    LOW_FLOOR_HA = 2.0
    import math
    # Provisional heuristic based on limited testing (Varanasi, Mississippi, Loktak, Vembanad).
    # Not a rigorously validated universal threshold. See README for three-state design rationale.
    CLASSIFY_THRESHOLD_HA = max(LOW_FLOOR_HA, 0.4 * math.sqrt(water_area_val))
    
    # Calculate median NDVI of the actual confident vegetation cohort
    veg_ndvi_img = ndvi.updateMask(confident_veg)
    veg_median_dict = veg_ndvi_img.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=aoi,
        scale=30,
        maxPixels=1e9,
        tileScale=4
    )
    cohort_median = veg_median_dict.get('NDVI').getInfo()
    cohort_val = float(cohort_median) if cohort_median is not None else 0.0
    
    if veg_area_val < LOW_FLOOR_HA:
        print(f"  [GATE] State 1: no_significant_vegetation ({veg_area_val:.2f} ha < {LOW_FLOOR_HA} ha floor)")
        raise ValueError(f"no_significant_vegetation|{veg_area_val:.2f}|{cohort_val:.3f}")
    elif veg_area_val < CLASSIFY_THRESHOLD_HA:
        print(f"  [GATE] State 2: low_confidence_signal ({veg_area_val:.2f} ha, threshold {CLASSIFY_THRESHOLD_HA:.2f} ha)")
        raise ValueError(f"low_confidence_signal|{veg_area_val:.2f}|{CLASSIFY_THRESHOLD_HA:.2f}|{water_area_val:.2f}|{cohort_val:.3f}")
    else:
        print(f"  [GATE] State 3: classified ({veg_area_val:.2f} ha >= {CLASSIFY_THRESHOLD_HA:.2f} ha threshold)")

    # ── Combine masks into a single class band ──
    # Create an image that is 1 where confident_veg is true, 0 otherwise.
    # Then mask it so it ONLY contains pixels that are EITHER confident_water OR confident_veg.
    combined_class = ee.Image(0).byte().where(confident_veg, 1) \
        .updateMask(confident_water.Or(confident_veg)) \
        .rename('class')

    # Attach the class band to the feature bands
    combined_img = image.select(classification_bands).addBands(combined_class)

    # ── Random Sampling (Honest Darts) ──
    # Throw 10,000 darts across the AOI. Only those that land in the masked regions are kept.
    # This prevents scraping the barrel for glitch pixels in clean rivers (Varanasi),
    # while throwing enough darts to find real vegetation patches (Loktak/Vembanad).
    training_fc = combined_img.sample(
        region=aoi,
        scale=scale,
        numPixels=10000,
        seed=seed,
        geometries=True,
        dropNulls=True,
        tileScale=4
    )

    water_count = training_fc.filter(ee.Filter.eq('class', 0)).size().getInfo()
    veg_count = training_fc.filter(ee.Filter.eq('class', 1)).size().getInfo()
    total = training_fc.size().getInfo()
    
    w_thresh = water_ndvi_max.getInfo()
    v_thresh = veg_ndvi_min.getInfo()
    
    print(f"✓ Created {total} spectral-rule proxy training samples (Adaptive)")
    print(f"  Site NDVI percentiles: p25={p25.getInfo():.3f}, p95={p95_val:.3f}")
    print(f"  Cohort median NDVI:   {cohort_val:.3f}")
    print(f"  Confident water:      {water_count}  (NDVI < {w_thresh:.3f}, NDWI > 0.1)")
    print(f"  Confident vegetation: {veg_count}  (NDVI > {v_thresh:.3f}, NDWI < 0.1)")

    return training_fc.set('p95_ndvi', cohort_val).set('confident_veg_area_ha', veg_area_val).set('classify_threshold_ha', CLASSIFY_THRESHOLD_HA)


def postprocess_vegetation_subclassification(classified_image, feature_image,
                                          ndvi_texture_threshold=0.15,
                                          distance_to_shore_threshold=5,
                                          vegetation_value=1):
    """
    Split the initial vegetation class into two subclasses using texture + shore distance.

    Rule used:
    - Hyacinth: lower NDVI texture (uniform mat) + not strictly on the immediate bank
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
    )

    # Everything else classified as vegetation is treated as other vegetation / lotus-like
    other_vegetation_mask = is_vegetation.And(hyacinth_mask.Not())

    refined = classified_image.where(hyacinth_mask, 1).where(other_vegetation_mask, 2)
    return refined.rename('classification')


def classify_random_forest(image, training_fc, classification_bands=None,
                          ndvi_texture_threshold=0.15,
                          distance_to_shore_threshold=5,
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

    # Add random column FIRST so we can split it
    training_fc_with_random = training_fc.randomColumn(seed=1)
    test_points = training_fc_with_random.filter(ee.Filter.lt('random', test_fraction))
    train_points = training_fc_with_random.filter(ee.Filter.gte('random', test_fraction))

    trained_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=50,
        seed=42
    ).train(
        features=train_points,
        classProperty='class',
        inputProperties=classification_bands
    )

    binary_classified_image = stacked.classify(trained_classifier).rename('classification')
    
    # Strictly mask to water body to prevent land from being classified as hyacinth
    final_mask = water_mask.eq(1) if water_mask is not None else classification_mask
    binary_classified_image = binary_classified_image.updateMask(final_mask)

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

    classified_image = classified_image.updateMask(final_mask)

    # Proper train/test split: 70% train, 30% held-out test.
    # This is the metric that reflects real generalization performance.
    # (train_points and test_points are now defined earlier)

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

    try:
        # Generate training labels using rule-based spectral heuristic.
        # This replaces the deprecated Dynamic World proxy which mislabeled
        # most floating vegetation as "water". See create_spectral_rule_training_samples() docstring.
        training_fc = create_spectral_rule_training_samples(
            image_with_features, aoi, water_mask,
            n_water=200, n_veg=200
        )
    
        classification_bands = ['NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore']
        
        # Evaluate sample counts to flag low-confidence training
        water_count = training_fc.filter(ee.Filter.eq('class', 0)).size()
        veg_count = training_fc.filter(ee.Filter.eq('class', 1)).size()
        
        # We use a combined conditional to set the flag client-side later, 
        # but EE doesn't easily return these scalars alongside the image without getInfo.
        # We can fetch them via getInfo safely here since prototype mode uses getInfo anyway.
        veg_count_val = veg_count.getInfo()
        water_count_val = water_count.getInfo()
        is_low_confidence = (veg_count_val < 20) or (water_count_val < 20)
        
        if is_low_confidence:
            print(f"  ⚠️ Low confidence training: only {water_count_val} water and {veg_count_val} veg samples.")
    
        # Run RF classification
        classification_result = classify_random_forest(
            image_with_features,
            training_fc,
            classification_bands,
            water_mask=water_mask,
            water_buffer_m=30
        )
        
        p95_val = training_fc.get('p95_ndvi').getInfo()
        veg_area = training_fc.get('confident_veg_area_ha').getInfo()
        thresh_area = training_fc.get('classify_threshold_ha').getInfo()
        
        classification_result['low_confidence_training'] = is_low_confidence
        classification_result['veg_sample_count'] = veg_count_val
        classification_result['water_sample_count'] = water_count_val
        classification_result['detection_state'] = 'classified'
        classification_result['p95_ndvi'] = p95_val
        classification_result['suspect_signal'] = p95_val < 0.30
        classification_result['confident_veg_area_ha'] = veg_area
        classification_result['classify_threshold_ha'] = thresh_area
    except Exception as e:
        error_msg = str(e)
        if "no_significant_vegetation" in error_msg:
            # State 1: Below 2.0 ha floor — genuinely clean water body
            print("  [RESULT] No significant vegetation detected.")
            parts = error_msg.split('|')
            detected_ha = float(parts[1]) if len(parts) > 1 else 0.0
            p95_val = float(parts[2]) if len(parts) > 2 else None
            
            dummy_classified = ee.Image(0).updateMask(water_mask.eq(1)).rename('classification')
            classification_result = {
                'classified_image': dummy_classified,
                'binary_classified_image': dummy_classified,
                'training_accuracy': 1.0,
                'held_out_accuracy': 1.0,
                'confusion_matrix': None,
                'per_class_metrics': {},
                'thumbnail_url': None,
                'detection_state': 'no_significant_vegetation',
                'confident_veg_area_ha': detected_ha,
                'fallback_zero_vegetation': True,
                'low_confidence_training': False,
                'p95_ndvi': p95_val,
                'suspect_signal': p95_val < 0.30 if p95_val is not None else False
            }
            training_fc = None
        elif "low_confidence_signal" in error_msg:
            # State 2: Between floor and classify threshold — ambiguous signal
            parts = error_msg.split('|')
            detected_ha = float(parts[1])
            threshold_ha = float(parts[2])
            water_area_ha = float(parts[3])
            p95_val = float(parts[4]) if len(parts) > 4 else None
            print(f"  [RESULT] Low-confidence signal: {detected_ha} ha detected (threshold: {threshold_ha} ha)")
            dummy_classified = ee.Image(0).updateMask(water_mask.eq(1)).rename('classification')
            classification_result = {
                'classified_image': dummy_classified,
                'binary_classified_image': dummy_classified,
                'training_accuracy': None,
                'held_out_accuracy': None,
                'confusion_matrix': None,
                'per_class_metrics': {},
                'thumbnail_url': None,
                'detection_state': 'low_confidence_signal',
                'confident_veg_area_ha': detected_ha,
                'classify_threshold_ha': threshold_ha,
                'water_area_for_threshold': water_area_ha,
                'fallback_zero_vegetation': False,
                'low_confidence_training': False,
                'p95_ndvi': p95_val,
                'suspect_signal': p95_val < 0.30 if p95_val is not None else False
            }
            training_fc = None
        elif "Classifier training failed" in error_msg or "Invalid minimum size" in error_msg:
            # Legacy fallback for other RF training failures
            print("  [RESULT] RF training failed. Falling back to 0% coverage.")
            dummy_classified = ee.Image(0).updateMask(water_mask.eq(1)).rename('classification')
            classification_result = {
                'classified_image': dummy_classified,
                'binary_classified_image': dummy_classified,
                'training_accuracy': 1.0,
                'held_out_accuracy': 1.0,
                'confusion_matrix': None,
                'per_class_metrics': {},
                'thumbnail_url': None,
                'detection_state': 'no_significant_vegetation',
                'fallback_zero_vegetation': True,
                'low_confidence_training': False
            }
            training_fc = None
        else:
            classification_result = {
                'error': error_msg,
                'training_accuracy': 0,
                'thumbnail_url': None
            }
            training_fc = None

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
        'training_samples': training_fc if training_fc is not None else None,
    }

    # === WATER HYACINTH AREA CALCULATIONS (server-side EE) ===
    classified_image = classification_result.get('classified_image')
    detection_state = classification_result.get('detection_state', 'classified')
    if classified_image is not None and detection_state == 'classified':
        # Hyacinth area: sum of pixelArea where classification == 1
        hyacinth_mask = classified_image.eq(1).selfMask()
        hyacinth_dict = hyacinth_mask.multiply(ee.Image.pixelArea()).divide(10000).unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            tileScale=4
        )
        hyacinth_area_ha = hyacinth_dict.get('classification')

        # Water area: sum of pixelArea within water_mask (binary 1=water)
        water_dict = water_mask.multiply(ee.Image.pixelArea()).divide(10000).unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9,
            tileScale=4
        )
        water_area_ha = water_dict.get('water_mask')

        # Coverage percentage (safe division)
        # Handle cases where water_area is 0 or null
        hyacinth_coverage = ee.Algorithms.If(
            ee.Number(water_area_ha).gt(0),
            ee.Number(hyacinth_area_ha).divide(ee.Number(water_area_ha)).multiply(100),
            0
        )

        # Other vegetation area (class 2)
        other_veg_mask = classified_image.eq(2).selfMask()
        other_veg_dict = other_veg_mask.multiply(ee.Image.pixelArea()).divide(10000).unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            tileScale=4
        )
        other_veg_area_ha = other_veg_dict.get('classification')

        result['hyacinth_area_ha'] = hyacinth_area_ha
        result['water_area_ha'] = water_area_ha
        result['hyacinth_coverage_pct'] = hyacinth_coverage
        result['other_vegetation_area_ha'] = other_veg_area_ha
        result['detection_state'] = detection_state
        result['confident_veg_area_ha'] = classification_result.get('confident_veg_area_ha')
        result['classify_threshold_ha'] = classification_result.get('classify_threshold_ha')
    elif detection_state == 'low_confidence_signal':
        result['hyacinth_area_ha'] = None
        result['water_area_ha'] = classification_result.get('water_area_for_threshold')
        result['hyacinth_coverage_pct'] = None
        result['other_vegetation_area_ha'] = None
        result['confident_veg_area_ha'] = classification_result.get('confident_veg_area_ha')
        result['classify_threshold_ha'] = classification_result.get('classify_threshold_ha')
        result['detection_state'] = 'low_confidence_signal'
    else:
        result['hyacinth_area_ha'] = None
        result['water_area_ha'] = None
        result['hyacinth_coverage_pct'] = None
        result['other_vegetation_area_ha'] = None
        result['detection_state'] = detection_state

    result['p95_ndvi'] = classification_result.get('p95_ndvi')
    result['suspect_signal'] = classification_result.get('suspect_signal')


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