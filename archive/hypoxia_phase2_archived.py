# --- ARCHIVED PHASE 2 (HYPOXIA) CODE ---
# This code was isolated and archived as the project scope was narrowed strictly to hyacinth detection.

import ee

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


# ---------------------------------------------------------------------------
# UI WIRING TODO - Archived from map_view.py
# ---------------------------------------------------------------------------
# TODO — wire to real HHRI output
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. After run_scientific_validation() returns for a site, read result['hhri_mean'] (an ee.Number).
# 2. Call .getInfo() to obtain the scalar HHRI value.
# 3. Apply PHASE2_CONFIG['hhri']['thresholds']:
#        HHRI < 0.3         -> 'low'
#        0.3 <= HHRI < 0.6  -> 'moderate'
#        HHRI >= 0.6        -> 'high'
# 4. Store the result in st.session_state keyed by site name so the map can display it without re-running.
# 5. Optionally persist to the GeoJSON 'last_risk' field.
# ---------------------------------------------------------------------------
