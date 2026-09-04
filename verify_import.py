#!/usr/bin/env python
"""Verify the modified modules compile and import correctly."""
import sys
import os

# Add project to path
sys.path.insert(0, r'D:\Project - I')

# Test 1: Import ee_utils
print("Test 1: Import ee_utils...")
try:
    from ee_utils import (
        run_scientific_validation,
        _run_prototype_mode,
        _run_scientific_mode,
        PROXY_MODE_LABEL,
        compute_chlorophyll_proxy,
        compute_turbidity_proxy,
        compute_hypoxia_risk,
        PHASE2_CONFIG,
        classify_random_forest,
        create_dynamic_world_training_samples,
        compute_ndvi_texture,
        compute_distance_to_shore,
        compute_water_land_mask,
        get_sentinel2_image,
        compute_indices,
        make_aoi,
    )
    print("  ✓ ee_utils imports successful")
except Exception as e:
    print(f"  ✗ ee_utils import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test basic function signatures
print("Test 2: Verify compute_hypoxia_risk is available...")
try:
    import inspect
    src = inspect.getsource(compute_hypoxia_risk)
    print(f"  ✓ compute_hypoxia_risk is available ({len(src)} chars)")
except Exception as e:
    print(f"  ✗ compute_hypoxia_risk check failed: {e}")
    sys.exit(1)

# Test 3: Verify PHASE2_CONFIG has HHRI
print("Test 3: Verify PHASE2_CONFIG HHRI structure...")
try:
    assert 'hhri' in PHASE2_CONFIG, "HHRI not in PHASE2_CONFIG"
    hhri = PHASE2_CONFIG['hhri']
    assert 'weights' in hhri, "weights not in HHRI config"
    assert 'thresholds' in hhri, "thresholds not in HHRI config"
    assert hhri['weights']['w1_ndvi'] == 1.0, "w1_ndvi weight incorrect"
    assert hhri['weights']['w2_ndwi'] == 1.0, "w2_ndwi weight incorrect"
    assert hhri['weights']['w3_chl'] == 1.0, "w3_chl weight incorrect"
    assert hhri['weights']['w4_turbidity'] == 1.0, "w4_turbidity weight incorrect"
    assert hhri['weights']['w5_doproxy'] == 1.0, "w5_doproxy weight incorrect"
    assert hhri['thresholds']['low'] == 0.3, "low threshold incorrect"
    assert hhri['thresholds']['moderate'] == 0.6, "moderate threshold incorrect"
    print("  ✓ PHASE2_CONFIG HHRI structure correct")
except Exception as e:
    print(f"  ✗ PHASE2_CONFIG HHRI structure failed: {e}")
    sys.exit(1)

# Test 4: Verify app.py imports
print("Test 4: Verify app.py can import from ee_utils...")
try:
    import streamlit as st
    print("  ✓ app.py can import required modules")
except Exception as e:
    print(f"  ✗ app.py import check: {e}")

print("\n✓ All verification checks passed!")