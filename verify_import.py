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

