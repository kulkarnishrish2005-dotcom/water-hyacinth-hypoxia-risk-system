"""
Water Hyacinth Detection System - Streamlit UI (Phase 1)

Two-page app:
  Page 1 "Search": text input for city name → discover water bodies
  Page 2 "Analysis": select water body → run full pipeline → display outputs

Uses: streamlit, geopy, earthengine-api, geemap, matplotlib
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import geemap
import os
from datetime import datetime

# Import our Earth Engine utilities
from ee_utils import (
    EE_PROJECT_ID,
    initialize_ee,
    make_aoi,
    get_sentinel2_image,
    compute_indices,
    compute_water_land_mask,
    discover_water_bodies,
    compute_ndvi_texture,
    compute_distance_to_shore,
    create_training_fc,
    classify_random_forest,
    run_full_pipeline,
    compute_ndvi_difference,
    run_scientific_validation,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION & SESSION STATE INIT
# ---------------------------------------------------------

def init_session_state():
    """Initialize streamlit session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'search'  # or 'analysis'
    if 'selected_water_body' not in st.session_state:
        st.session_state.selected_water_body = None
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'city_searched' not in st.session_state:
        st.session_state.city_searched = None
    if 'water_bodies' not in st.session_state:
        st.session_state.water_bodies = []
    if 'ground_truth_path' not in st.session_state:
        st.session_state.ground_truth_path = None
    if 'mode' not in st.session_state:
        st.session_state.mode = 'prototype'


def switch_page(page):
    """Switch between app pages."""
    st.session_state.page = page


# ---------------------------------------------------------
# PAGE 1: SEARCH
# ---------------------------------------------------------

def page_search():
    """Page 1: Search for water bodies by city name."""
    st.title("🔍 Water Hyacinth Detection — Search Page")
    st.markdown("### Search-by-city: find water bodies near your location of interest")

    with st.form(key='search_form', clear_on_submit=False):
        city_name = st.text_input(
            "Enter a city or place name:",
            placeholder="e.g., Pune, Mumbai, Loktak Lake",
            help="Will search for water bodies within 15 km radius using JRC GSW dataset"
        )
        search_submitted = st.form_submit_button("🔍 Search Water Bodies")

    if search_submitted:
        if not city_name.strip():
            st.error("Please enter a city name.")
            return

        with st.spinner(f"Geocoding '{city_name}' and discovering water bodies..."):
            try:
                # Discover water bodies using JRC GSW
                result = discover_water_bodies(city_name, radius_km=25)

                st.session_state.city_searched = city_name
                st.session_state.water_bodies = result.get('water_bodies', [])

                # Store basic search info
                st.session_state.search_center = {
                    'lat': result.get('center_lat'),
                    'lon': result.get('center_lon'),
                    'polygon_count': result.get('polygon_count', 0)
                }

                # Determine mode based on whether ground truth CSV is available
                ground_truth_path = st.session_state.get('ground_truth_path', None)
                if ground_truth_path and os.path.exists(ground_truth_path):
                    mode = 'scientific'
                else:
                    mode = 'prototype'

                st.session_state.mode = mode

                st.session_state.page = 'search'

                if result.get('polygon_count', 0) > 0:
                    if mode == 'scientific':
                        st.success(f"✓ Found {result['polygon_count']} water body(ies) near {city_name}. "
                                   "Select a water body to configure scientific validation splits.")
                    else:
                        st.success(f"✓ Found {result['polygon_count']} water body(ies) near {city_name}. "
                                   "Running in prototype / proxy mode — Dynamic World labels are automatic proxies only.")
                else:
                    st.warning("⚠️ No water bodies detected in the specified radius. "
                               "Try adjusting the date range or trying another city.")

            except ValueError as e:
                st.error(f"Geocoding error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display discovered water bodies
    if st.session_state.get('water_bodies'):
        st.markdown("### 📍 Detected Water Bodies")

        for i, wb in enumerate(st.session_state['water_bodies']):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Water Body {i+1}**")
                st.caption(f"Centroid: {wb.get('centroid_lat', 'N/A')}, {wb.get('centroid_lon', 'N/A')}")

            with col2:
                est_size = wb.get('estimated_size_ha', 'N/A')
                if isinstance(est_size, (int, float)):
                    st.markdown(f"{est_size:.1f} ha")
                else:
                    st.markdown(f"{est_size}")

            with col3:
                if st.button(f"Select", key=f"select_{i}", use_container_width=True):
                    st.session_state.selected_water_body = wb
                    st.session_state.page = 'analysis'
                    st.rerun()

        # Also show the search center
        if st.session_state.get('search_center'):
            st.markdown(f"**Search Center:** {st.session_state['search_center']['lat']}, "
                        f"{st.session_state['search_center']['lon']}")


# ---------------------------------------------------------
# PAGE 2: ANALYSIS
# ---------------------------------------------------------

def page_analysis():
    """Page 2: Run full pipeline on selected water body."""
    st.title("📊 Water Hyacinth Detection — Analysis Page")
    st.markdown("### Full pipeline: data pull → indices → masking → RF classification")

    # Check we have a selected water body
    if not st.session_state.get('selected_water_body'):
        st.info("👈 Please select a water body from the Search page first.")
        if st.button("← Go Back to Search"):
            switch_page('search')
            st.rerun()
        return

    wb = st.session_state.selected_water_body
    lat, lon = wb['centroid_lat'], wb['centroid_lon']

    st.markdown(f"**Selected Water Body:** Centroid at {lat}, {lon}")

    # Determine mode
    mode = st.session_state.get('mode', 'prototype')
    ground_truth_path = st.session_state.get('ground_truth_path', None)

    with st.form(key='analysis_form'):
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start date",
                value=datetime(2025, 1, 1),
                help="Sentinel-2 imagery date range start"
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=datetime(2025, 3, 1),
                help="Sentinel-2 imagery date range end"
            )

        # Cloud cover threshold
        max_cloud = st.slider(
            "Max cloud cover %",
            min_value=5,
            max_value=25,
            value=10,
            help="Only use Sentinel-2 images with cloud cover below this threshold"
        )

        # Initialize validation split variables (will be None in prototype mode)
        train_sites = None
        validation_sites = None
        train_regions = None
        validation_regions = None
        train_sectors = None
        validation_sectors = None

        # Validation split configuration (scientific mode only)
        if mode == 'scientific':
            st.markdown("#### Validation Split Configuration")
            col1, col2 = st.columns(2)
            with col1:
                train_sites = st.text_input(
                    "Train sites (comma-separated, optional)",
                    placeholder="e.g., site_a,site_b"
                )
            with col2:
                validation_sites = st.text_input(
                    "Validation sites (comma-separated, optional)",
                    placeholder="e.g., site_c"
                )
            col3, col4 = st.columns(2)
            with col3:
                train_regions = st.text_input(
                    "Train regions (comma-separated, optional)",
                    placeholder="e.g., north,south"
                )
            with col4:
                validation_regions = st.text_input(
                    "Validation regions (comma-separated, optional)",
                    placeholder="e.g., east,west"
                )
            col5, col6 = st.columns(2)
            with col5:
                train_sectors = st.text_input(
                    "Train sectors (comma-separated, optional)",
                    placeholder="e.g., north,center"
                )
            with col6:
                validation_sectors = st.text_input(
                    "Validation sectors (comma-separated, optional)",
                    placeholder="e.g., south,east"
                )

        run_analysis = st.form_submit_button("▶️ Run Full Pipeline")

    if run_analysis:
        with st.spinner("Running full analysis pipeline — this may take 30-60 seconds..."):
            try:
                # Prepare split parameters
                train_sites_list = [s.strip() for s in train_sites.split(',') if s.strip()] if train_sites else None
                validation_sites_list = [s.strip() for s in validation_sites.split(',') if s.strip()] if validation_sites else None
                train_regions_list = [s.strip() for s in train_regions.split(',') if s.strip()] if train_regions else None
                validation_regions_list = [s.strip() for s in validation_regions.split(',') if s.strip()] if validation_regions else None
                train_sectors_list = [s.strip() for s in train_sectors.split(',') if s.strip()] if train_sectors else None
                validation_sectors_list = [s.strip() for s in validation_sectors.split(',') if s.strip()] if validation_sectors else None

                # Run the full pipeline with proper mode
                if mode == 'scientific':
                    results = run_scientific_validation(
                        ground_truth_path=ground_truth_path,
                        analysis_date=datetime(start_date.year, start_date.month, start_date.day),
                        lat=lat,
                        lon=lon,
                        train_sites=train_sites_list,
                        validation_sites=validation_sites_list,
                        train_regions=train_regions_list,
                        validation_regions=validation_regions_list,
                        train_sectors=train_sectors_list,
                        validation_sectors=validation_sectors_list,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        max_cloud=max_cloud,
                        scale=10,
                        mode='scientific'
                    )
                else:
                    # Prototype mode: no ground-truth CSV needed, no validation splits needed
                    results = run_scientific_validation(
                        analysis_date=datetime(start_date.year, start_date.month, start_date.day),
                        lat=lat,
                        lon=lon,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        max_cloud=max_cloud,
                        scale=10,
                        mode='prototype'
                    )

                if 'error' in results:
                    st.error(f"❌ Pipeline error: {results['error']}")
                    return

                # Store results in session state
                st.session_state.pipeline_results = results

                # Store ground truth path for future runs
                st.session_state.ground_truth_path = ground_truth_path
                st.session_state.mode = mode

                # Default to analysis page
                st.session_state.page = 'analysis'
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error running pipeline: {e}")
                import traceback
                st.exception(traceback.format_exc())

    # Display results if available
    if st.session_state.get('pipeline_results'):
        results = st.session_state.pipeline_results

        thumbnails = results.get('thumbnails', {})

        # Mode label
        mode_label = results.get('scientific_note', '')
        if 'Prototype' in mode_label:
            st.warning(mode_label)
        else:
            st.success(mode_label)

        # =================================================================
        # 1. WATER HYACINTH DETECTION
        # =================================================================
        st.markdown("# 🌿 WATER HYACINTH DETECTION")

        st.markdown("**Where is the water hyacinth and how much was detected?**")

        # ---- A. Summary Metrics ----
        st.markdown("## A. Summary Metrics")

        col_a, col_b, col_c, col_d = st.columns(4)

        # Helper to safely get scalar value from EE Number
        def _get_scalar(ee_obj, label="value"):
            if ee_obj is None:
                return None
            try:
                info = ee_obj.getInfo()
                if isinstance(info, dict):
                    vals = [v for v in info.values() if v is not None]
                    if vals:
                        info = vals[0]
                if isinstance(info, (int, float)):
                    return float(info)
            except Exception:
                pass
            return None

        # Water area (m²) → km²
        water_area_ha_val = _get_scalar(results.get('water_area_ha'))
        if water_area_ha_val is not None:
            water_area_km2 = water_area_ha_val / 100.0  # 1 km² = 100 ha
            with col_a:
                st.metric("Total Water Area Analyzed", f"{water_area_km2:.2f} km²")
        else:
            with col_a:
                st.metric("Total Water Area Analyzed", "N/A")

        # Hyacinth area (m²) → km²
        hyacinth_area_ha_val = _get_scalar(results.get('hyacinth_area_ha'))
        if hyacinth_area_ha_val is not None:
            hyacinth_area_km2 = hyacinth_area_ha_val / 100.0
            with col_b:
                st.metric("Water Hyacinth Area", f"{hyacinth_area_km2:.2f} km²")
        else:
            with col_b:
                st.metric("Water Hyacinth Area", "N/A")

        # Hyacinth coverage %
        coverage_val = _get_scalar(results.get('hyacinth_coverage_pct'))
        with col_c:
            if coverage_val is not None:
                st.metric("Water Hyacinth Coverage", f"{coverage_val:.2f} %")
            else:
                st.metric("Water Hyacinth Coverage", "N/A")

        # Other Vegetation area
        other_veg_ha_val = _get_scalar(results.get('other_vegetation_area_ha'))
        with col_d:
            if other_veg_ha_val is not None:
                other_veg_km2 = other_veg_ha_val / 100.0
                st.metric("Other Vegetation Area", f"{other_veg_km2:.2f} km²")
            else:
                st.metric("Other Vegetation Area", "N/A")

        # Show the actual formulas used
        st.caption(
            f"**Calculation method (server-side EE):** "
            f"Hyacinth area = Σ pixelArea where classification == 1; "
            f"Water area = Σ pixelArea within water_mask; "
            f"Coverage = (Hyacinth area / Water area) × 100. "
            f"Other Vegetation = Σ pixelArea where classification == 2."
        )

        # ---- B. Water Hyacinth Classification Map ----
        st.markdown("## B. Water Hyacinth Classification")
        st.caption(
            "Legend: 🟦 Water (class 0) · 🟩 Water Hyacinth (class 1) · 🟪 Other Vegetation (class 2)"
        )

        if thumbnails.get('classified'):
            st.image(
                thumbnails['classified'],
                use_container_width=True,
                caption="Random Forest Classification — Water / Water Hyacinth / Other Vegetation"
            )
        else:
            st.info("Classification map not available")

        # ---- C. Supporting Remote-Sensing Outputs ----
        st.markdown("## C. Supporting Remote-Sensing Outputs")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### True Color")
            if thumbnails.get('true_color'):
                st.image(thumbnails['true_color'], use_container_width=True, caption="Sentinel-2 True Color")
            else:
                st.info("True color not available")

        with col2:
            st.markdown("#### Water / Land Mask")
            if results.get('water_mask') is not None:
                water_mask_thumb = results['water_mask'].getThumbURL({
                    'bands': ['water_mask'],
                    'min': 0, 'max': 1,
                    'region': results['aoi'], 'dimensions': 512
                })
                st.image(water_mask_thumb, use_container_width=True, caption="Water / Land Mask")
            else:
                st.info("Water mask not available")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### NDVI (Vegetation / Hyacinth)")
            if thumbnails.get('ndvi'):
                st.image(thumbnails['ndvi'], use_container_width=True, caption="NDVI Index")
            else:
                st.info("NDVI not available")

        with col4:
            st.markdown("#### NDWI (Water)")
            if thumbnails.get('ndwi'):
                st.image(thumbnails['ndwi'], use_container_width=True, caption="NDWI Index")
            else:
                st.info("NDWI not available")

        # =================================================================
        # 2. HYPOXIA RISK ASSESSMENT
        # =================================================================
        st.markdown("# ⚠️ HYPOXIA RISK ASSESSMENT")
        st.markdown("**Estimated Hypoxia Risk** — satellite-derived proxy; not a direct dissolved oxygen measurement.")
        st.caption(
            "HHRI = w1·NDVI_norm + w2·(1−NDWI)_norm + w3·Chl_norm + w4·Turbidity_norm − w5·Temp_norm. "
            "Thresholds: LOW < 0.3 ≤ MODERATE < 0.6 ≤ HIGH (configurable in PHASE2_CONFIG)."
        )

        # ---- D. Phase 2 Metrics ----
        st.markdown("## D. Phase 2 Metrics")

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        # Chlorophyll proxy band mean (from reduced scalar)
        chl_mean = _get_scalar(results.get('chl_proxy_mean'))
        with col_p1:
            if chl_mean is not None:
                st.metric("Chlorophyll-a Proxy", f"{chl_mean:.4f}")
            else:
                st.metric("Chlorophyll-a Proxy", "N/A")
            st.caption("Gitelson red-edge · NOT measured Chl-a")

        # Turbidity proxy band mean (from reduced scalar)
        turb_mean = _get_scalar(results.get('turb_proxy_mean'))
        with col_p2:
            if turb_mean is not None:
                st.metric("Turbidity Proxy", f"{turb_mean:.4f}")
            else:
                st.metric("Turbidity Proxy", "N/A")
            st.caption("Red/NIR band ratio · NOT measured turbidity")

        # Temperature (placeholder)
        temp_band = results.get('temperature_band')
        temp_mean = _get_scalar(temp_band) if temp_band is not None else None
        with col_p3:
            if temp_mean is not None:
                st.metric("Surface Temperature", f"{temp_mean:.4f} (norm.)")
            else:
                st.metric("Surface Temperature", "Placeholder")
            st.caption("Sentinel-2 has no thermal bands · placeholder")

        # HHRI scalar (reduced over water AOI)
        hhri_mean_obj = results.get('hhri_mean')
        hhri_scalar_val = _get_scalar(hhri_mean_obj)
        with col_p4:
            if hhri_scalar_val is not None:
                st.metric("HHRI", f"{hhri_scalar_val:.2f}")
            else:
                st.metric("HHRI", "N/A")
            st.caption("HHRI mean over water AOI · weighted proxy index (values may exceed 1)")

        # ---- E. Hypoxia Risk Classification ----
        st.markdown("## E. Estimated Hypoxia Risk Level")

        # Determine risk level from HHRI scalar using existing thresholds
        low_thr = 0.3
        mod_thr = 0.6
        if hhri_scalar_val is not None:
            if hhri_scalar_val < low_thr:
                risk_level = "LOW"
                risk_color = "🟢"
            elif hhri_scalar_val < mod_thr:
                risk_level = "MODERATE"
                risk_color = "🟡"
            else:
                risk_level = "HIGH"
                risk_color = "🔴"
            st.markdown(f"### {risk_color} Estimated Hypoxia Risk: **{risk_level}**")
            st.caption(
                f"Derived from HHRI = {hhri_scalar_val:.2f} using existing thresholds "
                f"LOW<{low_thr}≤MODERATE<{mod_thr}≤HIGH (PHASE2_CONFIG['hhri']['thresholds'])."
            )
        else:
            st.warning("⚠ HHRI scalar not available; risk level cannot be derived.")

        # ---- F. Hypoxia Risk Map ----
        st.markdown("## F. Estimated Hypoxia Risk Map")
        st.caption(
            "Spatial hypoxia-risk classification (categorical). "
            "🟢 LOW · 🟡 MODERATE · 🔴 HIGH"
        )

        hypoxia_map_url = thumbnails.get('hypoxia_risk')
        if hypoxia_map_url:
            st.image(
                hypoxia_map_url,
                use_container_width=True,
                caption="Estimated Hypoxia Risk Map (LOW / MODERATE / HIGH)"
            )
        else:
            st.info("Hypoxia risk map not available")

        # =================================================================
        # 3. SCIENTIFIC STATUS
        # =================================================================
        st.markdown("---")
        st.markdown("## 🔬 Scientific Status")
        st.markdown(
            """
**Mode:** Prototype / Proxy
- Water hyacinth classification uses Dynamic World labels as automatic proxies only — *not* authoritative species-level ground truth.
- HHRI is a satellite-derived composite risk index combining NDVI, 1−NDWI, chlorophyll-a proxy, turbidity proxy, and temperature placeholder. It does **not** represent measured dissolved oxygen.
- Surface temperature is currently a placeholder because Sentinel-2 has no thermal bands (Landsat integration required for actual temperature).
- HHRI weights and thresholds (LOW < 0.3 ≤ MODERATE < 0.6 ≤ HIGH) are configurable in `PHASE2_CONFIG` but are *not empirically validated* against field measurements.
"""
        )

        # =================================================================
        # 4. MAIN RESULT SUMMARY (top-of-page)
        # =================================================================
        st.markdown("---")
        st.markdown("## 📋 Main Result Summary")

        st.code(
            f"""
WATER HYACINTH DETECTION
-------------------------
Water Area:        {f'{water_area_km2:.2f} km²' if water_area_ha_val is not None else 'N/A'}
Hyacinth Area:     {f'{hyacinth_area_km2:.2f} km²' if hyacinth_area_ha_val is not None else 'N/A'}
Hyacinth Coverage: {f'{coverage_val:.2f} %' if coverage_val is not None else 'N/A'}

HYPOXIA RISK ASSESSMENT
------------------------
HHRI:              {f'{hhri_scalar_val:.2f}' if hhri_scalar_val is not None else 'N/A'}
Estimated Risk:    {risk_level if hhri_scalar_val is not None else 'N/A'}
""",
            language="text",
        )

        st.caption("All values above are computed server-side from the existing Earth Engine pipeline; no values are hard-coded.")

        st.markdown("### Navigation")
        if st.button("← Back to Search Page", key="nav_back_search_summary"):
            switch_page('search')
            st.session_state.selected_water_body = None
            st.session_state.pipeline_results = None
            st.rerun()
        st.info(
            "Scientific validation uses independent human-labeled ground truth with spatial hold-out splits. "
            "Dynamic World proxy evaluation is not scientific validation and should not be reported as accuracy."
        )

        # Navigation button
        if st.button("← Back to Search Page", key="nav_back_search_end"):
            switch_page('search')
            st.session_state.selected_water_body = None
            st.session_state.pipeline_results = None
            st.rerun()


# ---------------------------------------------------------
# MAIN APP ENTRY
# ---------------------------------------------------------

def main():
    """Main Streamlit application entry point."""
    # Page config
    st.set_page_config(
        page_title="Water Hyacinth Detection System",
        page_icon="🌿",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Ensure Earth Engine is initialized
    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=EE_PROJECT_ID)
        except Exception as e:
            st.error(f"❌ Earth Engine initialization failed: {e}")
            st.stop()

    # Route to the appropriate page
    if st.session_state.page == 'search':
        page_search()
    elif st.session_state.page == 'analysis':
        page_analysis()


if __name__ == "__main__":
    main()