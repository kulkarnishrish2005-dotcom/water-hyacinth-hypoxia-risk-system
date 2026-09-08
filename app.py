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
# CSS STYLING (STAGE 1-5)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        /* Color Palette */
        :root {
            --bg-deep: #0a1929;
            --bg-card: #132f4c;
            --accent-cyan: #14b8a6;
            --accent-hover: #0d9488;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border-color: #1e40af;
            --risk-high: #ef4444;
            --risk-mod: #f59e0b;
            --risk-low: #10b981;
        }

        /* Base Typography & Background - USING SYSTEM FONTS FOR CSP SAFETY */
        .stApp {
            background-color: var(--bg-deep) !important;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: var(--text-main) !important;
        }

        /* Monospace / Tabular Numbers for Data */
        .metric-value, .stMetricValue, .tabular-data {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
            font-weight: 700 !important;
            font-variant-numeric: tabular-nums;
        }

        /* Global Card System (Forms, Charts) */
        div[data-testid="stForm"], 
        .stDataFrame,
        div.stPlotlyChart {
            background-color: var(--bg-card) !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Map Container Wrapper (Fixing the iframe styling) */
        /* Streamlit wraps folium in a container. We style the iframe but ensure overflow is hidden */
        iframe[title="streamlit_folium.st_folium"] {
            border-radius: 12px !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
            border: 1px solid var(--border-color) !important;
            background-color: var(--bg-card); /* Prevents white flash */
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0f2238 !important;
            border-right: 1px solid var(--border-color);
        }

        /* Buttons */
        button[data-testid="baseButton-secondary"], 
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-secondaryFormSubmit"] {
            background-color: var(--accent-cyan) !important;
            color: #0a1929 !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[data-testid="baseButton-secondary"]:hover, 
        button[data-testid="baseButton-primary"]:hover,
        button[data-testid="baseButton-secondaryFormSubmit"]:hover {
            background-color: var(--accent-hover) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(20, 184, 166, 0.4) !important;
            color: #ffffff !important;
        }

        /* Inputs */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: var(--bg-deep) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Hero / Header Section */
        .hero-container {
            padding: 1.5rem 0 2rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid;
            border-image: linear-gradient(to right, var(--accent-cyan), transparent) 1;
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-muted) !important;
            margin-top: 0;
            font-weight: 400;
        }

        /* Custom Stat Cards (Stage 4) */
        .stat-card {
            background-color: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-cyan);
            box-shadow: 0 10px 15px -3px rgba(20, 184, 166, 0.1);
        }
        .stat-label {
            color: var(--text-muted);
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .stat-value {
            font-size: 2rem;
            color: var(--text-main);
            margin: 0;
            line-height: 1.2;
        }
        .stat-unit {
            font-size: 1rem;
            color: var(--text-muted);
            font-weight: 400;
        }

        /* Image Cards & Legends (Stage 5) */
        .image-card {
            background-color: var(--bg-card);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }
        .image-card img {
            border-radius: 8px;
            width: 100%;
        }
        .image-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .custom-legend {
            display: flex;
            gap: 1.5rem;
            padding: 1rem;
            background: var(--bg-deep);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 1: SEARCH
# ---------------------------------------------------------

def _render_city_search():
    """City search UI — renders inside the sidebar."""
    st.markdown("### Or search by name")
    
    with st.form(key='search_form', clear_on_submit=False):
        city_name = st.text_input(
            "Enter a city or place name:",
            placeholder="e.g., Pune, Mumbai, Loktak Lake",
            help="Will search for water bodies within 15 km radius using JRC GSW dataset"
        )
        search_submitted = st.form_submit_button("🔍 Search")

    if search_submitted:
        if not city_name.strip():
            st.error("Please enter a city name.")
            return

        with st.spinner(f"Geocoding '{city_name}'..."):
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
                    st.success(f"✓ Found {result['polygon_count']} water body(ies) near {city_name}.")
                else:
                    st.warning("⚠️ No water bodies detected. Adjust date or try another city.")

            except ValueError as e:
                st.error(f"Geocoding error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display discovered water bodies in sidebar
    if st.session_state.get('water_bodies'):
        st.markdown("### 📍 Detected Sites")
        for i, wb in enumerate(st.session_state['water_bodies']):
            with st.container():
                st.markdown(f"**{wb.get('name', f'Water Body {i+1}')}**")
                est_size = wb.get('estimated_size_ha', 'N/A')
                if isinstance(est_size, (int, float)):
                    st.caption(f"{est_size:.1f} ha")
                if st.button(f"Select", key=f"select_{i}", use_container_width=True):
                    st.session_state.selected_water_body = wb
                    st.session_state.page = 'analysis'
                    st.rerun()


# ---------------------------------------------------------
# PAGE 1 (wrapper): MAIN MAP VIEW
# ---------------------------------------------------------

def page_search():
    """Page 1: Default view - Interactive Map for site selection."""
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">
                <span style="color: var(--accent-cyan);">💧</span> AquaWatch System
            </div>
            <div class="hero-subtitle">Satellite-driven Water Hyacinth Detection</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for secondary controls
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        _render_city_search()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("Click a marker on the map to select a water body for analysis, or search by city above.")

    # Main area map
    from map_view import render_india_map
    render_india_map()


# ---------------------------------------------------------
# PAGE 2: ANALYSIS
# ---------------------------------------------------------

def page_analysis():
    """Page 2: Run full pipeline on selected water body."""
    st.markdown("""
        <div class="hero-container" style="margin-bottom: 1rem; padding-bottom: 1rem;">
            <div class="hero-title" style="font-size: 2rem;">
                <span style="color: var(--accent-cyan);">📊</span> Analysis Pipeline
            </div>
            <div class="hero-subtitle">Satellite-driven Water Hyacinth Detection</div>
        </div>
    """, unsafe_allow_html=True)

    # Check we have a selected water body
    if not st.session_state.get('selected_water_body'):
        st.info("👈 Please select a water body from the Search page first.")
        if st.button("← Go Back to Search"):
            switch_page('search')
            st.rerun()
        return

    wb = st.session_state.selected_water_body
    lat, lon = wb['centroid_lat'], wb['centroid_lon']

    wb_name = wb.get('name')
    if wb_name:
        st.markdown(f"**Selected Water Body:** {wb_name} — Centroid at {lat}, {lon}")
    else:
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
                    
                if results.get('classification', {}).get('error'):
                    st.error(f"❌ Classification Error: {results['classification']['error']}")
                    # We don't return here, we let the UI render what it can (like the thumbnails)

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

        # ---- A. Summary Metrics (STAGE 4 Redesign) ----
        st.markdown("## A. Summary Metrics")

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

        # Values
        water_area_ha_val = _get_scalar(results.get('water_area_ha'))
        hyacinth_area_ha_val = _get_scalar(results.get('hyacinth_area_ha'))
        coverage_val = _get_scalar(results.get('hyacinth_coverage_pct'))
        other_veg_ha_val = _get_scalar(results.get('other_vegetation_area_ha'))

        detection_state = results.get('detection_state', 'classified')

        if detection_state == 'no_significant_vegetation':
            st.success("✅ **No significant vegetation detected.** Water body appears clean for this date range.")
            if water_area_ha_val is not None:
                st.caption(f"Total water area: {water_area_ha_val / 100.0:.2f} km²")
        elif detection_state == 'low_confidence_signal':
            detected_ha = results.get('confident_veg_area_ha', 0)
            threshold_ha = results.get('classify_threshold_ha', 0)
            st.warning(
                f"⚠️ **Low-confidence vegetation signal detected:** {detected_ha:.1f} ha of spectral "
                f"signal found, below the {threshold_ha:.1f} ha threshold needed for reliable "
                f"classification at this water body's scale. This may indicate early-stage, patchy, "
                f"or seasonal vegetation — or residual shoreline noise. Coverage percentage is NOT "
                f"computed for this state."
            )
            if water_area_ha_val is not None:
                st.caption(f"Total water area: {water_area_ha_val / 100.0:.2f} km²")
        else:
            detected_ha = results.get('confident_veg_area_ha')
            threshold_ha = results.get('classify_threshold_ha')
            
            if detected_ha is not None and threshold_ha is not None:
                st.success(f"✅ **Area Gate Passed:** {detected_ha:.2f} ha detected, {threshold_ha:.2f} ha required.")
            
            if results.get('suspect_signal'):
                p95_val = results.get('p95_ndvi', 'unknown')
                if isinstance(p95_val, float):
                    p95_val = f"{p95_val:.3f}"
                st.warning(
                    f"⚠️ **Spectral signal is weak (Cohort Median NDVI = {p95_val})** — this result may reflect "
                    "ambiguous vegetation/mud/algae rather than confirmed water hyacinth. Recommend "
                    "manual review of the NDVI thumbnail before treating this coverage % as reliable."
                )

            water_str = f"{water_area_ha_val / 100.0:.2f}" if water_area_ha_val is not None else "N/A"
            hya_str = f"{hyacinth_area_ha_val / 100.0:.2f}" if hyacinth_area_ha_val is not None else "N/A"
            other_str = f"{other_veg_ha_val / 100.0:.2f}" if other_veg_ha_val is not None else "N/A"            is_suspect = results.get('suspect_signal')

            # Color coding for coverage
            cov_str = "N/A"
            cov_color = "var(--text-main)"
            unit_str = '<span class="stat-unit">%</span>'
            if coverage_val is not None:
                if is_suspect:
                    cov_str = "Unverified"
                    cov_color = "var(--text-muted, gray)"
                    unit_str = '<div style="font-size: 0.9rem; font-weight: normal; margin-top: 5px;">spectral signal below confidence threshold</div>'
                else:
                    cov_str = f"{coverage_val:.2f}"
                    if coverage_val > 15:
                        cov_color = "var(--risk-high)"
                    elif coverage_val > 5:
                        cov_color = "var(--risk-mod)"
                    else:
                        cov_color = "var(--risk-low)"
    
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">🌊 Total Water Area</div>
                        <div class="stat-value tabular-data">{water_str} <span class="stat-unit">km²</span></div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_b:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">🌿 Hyacinth Area</div>
                        <div class="stat-value tabular-data">{hya_str} <span class="stat-unit">km²</span></div>
                    </div>
                """, unsafe_allow_html=True)            with col_c:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">⚠️ Hyacinth Coverage</div>
                        <div class="stat-value tabular-data" style="color: {cov_color};">{cov_str} {unit_str}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_d:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">🌾 Other Vegetation</div>
                        <div class="stat-value tabular-data">{other_str} <span class="stat-unit">km²</span></div>
                    </div>
                """, unsafe_allow_html=True)
    
            st.markdown("<br>", unsafe_allow_html=True)
            
            cls_result = results.get('classification', {})
            if cls_result.get('fallback_zero_vegetation'):
                st.warning("⚠️ **Low Confidence Zero:** No confident vegetation training samples were found for this site/date range. The 0% coverage result may reflect threshold sensitivity rather than a confirmed absence of vegetation.")
            elif cls_result.get('low_confidence_training'):
                st.warning(f"⚠️ **Statistically Weak Training Data:** The classifier successfully trained, but on a very small sample pool ({cls_result.get('veg_sample_count')} vegetation / {cls_result.get('water_sample_count')} water samples). The reported accuracy and coverage percentages may be inflated by small-sample luck and should be interpreted with caution. Consider widening the date range to pull more cloud-free imagery.")
    
            st.caption(
                "ℹ️ **Disclaimer on Risk Thresholds:** Coverage color thresholds are illustrative, not calibrated. "
                "The underlying mechanism — that dense hyacinth mats sharply reduce dissolved oxygen by blocking light and atmospheric "
                "gas exchange — is supported by field studies (e.g. Mironga et al., 2012, *Int. J. Humanities & Social Sciences*, "
                "comparing hyacinth-covered vs open-water DO in Lake Naivasha; Villamagna & Murphy, 2010, *Freshwater Biology*, "
                "review of hyacinth ecological impacts). However, neither source establishes a specific coverage-percentage threshold "
                "for ecological impact onset — the 5%/15% cutoffs used here are provisional UI defaults."
            )
            st.caption(
                "**Calculation method (server-side EE):** "
                "Hyacinth area = Σ pixelArea where classification == 1; "
                "Water area = Σ pixelArea within water_mask; "
                "Coverage = (Hyacinth area / Water area) × 100. "
                "Other Vegetation = Σ pixelArea where classification == 2."
            )

        # ---- B. Water Hyacinth Classification Map (STAGE 5 Redesign) ----
        st.markdown("## B. Water Hyacinth Classification")
        
        st.markdown("""
            <div class="custom-legend">
                <div class="legend-item"><div class="legend-color" style="background: blue;"></div> Water (Class 0)</div>
                <div class="legend-item"><div class="legend-color" style="background: darkgreen;"></div> Water Hyacinth (Class 1)</div>
                <div class="legend-item"><div class="legend-color" style="background: pink;"></div> Other Vegetation (Class 2)</div>
            </div>
        """, unsafe_allow_html=True)

        if thumbnails.get('classified'):
            st.markdown(f"""
                <div class="image-card">
                    <div class="image-title">🗺️ Random Forest Classification Map</div>
                    <img src="{thumbnails['classified']}" alt="Classification Map" />
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Classification map not available")

        # ---- C. Supporting Remote-Sensing Outputs (STAGE 5 Redesign) ----
        st.markdown("## C. Supporting Remote-Sensing Outputs")

        col1, col2 = st.columns(2)
        with col1:
            if thumbnails.get('true_color'):
                st.markdown(f"""
                    <div class="image-card">
                        <div class="image-title">📷 Sentinel-2 True Color</div>
                        <img src="{thumbnails['true_color']}" alt="True Color" />
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("True color not available")

        with col2:
            if results.get('water_mask') is not None:
                water_mask_thumb = results['water_mask'].getThumbURL({
                    'bands': ['water_mask'],
                    'min': 0, 'max': 1,
                    'region': results['aoi'], 'dimensions': 512
                })
                st.markdown(f"""
                    <div class="image-card">
                        <div class="image-title">💧 Water / Land Mask</div>
                        <img src="{water_mask_thumb}" alt="Water Mask" />
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Water mask not available")

        col3, col4 = st.columns(2)
        with col3:
            if thumbnails.get('ndvi'):
                st.markdown(f"""
                    <div class="image-card">
                        <div class="image-title">🌱 NDVI (Vegetation Index)</div>
                        <img src="{thumbnails['ndvi']}" alt="NDVI Index" />
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("NDVI not available")

        with col4:
            if thumbnails.get('ndwi'):
                st.markdown(f"""
                    <div class="image-card">
                        <div class="image-title">🌊 NDWI (Water Index)</div>
                        <img src="{thumbnails['ndwi']}" alt="NDWI Index" />
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("NDWI not available")

        # =================================================================

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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    inject_custom_css()

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