"""
Interactive map of India showing major water bodies for site selection.

Renders a Folium map with curated markers (lakes, rivers, dams) loaded from
data/india_water_bodies.geojson.  The user clicks a marker, reviews the site
details displayed below the map, and presses "Run analysis here" to populate
``st.session_state.selected_water_body`` — the same dict shape that
``page_analysis()`` already consumes.

Future enhancement
------------------
Swap the static GeoJSON with a live query to GEE
(``ee_utils.discover_water_bodies``) or an OpenStreetMap Overpass API call.
"""

import json
import os
import random

import folium
import streamlit as st
from streamlit_folium import st_folium


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_GEOJSON_PATH = os.path.join(_DATA_DIR, "india_water_bodies.geojson")

# Marker colour per water-body type
_TYPE_COLORS = {
    "lake": "blue",
    "river": "green",
    "dam": "orange",
}

# Emoji labels for the legend / filter checkboxes
_TYPE_LABELS = {
    "lake": "🔵 Lakes",
    "river": "🟢 Rivers",
    "dam": "🟠 Dams / Reservoirs",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def _load_water_bodies():
    """Load the curated India water-bodies GeoJSON.

    Cached across Streamlit reruns for performance.

    To use a live source instead, replace this function body with a call to
    ``ee_utils.discover_water_bodies()`` or an Overpass API query and return
    a dict that follows the same GeoJSON FeatureCollection schema.
    """
    with open(_GEOJSON_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)





# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_site_by_name(name, geojson_data):
    """Look up a water-body feature by its ``name`` property."""
    for feature in geojson_data.get("features", []):
        if feature["properties"].get("name") == name:
            return feature
    return None


def _find_site_by_coords(lat, lng, geojson_data, tolerance=0.02):
    """Fall-back: find the nearest water body within *tolerance* degrees."""
    best, best_dist = None, float("inf")
    for feature in geojson_data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        d = abs(coords[1] - lat) + abs(coords[0] - lng)
        if d < best_dist and d < tolerance:
            best, best_dist = feature, d
    return best


# ---------------------------------------------------------------------------
# Public API — call from app.py
# ---------------------------------------------------------------------------

def render_india_map():
    """Render the interactive India water-bodies map with site selection.

    Call this inside a Streamlit container (tab, column, page).  On marker
    selection + button click it sets ``st.session_state.selected_water_body``
    and switches to the analysis page — the **same** transition path used by
    the existing "Select" button in the city-search flow.
    """

    # --- Type-filter checkboxes in Sidebar --------------------------------------
    with st.sidebar:
        st.markdown("### 🗺️ Map Filters")
        show_lakes = st.checkbox("🔵 Lakes", value=True, key="map_show_lakes")
        show_rivers = st.checkbox("🟢 Rivers", value=True, key="map_show_rivers")
        show_dams = st.checkbox("🟠 Dams / Reservoirs", value=True, key="map_show_dams")
        st.markdown("---")

    visible_types = set()
    if show_lakes:
        visible_types.add("lake")
    if show_rivers:
        visible_types.add("river")
    if show_dams:
        visible_types.add("dam")

    # --- Load data -------------------------------------------------------------
    geojson_data = _load_water_bodies()

    # --- Build Folium map ------------------------------------------------------
    m = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles=None)

    # Base tile layers — street (default) + satellite toggle
    folium.TileLayer(
        tiles="openstreetmap",
        name="Street Map",
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(m)

    # --- One FeatureGroup per type (enables layer-control toggle) --------------
    groups = {
        "lake": folium.FeatureGroup(name=_TYPE_LABELS["lake"]),
        "river": folium.FeatureGroup(name=_TYPE_LABELS["river"]),
        "dam": folium.FeatureGroup(name=_TYPE_LABELS["dam"]),
    }

    for feature in geojson_data.get("features", []):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        site_lon, site_lat = coords[0], coords[1]
        wb_type = props.get("type", "lake")

        if wb_type not in visible_types:
            continue

        name = props.get("name", "Unknown")
        state = props.get("state", "Unknown")

        color = _TYPE_COLORS.get(wb_type, "gray")

        popup_html = (
            f'<div style="font-family:sans-serif;min-width:180px;">'
            f'<b style="font-size:14px;">{name}</b><br>'
            f'<span style="color:#666;">Type:</span> {wb_type.title()}<br>'
            f'<span style="color:#666;">State:</span> {state}<br>'
            f'<span style="color:#666;">Coords:</span> '
            f'{site_lat:.4f}°N, {site_lon:.4f}°E<br>'
            f'<hr style="margin:4px 0;">'
            f"<i>Click marker, then press<br>"
            f'"Run analysis here" below the map.</i>'
            f"</div>"
        )

        marker = folium.Marker(
            location=[site_lat, site_lon],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=name,
            icon=folium.Icon(color=color, icon="info-sign"),
        )

        group = groups.get(wb_type)
        if group is not None:
            marker.add_to(group)

    for group in groups.values():
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # --- Render map & capture interaction --------------------------------------
    map_data = st_folium(
        m, height=500, use_container_width=True, key="india_water_bodies_map"
    )

    # --- Handle marker click ---------------------------------------------------
    selected_feature = None

    if map_data:
        # Primary: match by tooltip (the marker's name text)
        tooltip_text = map_data.get("last_object_clicked_tooltip")
        if tooltip_text:
            selected_feature = _find_site_by_name(tooltip_text, geojson_data)

        # Fallback: match by coordinates (for older streamlit-folium versions)
        if selected_feature is None:
            clicked = map_data.get("last_object_clicked")
            if clicked and isinstance(clicked, dict):
                selected_feature = _find_site_by_coords(
                    clicked.get("lat", 0), clicked.get("lng", 0), geojson_data
                )

    # --- Show selection details + action button --------------------------------
    if selected_feature is not None:
        props = selected_feature["properties"]
        coords = selected_feature["geometry"]["coordinates"]
        site_lat, site_lon = coords[1], coords[0]
        site_name = props.get("name", "Unknown")
        site_type = props.get("type", "unknown").title()
        site_state = props.get("state", "India")

        st.success(f"📍 **{site_name}** — {site_type}, {site_state}")

        info_col, btn_col = st.columns([3, 1])
        with info_col:
            st.markdown(
                f"**Coordinates:** {site_lat:.4f}°N, {site_lon:.4f}°E"
            )
        with btn_col:
            if st.button(
                "🎯 Run analysis here",
                key="map_select_site",
                use_container_width=True,
            ):
                # Build the SAME dict shape page_analysis() expects
                st.session_state.selected_water_body = {
                    "centroid_lat": site_lat,
                    "centroid_lon": site_lon,
                    "name": site_name,
                    "estimated_size_ha": None,
                }
                st.session_state.page = "analysis"
                st.rerun()
    else:
        st.info("👆 Click a marker on the map to select a water body for analysis.")
