with open('archive/hypoxia_phase2_archived.py', 'a', encoding='utf-8') as f:
    f.write('''

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
''')
