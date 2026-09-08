"""
Re-run Vembanad and Loktak using EXACTLY 2025-01-01 to 2025-03-01.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

SITES = [
    ("Loktak",   24.55, 93.85, "2025-01-01", "2025-03-01", 10),
    ("Vembanad",  9.60, 76.38, "2025-01-01", "2025-03-01", 10),
]

for name, lat, lon, sd, ed, mc in SITES:
    print(f"\n{'='*60}")
    print(f"  TESTING: {name} ({lat}, {lon})")
    print(f"  Date range: {sd} to {ed}, max_cloud: {mc}%")
    print(f"{'='*60}")
    try:
        res = run_scientific_validation(
            analysis_date=datetime(2025, 2, 1),
            lat=lat, lon=lon,
            start_date=sd, end_date=ed, max_cloud=mc, mode='prototype'
        )
        if res.get('error'):
            print(f"ERROR: {res['error']}")
            continue
        
        state = res.get('detection_state', 'unknown')
        if state == 'classified':
            ha = res.get('hyacinth_area_ha')
            wa = res.get('water_area_ha')
            cov = res.get('hyacinth_coverage_pct')
            if hasattr(ha, 'getInfo'): ha = ha.getInfo()
            if hasattr(wa, 'getInfo'): wa = wa.getInfo()
            if hasattr(cov, 'getInfo'): cov = cov.getInfo()
            print(f"RESULT: STATE 3 | Hyacinth: {ha:.2f} ha | Water: {wa:.2f} ha | Coverage: {cov:.2f}%")
        elif state == 'low_confidence_signal':
            detected = res.get('confident_veg_area_ha', 0)
            threshold = res.get('classify_threshold_ha', 0)
            print(f"RESULT: STATE 2 | Detected: {detected:.2f} ha | Threshold: {threshold:.2f} ha")
        elif state == 'no_significant_vegetation':
            print(f"RESULT: STATE 1")
    except Exception as e:
        print(f"ERROR: {e}")
