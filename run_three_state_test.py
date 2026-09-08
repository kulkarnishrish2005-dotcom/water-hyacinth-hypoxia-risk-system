"""
Seven-site validation of the three-state vegetation detection gate.
Tests: Varanasi, Loktak, Vembanad, Dal Lake, Yamuna, Bhakra Nangal.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

SITES = [
    # (name, lat, lon, start_date, end_date, max_cloud)
    ("Varanasi",          25.30, 83.01, "2024-04-01", "2024-06-30", 10),
    ("Loktak",            24.55, 93.85, "2024-04-01", "2024-06-30", 10),
    ("Vembanad",           9.60, 76.38, "2024-01-01", "2024-03-31", 30),
    ("Dal Lake",          34.12, 74.86, "2024-04-01", "2024-06-30", 10),
    ("Yamuna at Delhi",   28.61, 77.25, "2024-04-01", "2024-06-30", 10),
    ("Bhakra Nangal Dam", 31.42, 76.43, "2024-04-01", "2024-06-30", 10),
]

results_summary = []

for name, lat, lon, sd, ed, mc in SITES:
    print(f"\n{'='*60}")
    print(f"  TESTING: {name} ({lat}, {lon})")
    print(f"  Date range: {sd} to {ed}, max_cloud: {mc}%")
    print(f"{'='*60}")
    try:
        res = run_scientific_validation(
            analysis_date=datetime(2024, 6, 15),
            lat=lat, lon=lon,
            start_date=sd, end_date=ed, max_cloud=mc, mode='prototype'
        )
        
        if res.get('error'):
            results_summary.append(f"{name}: ERROR - {res['error']}")
            continue
            
        state = res.get('detection_state', 'unknown')
        
        if state == 'classified':
            ha = res.get('hyacinth_area_ha')
            wa = res.get('water_area_ha')
            cov = res.get('hyacinth_coverage_pct')
            if ha is not None:
                try: ha = ha.getInfo() if hasattr(ha, 'getInfo') else ha
                except: pass
            if wa is not None:
                try: wa = wa.getInfo() if hasattr(wa, 'getInfo') else wa
                except: pass
            if cov is not None:
                try: cov = cov.getInfo() if hasattr(cov, 'getInfo') else cov
                except: pass
            results_summary.append(
                f"{name}: STATE 3 (classified) | "
                f"Hyacinth: {ha:.2f} ha | Water: {wa:.2f} ha | Coverage: {cov:.2f}%"
            )
        elif state == 'low_confidence_signal':
            detected = res.get('confident_veg_area_ha', 0)
            threshold = res.get('classify_threshold_ha', 0)
            results_summary.append(
                f"{name}: STATE 2 (low_confidence_signal) | "
                f"Detected: {detected:.2f} ha | Threshold: {threshold:.2f} ha"
            )
        elif state == 'no_significant_vegetation':
            results_summary.append(f"{name}: STATE 1 (no_significant_vegetation)")
        else:
            results_summary.append(f"{name}: STATE: {state}")
    except Exception as e:
        results_summary.append(f"{name}: ERROR - {e}")

print("\n" + "="*60)
print("  FINAL SUMMARY - ALL 6 SITES")
print("="*60)
for line in results_summary:
    print(f"  {line}")
print("="*60)
