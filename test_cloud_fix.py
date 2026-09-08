import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

# We test Vembanad 2024-01-01 to 2024-03-31 with max_cloud=30.
# Previously this yielded 0.44 ha because the cloudiest image was on top.
print("\n=== Testing Vembanad with max_cloud=30 (Cloud fix verification) ===")
res = run_scientific_validation(
    analysis_date=datetime(2024, 2, 1),
    lat=9.60, lon=76.38,
    start_date='2024-01-01', end_date='2024-03-31', max_cloud=30, mode='prototype'
)

cls = res.get('classification', {})
ha = res.get('hyacinth_area_ha')
wa = res.get('water_area_ha')
cov = res.get('hyacinth_coverage_pct')
detected = res.get('confident_veg_area_ha')
state = res.get('detection_state')

if hasattr(ha, 'getInfo'): ha = ha.getInfo()
if hasattr(wa, 'getInfo'): wa = wa.getInfo()
if hasattr(cov, 'getInfo'): cov = cov.getInfo()
if hasattr(detected, 'getInfo'): detected = detected.getInfo()

print(f"State: {state}")
print(f"Detected Veg (ha): {detected}")
print(f"Hyacinth Area (ha): {ha}")
print(f"Total Water (ha): {wa}")
