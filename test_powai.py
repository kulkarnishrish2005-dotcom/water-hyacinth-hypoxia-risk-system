import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

# Powai Lake, Mumbai: 19.127, 72.905
res = run_scientific_validation(
    analysis_date=datetime(2025, 2, 1),
    lat=19.127, lon=72.905,
    start_date='2025-01-01', end_date='2025-03-01', max_cloud=10, mode='prototype'
)
print("State:", res.get('detection_state'))
conf_veg = res.get('confident_veg_area_ha')
print("Conf Veg Area:", conf_veg.getInfo() if hasattr(conf_veg, 'getInfo') else conf_veg)
cohort = res.get('p95_ndvi')
print("Cohort Median:", cohort.getInfo() if hasattr(cohort, 'getInfo') else cohort)
print("Suspect:", res.get('suspect_signal'))
if res.get('detection_state') == 'classified':
    cov = res.get('hyacinth_coverage_pct')
    print("Coverage:", cov.getInfo() if hasattr(cov, 'getInfo') else cov)
