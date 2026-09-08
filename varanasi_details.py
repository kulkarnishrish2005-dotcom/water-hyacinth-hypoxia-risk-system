import sys
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

res = run_scientific_validation(
    analysis_date=datetime(2025, 2, 1),
    lat=25.30, lon=83.01,
    start_date='2025-01-01', end_date='2025-03-01', max_cloud=10, mode='prototype'
)
print('Varanasi 2025 details:')
print('Total Water:', res.get('water_area_ha').getInfo() if res.get('water_area_ha') else 'N/A')
print('Hyacinth Area:', res.get('hyacinth_area_ha').getInfo() if res.get('hyacinth_area_ha') else 'N/A')
print('Coverage:', res.get('hyacinth_coverage_pct').getInfo() if res.get('hyacinth_coverage_pct') else 'N/A')
