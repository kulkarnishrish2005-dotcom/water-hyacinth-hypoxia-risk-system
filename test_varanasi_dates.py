import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

print('=== Varanasi 2024 (Original Test Dates) ===')
run_scientific_validation(
    analysis_date=datetime(2024, 6, 1),
    lat=25.30, lon=83.01,
    start_date='2024-04-01', end_date='2024-06-30', max_cloud=10, mode='prototype'
)

print('\n=== Varanasi 2025 (Canonical Dates) ===')
run_scientific_validation(
    analysis_date=datetime(2025, 2, 1),
    lat=25.30, lon=83.01,
    start_date='2025-01-01', end_date='2025-03-01', max_cloud=10, mode='prototype'
)
