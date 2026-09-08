import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime
initialize_ee()

def check_site(name, date, lat, lon, start, end):
    print(f'\n=== Testing {name} ===')
    try:
        res = run_scientific_validation(
            analysis_date=datetime(*date),
            lat=lat, lon=lon,
            start_date=start, end_date=end, mode='prototype'
        )
        cls = res.get('classification', {})
        print(f'{name} Hyacinth Area (ha):', res.get('hyacinth_area_ha').getInfo() if res.get('hyacinth_area_ha') else 'N/A')
        print(f'{name} Total Water Area (ha):', res.get('water_area_ha').getInfo() if res.get('water_area_ha') else 'N/A')
        print(f'{name} Coverage %:', res.get('hyacinth_coverage_pct').getInfo() if res.get('hyacinth_coverage_pct') else 'N/A')
        print(f'{name} Train Samples - Water: {cls.get("water_sample_count")}, Veg: {cls.get("veg_sample_count")}')
        print(f'{name} Low Confidence Flag: {cls.get("low_confidence_training")}')
        
        if cls.get('fallback_zero_vegetation'):
            print(f'{name} Fallback Flag: True')
    except Exception as e:
        print(f'Error on {name}: {e}')

check_site('Loktak', (2024, 10, 15), 24.55, 93.85, '2024-10-01', '2024-12-31')
check_site('Vembanad', (2024, 1, 15), 9.60, 76.38, '2024-01-01', '2024-03-31')
check_site('Varanasi', (2024, 5, 15), 25.30, 83.01, '2024-04-01', '2024-06-30')
