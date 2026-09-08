import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

sites = [
    ('Varanasi', 25.30, 83.01),
    ('Loktak', 24.55, 93.85),
    ('Vembanad', 9.60, 76.38),
    ('Bhakra Nangal', 31.42, 76.43)
]

for name, lat, lon in sites:
    res = run_scientific_validation(
        analysis_date=datetime(2025, 2, 1),
        lat=lat, lon=lon,
        start_date='2025-01-01', end_date='2025-03-01', max_cloud=10, mode='prototype'
    )
    
    ha = res.get('hyacinth_area_ha')
    wa = res.get('water_area_ha')
    if hasattr(ha, 'getInfo'): ha = ha.getInfo()
    if hasattr(wa, 'getInfo'): wa = wa.getInfo()
    
    print(f'{name} -> Hyacinth: {ha} ha, Water: {wa} ha')
