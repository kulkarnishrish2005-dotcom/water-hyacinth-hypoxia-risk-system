import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from ee_utils import initialize_ee, run_scientific_validation
from datetime import datetime

initialize_ee()

SITES = [
    ('Varanasi',          25.30, 83.01),
    ('Mississippi',       35.15, -90.05),
    ('Loktak',            24.55, 93.85),
    ('Vembanad',           9.60, 76.38),
    ('Dal Lake',          34.12, 74.86),
    ('Yamuna',            28.61, 77.25),
    ('Bhakra Nangal',     31.42, 76.43),
    ('Powai Lake',        19.127, 72.905),
]

sd = '2025-01-01'
ed = '2025-03-01'
mc = 10

print(f'\n{"="*105}')
print(f'CANONICAL VALIDATION PASS (max_cloud={mc}%, dates={sd} to {ed})')
print(f'{"="*105}')
print(f'{"Site":<15} | {"State":<25} | {"Conf Veg":<10} | {"Cohort":<6} | {"Suspect":<8} | {"Coverage %"}')
print('-'*105)

for name, lat, lon in SITES:
    try:
        res = run_scientific_validation(
            analysis_date=datetime(2025, 2, 1),
            lat=lat, lon=lon,
            start_date=sd, end_date=ed, max_cloud=mc, mode='prototype'
        )
        if res.get('error'):
            print(f'{name:<15} | {"ERROR":<25} | {"N/A":<10} | {"N/A":<6} | {"N/A":<8} | {res["error"]}')
            continue
            
        state = res.get('detection_state', 'unknown')
        detected = res.get('confident_veg_area_ha')
        if hasattr(detected, 'getInfo'):
            detected = detected.getInfo()
            
        if detected is None:
            detected_str = 'N/A'
        else:
            detected_str = f'{detected:.2f}'
            
        p95 = res.get('p95_ndvi')
        if hasattr(p95, 'getInfo'):
            p95 = p95.getInfo()
            
        p95_str = f'{p95:.3f}' if p95 is not None else 'N/A'
        
        suspect = res.get('suspect_signal', False)
        suspect_str = 'True' if suspect else 'False'
        if p95 is None: suspect_str = 'N/A'
            
        cov_str = 'N/A'
        if state == 'classified':
            cov = res.get('hyacinth_coverage_pct')
            if hasattr(cov, 'getInfo'):
                cov = cov.getInfo()
            if cov is not None:
                cov_str = f'{cov:.2f}%'
            
        print(f'{name:<15} | {state:<25} | {detected_str:<10} | {p95_str:<6} | {suspect_str:<8} | {cov_str}')
    except Exception as e:
        print(f'{name:<15} | {"ERROR":<25} | {"N/A":<10} | {"N/A":<6} | {"N/A":<8} | {e}')

print(f'{"="*105}\n')
