import ee
import sys
from ee_utils import initialize_ee, run_full_pipeline

# Initialize Earth Engine
initialize_ee()

lat = 26.9000
lon = 75.0800

print('Running pipeline for Sambhar Salt Lake...')
result = run_full_pipeline(
    lat=lat, 
    lon=lon, 
    start_date='2025-01-01', 
    end_date='2025-03-01', 
    max_cloud=5
)

if 'error' in result:
    print(f"Pipeline error: {result['error']}")
    sys.exit(1)

coverage = result.get('hyacinth_coverage_pct')

if coverage is not None:
    val = coverage.getInfo()
    print(f"\nSUCCESS! hyacinth_coverage_pct = {val:.2f}%")
else:
    print("\nSUCCESS, but hyacinth_coverage_pct was None.")
