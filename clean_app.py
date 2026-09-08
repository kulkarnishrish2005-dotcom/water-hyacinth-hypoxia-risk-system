import sys
import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    # Skip Phase 2 UI elements (using line index 736 to 861 since it's L737-862)
    if 736 <= i <= 861:
        continue
        
    # L350 hero subtitle
    if '<div class="hero-subtitle">' in line:
        new_lines.append('            <div class="hero-subtitle">Satellite-driven Water Hyacinth Detection</div>\n')
        continue
    
    # L651 disclaimer
    if 'pending Phase 2 calibration' in line:
        new_lines.append('            "for ecological impact onset — the 5%/15% cutoffs used here are provisional UI defaults."\n')
        continue

    # replace title
    line = line.replace('Water Hyacinth & Hypoxia Risk', 'Inland Hyacinth Sentinel')
    line = line.replace('Water Hyacinth Hypoxia Risk System', 'Inland Hyacinth Sentinel')
    new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
