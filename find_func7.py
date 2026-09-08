import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
start = -1
for i, line in enumerate(lines):
    if 'def _run_prototype_mode' in line:
        start = i
        break
for j in range(start, start+60):
    print(f'{j}: {lines[j].rstrip()}')
