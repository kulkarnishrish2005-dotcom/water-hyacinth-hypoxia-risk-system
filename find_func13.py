import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
start = -1
for i, line in reversed(list(enumerate(lines))):
    if 'def run_full_pipeline' in line:
        start = i
        break
for j in range(start, len(lines)):
    if 'return {' in lines[j]:
        for k in range(j, j+30):
            print(f'{k}: {lines[k].rstrip()}')
        break
