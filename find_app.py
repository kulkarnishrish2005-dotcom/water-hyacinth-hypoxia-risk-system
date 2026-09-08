import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').readlines()
start = -1
for i, line in enumerate(lines):
    if "detection_state == 'no_significant_vegetation'" in line:
        start = i - 5
        break
for j in range(start, start+40):
    print(f'{j}: {lines[j].rstrip()}')
