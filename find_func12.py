import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
start = -1
for i, line in enumerate(lines):
    if 'print("=== Pipeline Complete ===")' in line:
        start = i - 10
        break
for j in range(start, start+40):
    print(f'{j}: {lines[j].rstrip()}')
