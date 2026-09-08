import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').readlines()
start = -1
for i, line in enumerate(lines):
    if "else:" in line and "water_str = f" in lines[i+9]:
        start = i
        break
for j in range(start, start+30):
    print(f'{j}: {lines[j].rstrip()}')
