import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
for j in range(1450, 1490):
    print(f'{j}: {lines[j].rstrip()}')
