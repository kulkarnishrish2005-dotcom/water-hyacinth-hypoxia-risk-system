import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
for i, line in enumerate(lines):
    if 'def create_' in line:
        print(f'{i}: {line.strip()}')
