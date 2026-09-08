import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
for j in range(495, 515):
    print(f'{j}: {lines[j].rstrip()}')
