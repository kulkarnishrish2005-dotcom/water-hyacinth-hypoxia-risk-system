import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').readlines()
for j in range(604, 630):
    print(f'{j}: {lines[j].rstrip()}')
