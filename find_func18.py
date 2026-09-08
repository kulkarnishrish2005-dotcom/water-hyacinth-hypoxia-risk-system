import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
start = -1
for i, line in enumerate(lines):
    if "classification_result = classify_random_forest(" in line:
        start = i
        break
for j in range(start+7, start+20):
    print(f'{j}: {lines[j].rstrip()}')
