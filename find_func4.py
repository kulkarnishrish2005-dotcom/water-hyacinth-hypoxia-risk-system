import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('ee_utils.py', encoding='utf-8').readlines()
for i, line in enumerate(lines):
    if 'create_spectral_rule_proxy_training_samples' in line:
        print(f'{i}: {line.strip()}')
