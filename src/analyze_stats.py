import os
import json
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, type=str)
args = parser.parse_args()

print(args.path)
data = []
with open(args.path, "rt", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print("Total data length:", len(data))

all_domains = []
for item in data:
    if 'domains' in item:
        all_domains.extend(item['domains'])

domain_counter = Counter(all_domains)
print(f"Total unique domains: {len(domain_counter)}")
print("\nTop 10 domains:")
for domain, count in domain_counter.most_common(10):
    print(f"{domain}: {count} ({count/len(data):.2%})")