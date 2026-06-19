# check_phenotypic.py
import csv
with open('data/prompts/meta_prompts/ABIDE_phenotypic.csv') as f:
    rows = list(csv.reader(f))
print("Header:", rows[0])
print("Row 1:", rows[1])
print("Total rows:", len(rows))

# Find which column has the subject ID
# We need to match e.g. "50030" from "sub-control50030"
header = rows[0]
for i, col in enumerate(header):
    print(f"  col {i}: {col}")