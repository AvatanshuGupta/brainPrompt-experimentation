# download_abide_phenotypic.py
import urllib.request

url = "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/Phenotypic_V1_0b_preprocessed1.csv"
out_path = "data/prompts/meta_prompts/ABIDE_phenotypic.csv"

print("Downloading ABIDE phenotypic data...")
urllib.request.urlretrieve(url, out_path)
print("Done.")

import csv
with open(out_path) as f:
    rows = list(csv.reader(f))
print("Header:", rows[0])
print("Row 1:", rows[1])
print("Total rows:", len(rows))