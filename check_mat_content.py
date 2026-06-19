# check_mat_contents.py
from scipy.io import loadmat
import os

folder = r'D:\Datasets\abide\sub-control50030'
files = os.listdir(folder)
print("Files:", files)

# Check the timeseries mat file
mat_path = os.path.join(folder, 'sub-control50030_AAL116_features_timeseries.mat')
mat = loadmat(mat_path)
print("\nKeys in mat file:")
for k, v in mat.items():
    if not k.startswith('_'):
        print(f"  {k}: type={type(v)}, ", end='')
        if hasattr(v, 'shape'):
            print(f"shape={v.shape}")
        else:
            print(f"value={v}")