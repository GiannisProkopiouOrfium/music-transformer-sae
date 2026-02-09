import numpy as np
import pathlib

samples_dir = pathlib.Path('exp/sod/sparse_steering/generations_sas/average_pitch')
npy_file = list(samples_dir.glob('lambda_*/sample_*.npy'))[0]
tokens = np.load(npy_file)

print(f'File: {npy_file.name}')
print(f'Shape: {tokens.shape}')
print(f'Dtype: {tokens.dtype}')
print(f'Min/Max: [{tokens.min()}, {tokens.max()}]')
print(f'First 5 rows:\n{tokens[:5]}')
if len(tokens.shape) == 2:
    print(f'Type column unique: {np.unique(tokens[:, 0])}')
else:
    print('Token array is not 2D!')
