import argparse
import pymks
from pymks.datasets import make_microstructure
import numpy as np


def get_ms(shape=(101, 101), n_samples=1):
    size = shape
    n_phases = 2
    # grain_size = [(40, 20)]
    # v_frac = [(0.9, 0.1)]
    grain_size = [(40, 10)]
    v_frac = [(0.65, 0.35)]
    per_ch = 0.1
    
    #generate data
    dataset = np.concatenate([make_microstructure(n_samples=n_samples, 
                                                  size=size, 
                                                  n_phases=n_phases,
                                                  grain_size=gs,
                                                  volume_fraction=vf,
                                                  percent_variance=per_ch)
                                for gs, vf in zip(grain_size,v_frac)])
    return dataset

def generate_testset(lin_width, n_samples):
    ms_shape = (lin_width, lin_width)
    data = get_ms(shape=ms_shape, n_samples=n_samples)
    filename = f'pymks_dataset_{n_samples}.npy'
    np.save(filename, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lin_width', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()
    
    generate_testset(args.lin_width, args.n_samples)
