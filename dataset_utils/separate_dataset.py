import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='pymks_dataset_100.npy')
    args = parser.parse_args()

    assert os.path.isfile(args.infile)
    assert args.infile.endswith('.npy')

    data = np.load(args.infile)

    assert len(data.shape) in (2, 3, 4)

    for i, ms in enumerate(data):
        tail, head = os.path.split(args.infile)
        filename = os.path.join(tail, 'separated_npy_files', f'{head[:-4]}_{i}.npy')
        if ms.shape[-1] == 1:
            ms = ms.reshape(ms.shape[:-1])
        np.save(filename, ms)

if __name__ == "__main__":
    main()
