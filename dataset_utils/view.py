import matplotlib.pyplot as plt
import numpy as np
import argparse

def vf(x):
    return np.average(x / 255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='./samples_4x256x256x3.npz')
    parser.add_argument('--number', type=int, default=0)
    args = parser.parse_args()

    x = np.load(args.infile)['arr_0'][args.number]
    y = None
    try:
        y = np.load(args.infile)['arr_1'][args.number]
    except Exception:
        pass
    
    print(vf(x))

    savefig = False
    plt.figure(figsize=(4, 4))
    plt.imshow(x, cmap='gray')
    plt.tight_layout()
    if y is not None:
        plt.title(y)
    if savefig:
        plt.savefig('plot.png', dpi=600, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    


    
