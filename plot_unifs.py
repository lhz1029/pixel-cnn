import numpy as np
import matplotlib.pyplot as plt
import argparse
from functools import partial
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("-n", type=str, default='',
                        help="Location of checkpoint to restore")
    args = parser.parse_args()
    unifs = np.load(args.f)
    fig, axes = plt.subplots(5, 5)
    for i, el in enumerate(unifs[:25]):
        axes[i//5, i%5].hist(el.flatten())
    plt.savefig(args.n)
    # plt.show()