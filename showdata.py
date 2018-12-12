import numpy as np 
import sys

def plot_im(imgs):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(20.5, 10.5)
    axs[0].imshow(imgs[0,...])
    axs[1].imshow(imgs[1,...])
    fig.tight_layout()
    plt.show()

datapath = sys.argv[1]
data = np.load(datapath)
print("cubeindx",data["cubeindx"],"actionindx:",data["actionindx"])
plot_im(data["video"])