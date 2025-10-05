# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_lightcurve(flux, title=None, savepath=None):
    plt.figure(figsize=(8,2))
    plt.plot(flux)
    plt.axhline(0, color='k', alpha=0.2)
    if title:
        plt.title(title)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
