from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
sns.color_palette("viridis", as_cmap=True)

"""Plotting utility for EDA"""

def bar_subplots(data: list, subrow: int, figsize_ep=5)->None:
    subcol = ceil(len(data)/subrow)
    fig, axes = plt.subplots(subrow, subcol, figsize=(subrow*figsize_ep, subcol*figsize_ep), constrained_layout=True)
    current = 0
    for subaxes in axes:
        for axis in subaxes:
            data[current].plot(ax = axis, kind='bar')
            current += 1
            if current == len(data):
                return fig, axes

def heatmap(corr: DataFrame)->None:
    plt.figure(figsize=(15, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, mask=mask, cmap="viridis", linewidths=0.2, linecolor='black')
    return

