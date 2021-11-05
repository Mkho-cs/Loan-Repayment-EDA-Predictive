from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
sns.color_palette("viridis", as_cmap=True)

"""Plotting utility for EDA"""

def cat_subplots(data: list, subrow: int)->None:
    subcol = ceil(len(data)/subrow)
    fig, axes = plt.subplots(subrow, subcol, figsize = (subrow*8, subcol*8))
    plt.tight_layout()
    current = 0
    for subaxes in axes:
        for axis in subaxes:
            data[current].plot(ax = axis, kind='bar')
            current += 1
            if current == len(data): return

def corr_heatmap(data: DataFrame)->None:
    plt.figure(figsize=(15, 10))
    mask = np.zeros_like(data.corr())
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(data.corr(), annot=True, mask=mask, cmap="viridis", linewidths=0.2, linecolor='black')
    return

