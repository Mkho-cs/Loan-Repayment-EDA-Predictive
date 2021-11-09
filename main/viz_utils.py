from typing import Tuple
from typing_extensions import TypeAlias
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from math import ceil
sns.color_palette("viridis", as_cmap=True)

"""Plotting utility for EDA"""

def bar_subplots(data: list, subrow: int, figsize_ep=5)->Tuple[plt.Figure, plt.Axes]:
    subcol = ceil(len(data)/subrow)
    fig, axes = plt.subplots(subrow, subcol, figsize=(subrow*figsize_ep, subcol*figsize_ep), constrained_layout=True)
    current = 0
    for subaxes in axes:
        for axis in subaxes:
            data[current].plot(ax = axis, kind='bar')
            current += 1
            if current == len(data):
                return fig, axes

def hist_subplots(df: DataFrame, subrow:int, figsize_ep=5, nbins=15)->Tuple[plt.Figure, plt.Axes]:
    length = df.shape[1] 
    subcol = ceil(length/subrow)
    fig, axes = plt.subplots(subrow, subcol, figsize=(subrow*figsize_ep, subcol*figsize_ep), constrained_layout=True)
    current = 0
    for r in range(subrow):
        for c in range (subcol):
            df.hist(column=df.columns[current], ax=axes[r,c], bins=nbins, color='green')
            current += 1
            if current == length:
                return fig, axes

def heatmap(df: DataFrame)->None: 
    plt.figure(figsize=(15, 10))
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df, annot=True, mask=mask, cmap="viridis", linewidths=0.2, linecolor='black')
    return

