import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns

sns.color_palette("viridis", as_cmap=True)

class DataLoader:
    def __init__(self) -> None:
        self.data = None
    
    def load_csv(self, csv: str)-> None:
        self.data = pd.read_csv(csv)
    
    def load_pickle(self, pth: str)->None:
        self.data = pd.read_pickle(pth)

    def load_feather(self, file:str)->None:
        self.data = pd.read_feather(file)
    
    def pickle_df(self, pth :str)->None:
        self.data.to_pickle(pth)
    
    def feather_df(self, pth:str)-> None:
        self.data.to_feather(pth)

    def display(self, num: int, head: bool)->DataFrame:
        return self.data.head(num) if head else self.data.tail(num)
        
    
    def describe(self)->DataFrame:
        return self.data.describe(include='all')
    
    def get_col_unique_freq(self)->list:
        arr = []
        for col in self.data:
            if self.data[col].dtype.name == 'category':
                grouped_data = self.data.groupby(by=col).size()
                arr.append(grouped_data)
        
        return arr
    
    def create_subplots(self, data: list, subrow: int)->None:
        subcol = ceil(len(data)/subrow)
        fig, axes = plt.subplots(subrow, subcol, figsize = (subrow*8, subcol*8))
        current = 0
        for subaxes in axes:
            for axis in subaxes:
                data[current].plot(ax = axis, kind='bar')
                current += 1
                if current == len(data): return
        plt.tight_layout()

    def corr_heatmap(self)->None:
        plt.figure(figsize=(15, 10))
        mask = np.zeros_like(self.data.corr())
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(self.data.corr(), annot=True, mask=mask, cmap="viridis", linewidths=0.2, linecolor='black')
    
    def map_column(self, colname: str, ref: dict )->None:
        self.data[colname] = self.data[colname].replace(ref)

        
    def drop_column(self, cols: list)->DataFrame:
        return self.data.drop(cols, inplace = True, axis = 1)
    
    def display_types(self)-> pd.Series:
        return self.data.dtypes
    
    def column_to_string(self, col: list)->None:
        self.data[col] = self.data[col].astype(str)
    
    def column_to_cat(self, col: list)->None:
        self.data[col] = self.data[col].astype('category')
    
    