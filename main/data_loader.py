from numpy import dtype, not_equal
import pandas as pd
from pandas.core.frame import DataFrame, Series
from typing import TypeVar, List

Numeric = TypeVar('Numeric', int, float)

class DataLoader:
    def __init__(self, df:DataFrame=None) -> None:
        self.data = df
    
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
    
    def get_cols_unique_freq(self, datatype:str)->list:
        arr = []
        for col in self.data:
            if self.data[col].dtype.name == datatype:
                grouped_data = self.data.groupby(by=col).size()
                arr.append(grouped_data)
        
        return arr
    
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
    
    def convert_dtype(self, origin: str, target: str)->None:
        for col in self.data:
            if self.data[col].dtype.name == origin:
                self.data[col] = self.data[col].astype(target)
        return
    
    def single_eqfilter(self, col: str, val)->None:
        self.data = self.data.loc[self.data[col] == val]
        return
    
    def groupby_sum(self, group:List[str], target: List[str])->DataFrame:
        new = self.data.groupby(group)[target].sum().reset_index()
        return new
    
    def chosen_cols(self, cols: List[str])-> None:
        self.data = self.data[cols]
        return None
    
    def left_join(self, right: DataFrame, left_key: str, right_key: str)->DataFrame:
        joined = pd.merge(self.data, right, how='left', left_on=left_key, right_on=right_key)
        return joined
    
    def col_parse_year(self, colname:str)->Series:
        return pd.DatetimeIndex(self.data[colname]).year

    def add_col(self, newcol: Series, colname: str)->None:
        self.data[colname] = newcol
        return 
    
    def na_per_col(self)->DataFrame:
        return self.data.isna().sum()
    
    def fillna_median(self, colname: str)->None:
        self.data[colname].fillna(self.data[colname].median(), inplace=True)
        return
    
    def interpolate_smooth(self, colname: str)-> None:
        self.data[colname].interpolate(method='akima', inplace=True)
        return
    
    def drop_na_rows(self)->None:
        self.data.dropna(inplace=True)
        return
    
