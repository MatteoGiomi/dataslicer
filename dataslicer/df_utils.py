#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of dataframe-related utils
#
# Author: M. Giomi (matteo.giomi@desy.de)

import pandas as pd
from astropy.io import fits


def fits_to_df(fitsfile, extension, columns = None, keep_array_cols = False):
    """
        load a table extension contained into a fits file into a pandas 
        DataFrame.
        
        Parameters:
        -----------
            
            fitsfile: `str`
                valid path to the file you want to read.
            
            extension: `str` or `int`
                extension to be read. This will be the second parameter
                passed to astropy.io.fits.getdata
            
            columns: `list`
                list of column names to be read from the files. If None, 
                all the columns will be read.
            
            keep_array_cols: `bool`
                if True and array columns are present in the fits table,
                they will be converted and included in the dataframe.
        
        Returns:
        --------
            
            pandas.DataFrame with the content of the table extension of the fitsfile.
    """
    data = fits.getdata(fitsfile, extension)
    datadict = {}
    for dc in data.columns:
        if not columns is None and ( not any([uc.replace("*", "") in dc.name for uc in columns]) ):
            continue
        if int(dc.format[0]) > 1:
            if not keep_array_cols:
                continue
            else:
                datadict[dc.name] = data[dc.name].byteswap().newbyteorder().tolist()
        else:
            datadict[dc.name] = data[dc.name].byteswap().newbyteorder()
    return pd.DataFrame(datadict)


def check_col(col, df):
    """
        check if given column(s) is present in a DataFrame or a GroupBy object.
        
        Parameters:
        -----------
            
            col: `str` or `list`
                name or list of names of columns to check.
            
            df: `pandas.DataFrame`
    """
    # get column names for both dataframes or groupby objects
    try:
        df_cols = df.columns
    except AttributeError:
        df_cols = df.obj.columns
    
    # handle list / str / None
    if type(col) == str or col is None:
        cols = [col]
    else:
        cols = list(col)
    
    # check
    for c in cols:
        if (not c is None) and (c not in df_cols):
            raise KeyError("column %s not present in objtable dataframe. Availables are: %s"%
                (c, ", ".join(df_cols.values)))


# -------------------------------------- #
#  functions that opearate on df groups  #
# -------------------------------------- #

def groupby_to_df2(grpby):
    """
        convert a pandas groupby object into a dataframe
    """
    return grpby.apply(lambda x: x)

def subtract_dfs(df1, df2):
    """
        give two like-shaped dataframes, return a dataframe containing the rows
        of df1 which are not in df2.
    """
    
    df_all = df1.merge(df2.drop_duplicates(), on=None, 
            how='left', indicator=True)
    return df1[df_all['_merge'] == 'left_only']

def group_stats(group):
    """
        this function acts on a dataframe group and return some basic stats
        in a dictionary.
    """
    return {
        'min': group.min(), 
        'max': group.max(),
        'count': group.count(),
        'mean': group.mean(), 
        'std': group.std()}

