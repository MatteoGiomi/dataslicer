#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of dataframe-related utils
#
# Author: M. Giomi (matteo.giomi@desy.de)

import pandas as pd
from astropy.io import fits

def fits_to_df(fitsfile, extension, select_columns = 'all', keep_array_cols = False, 
    select_rows = None, downcast = False, verbose = False):
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
            
            select_columns: `list`
                list of column names to be read from the files. If 'all', 
                all the columns will be read.
            
            keep_array_cols: `bool`
                if True and array columns are present in the fits table,
                they will be converted and included in the dataframe.
            
            select_rows: `str` or None,
                query expression used to select the rows you want to retain. This is
                passed to pd.DataFrame.query as the expr parameter. If None, no cut
                will be applied. 
            
            downcast: `bool`
                if True, uses pd.to_numeric to downcast ints and floats columns
                in order to reduce RAM usage to the expense of ~ 30% loss in speed.
        
        Returns:
        --------
            
            pandas.DataFrame with the content of the table extension of the fitsfile.
    """
    if verbose:
        print ("fits_to_df: reading file %s"%fitsfile)
    
    magic_cols = [c for c in select_columns if '*' in c]   # select cols for which you want to use wildchar
    normal_cols = list(set(select_columns) - set(magic_cols))
    data = fits.getdata(fitsfile, extension)
    datadict = {}
    for dc in data.columns:
        if not select_columns == 'all':
            if not ( (dc.name in normal_cols) or 
                     (any([mc.replace("*", "") in dc.name for mc in magic_cols])) ):
                continue
        # check for array col
        if int(dc.format[0]) > 1:
            if not keep_array_cols:
                continue
            else:
                datadict[dc.name] = data[dc.name].byteswap().newbyteorder().tolist()
        else:
            datadict[dc.name] = data[dc.name].byteswap().newbyteorder()
    df = pd.DataFrame(datadict)
    if not select_rows is None:
        df = df.query(select_rows)
    if downcast:
        df = downcast_df(df)
    return df


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

def stringinlist(key, list_of_keys):
    """
        check if a string key is present at least once in list_of_keys.
        This supports '*' as wildchar: 
            
            >> ll = ['apple_gala', 'fuffa', 'apple_fuji', 'bananas']
            >> stringinlist('appl*', ll)
            >> True
            >> stringinlist('app', ll)
            >> False
            >> stringinlist('fuffa', ll)
            >> True
            >> stringinlist('fu*', ll)
            >> True
    """
    if '*' in key:
        mk = key.replace('*', '')
        return any([mk in key_from_list for key_from_list in list_of_keys])
    else:
        return (key in list_of_keys)

def strlist_in_strlist(strlist1, strlist2):
    """
        check if strlist1 is contained in strlist2. Primitive wildchar support.
    """
    ifound = 0
    for str1 in strlist1:
        if stringinlist(str1, strlist2):
            ifound += 1
    if ifound == len(strlist1):
        return True
    else:
        return False

def downcast_df(df, verbose = False):
    """
        use to_numeric function to downcast int and floats in a dataframe
        with the hope of reducing memory footprint and improving performances.
        Following:
        https://www.dataquest.io/blog/pandas-big-data/
    """
    # doncast ints and floats
    dwnc_ints = df.select_dtypes(include=['int']).apply(pd.to_numeric, downcast='unsigned')
    dwnc_floats = df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    
    # assign it to a new dataframe
    dwnc_df = df.copy()
    dwnc_df[dwnc_ints.columns] = dwnc_ints
    dwnc_df[dwnc_floats.columns] = dwnc_floats
   
    if verbose:
        print("before downcasting:", mem_usage(df))
        print("after downcasting:", mem_usage(dwnc_df))
    return dwnc_df
    

def mem_usage(pandas_obj):
    """
        return RAM footprint of a DataFrame or a Series. Taken from:
        https://www.dataquest.io/blog/pandas-big-data/
    """
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def subtract_dfs(df1, df2):
    """
        give two like-shaped dataframes, return a dataframe containing the rows
        of df1 which are not in df2.
    """
    
    df_all = df1.merge(df2.drop_duplicates(), on=None, 
            how='left', indicator=True)
    return df1[df_all['_merge'] == 'left_only']


# -------------------------------------- #
#  functions that opearate on df groups  #
# -------------------------------------- #

def cluster_op(gdf, col, function):
    """
        apply a function to each group in the dataframe and 
        return a dataframe with the results.
        
        Parameters:
        -----------
            
            gdf: `pandas.GroupByObject`
                groupby object.
        
            col: `str`
                name of the column on which the operation is to be performed.
            
            op: `callable` or `str`
                the function to be applied to the desired column of each group.
                Must reuturn a dictionary so that the results can be parsed into
                another dataframe. If string, it can be used to select functions
                from the df_utils module.
                
        
        Returns:
        --------
            
            pandas.DataFrame with the clusterID as index and the key in the
            function 
        
    """
    check_col(col, gdf)
    
    # see if it's in df_utils
    if type(function) == str:
        try:
            func = getattr(__name__, function)
            self.logger.info("using function %s from df_utils module"%function)
        except AttributeError:
            self.logger.info("using user defined function %s"%function.__name__)
    else:
        func = function
    # apply and return
    return gdf[col].apply(func).unstack()


def groupby_to_df2(grpby):
    """
        convert a pandas groupby object into a dataframe
    """
    return grpby.apply(lambda x: x)


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

