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



def tag_dust(df, dust_df_file, radius_multiply = 1., xname = 'xpos', yname = 'ypos',
    dust_df_query = None, remove_dust = False):
        """
            tag objects in df whose x/y position on the CCD quadrant falls on top
            of a dust grain.
            
            Parameters:
            -----------
                
                dust_df_file: `str`
                    path to a csv file containing x, y, and radius for all the dust
                    grain found in the image. This file is created by ztfimgtoolbox.get_dust_shapes
                
                radius_mulitply: `float`
                    factor to enlarge/shrink the dust grain radiuses.
                
                x[y]name: `str`
                    name of this object df columns describing the x/y position of the objects.
                
                dust_df_query: `str`
                    string to select dust grains to consider. Eg:
                        "(dx < x < (3072 - dx)) and (dy < ypos < (3080 - dy))"
                
                remove_dust: `bool`
                    if True, objects falling on the dust will be removed from the dataframe.
            
            Returns:
            --------
                
                pandas.DataFrame containing the objects contaminated by dust.
        """
        
        from shapely.geometry import Point
        from shapely import vectorized
        
        # read in the df file with the dust grains
        dust_df = pd.read_csv(dust_df_file)
        if not dust_df_query is None:
            dust_df.query(dust_df_query, inplace = True)
        print ("read position of %d dust grains from file: %s"%(len(dust_df), dust_df_file))
        
        # sort the dust dataframe by dust size, smaller first. This way, in case of
        # close-by dust grains, the source is assigned to the largest one.
        dust_df = dust_df.sort_values('r')
        
        # loop on the dust grains and match them to the sources
        print ("matching sources %d objects with %d dust grains.."%(len(df), len(dust_df)))
        for _, d in dust_df.iterrows():
            
            # create the geometrical object for this dust grain
            dust_g = Point(d['x'], d['y']).buffer( (radius_multiply * d['r']) )
            
            # find out all the sources that matches this dust grain
            flagin = vectorized.contains(dust_g, df[xname], df[yname])
#            print ("found %d sources inside a dust grain of radius: %.2f"%(sum(flagin), d['r']))
            
            # add the dust property to the dataframe
            df.loc[flagin, 'dust_x'] = d['x']
            df.loc[flagin, 'dust_y'] = d['y']
            df.loc[flagin, 'dust_r'] = d['r']
        
        dust_matches = df.count()['dust_r']
        print ("found %d sources on top of dust grains."%(dust_matches))
        
        # split the dataframe into dust and non-dust, eventually remove 
        # the dust from the source dataframe
        srcs_w_dust = df.dropna(axis=0, subset = ['dust_r'])
        if remove_dust:
            df = df[df['dust_r'].isna()]
            print ("removing sources affected by dust. %d retained."%(len(df)))
        return df, srcs_w_dust

# -------------------------------------------------- #
#  functions that takes care of the PS1 calibrators  #
# -------------------------------------------------- #


def match_to_PS1cal_fields():
    
    raise NotImplementedError("FIELD based PS1 cal matching still to be done.")
    # TODO: implement the following sketch for real
    
     # get the coordinates of all the calibrators for those fields
    ps1srcs = [src for src in ps1cal_query.src_coll.find(
    {'field': { "$in": fields}, 'rcid': {'$in': rcids}},
    { '_id': 1, 'ra': 1, 'dec': 1 })]
    
    # first find the closest calibrator for each object
    coords = SkyCoord(self.tab[xname], self.tab[yname], **skycoords_kwargs)
    ps1coords = SkyCoord(
        ra = ps1cals['ra']*u.deg, dec = ps1cals['dec']*u.deg, **PS1skycoords_kwargs)
    idps1, d2ps1, _  =  coords.match_to_catalog_sky(ps1coords)

    # add these indexes and distances to the table
    newcols = [Column(name = "PS1_ID", dtype = int, data = idps1),
             Column(name = "D2PS1", data = d2ps1.to('arcsec').value, unit = 'arcsec')]
    self.tab.add_columns(newcols)

    # then select the matches that are more apart than sep
    matches = self.tab[self.tab['D2PS1']<sep.to('arcsec').value]
    logging.info("%d objects are within %.2f arcsec to a PS1 calibrator"%(
        len(matches), sep.to('arcsec').value))

    # finally group them accoring to the PS1 id.
    cals = matches.group_by('PS1_ID')

    # check that, no more than one object per dataset is
    # associated to any PS1 calibrator
    if np.any(np.diff(cals.indices)>len(self.hdf5paths)):
        logging.warning(
            "some PS1 calibrator has more than %d matches."%len(self.hdf5paths))
        logging.warning("there are probably problems with the matching.")


def match_to_PS1cal(ras, decs, rs_arcsec, ids, ps1cal_query = None,
    col2rm = ['rcid', 'field', '_id', 'hpxid_16'],  show_pbar = True):
    """
        given a list of coordinates, return a dataframe containing
        all the PS1 calibrator sources that matches to those positions.
        
        Pararameters:
        -------------
            
            ras, decs: `array-like`
                sky coordinates (Equatorial, J2000) for which you want to find the matches
            
            rs_arcsec: `float`
                search radius in arcseconds for the matching. Only PS1 sources that are less
                than rs_arcsec away from one coordinate pair are retained.
            
            ps1cal_query: `extcats.CatalogQuery`
                object that will take care of doing the PS1 matching.
            
            ids: `str` or (str, ids) tuple
                ID of the sources to makes it possible to attach, to each coordinate pair,
                the corresponding PS1cal.
                If ids is string, the resulting datframe will have a column called
                ids and the element in this column will be the index of the coordinate
                pair in the inpu lists.
                If ids is a (str, ids) tuple, the resulting dataframe will have a column
                named ids[0] and will take the values given in ids[0].
            
            col2rm: `list`
                names of columns in the PS1 calibrator catalogs to be excluded from the
                resulting dataframe. If None, no PS1cal column is removed.
            
            show_pbar: `bool`
                if you want to use tqdm to show fancy progress bars.
        
        Returns:
        --------
        
            pandas.DataFrame with the found PS1cal sources, formatted according to 
            col2rm and id_key.
    """
    
    # initialize the catalog query object and grab the database
    if ps1cal_query is None:
        from extcats import CatalogQuery
        ps1cal_query = CatalogQuery.CatalogQuery(
            'ps1cal', 'ra', 'dec', dbclient = None, logger = None)
    
    if len(ras)!=len(decs):
        raise RuntimeError("ra/dec coordinate lists have different lengths.")
    if not type(ids) is str:
        if len(ids) != 2:
            raise ValueError("provided ids argument is not a list/tuple/ecc.. of 2 elements.")
            if not type(ids[0]) is str:
                raise ValueError("first element of ids should be a string.")
            if not len(ids[1]) == len(ras):
                raise ValueError("list of id values has different length than coordinates.")
    
    # loop on the coordinate pairs and create list of dictionaries
    ps1cps = []
    if show_pbar:
        import tqdm
        coord_iter = tqdm.tqdm(range(len(ras)))
    else:
        coord_iter = range(len(ras))
    for ic in coord_iter:
        ps1match =  ps1cal_query.findclosest(ra = ras[ic], dec = decs[ic], 
            rs_arcsec = rs_arcsec, method = 'healpix')
            
        # if you have a cp, add it to the list
        if ps1match != (None, None):
            buff = {}
            for c in ps1match[0].colnames:
                if (not col2rm is None) and (c not in col2rm):     # remove unwanted columns
                    buff[c] = ps1match[0][c]
            buff['dist2ps1'] = ps1match[1]
            if type(ids) is str:
                buff[ids] = ic
            else:
                buff[ids[0]] = ids[1][ic]
            ps1cps.append(buff)
                
    # merge the dataframe
    ps1cp_df = downcast_df(pd.DataFrame(ps1cps))
    return ps1cp_df


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

