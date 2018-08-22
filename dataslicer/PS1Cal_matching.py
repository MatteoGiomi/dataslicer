#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of functions to match ZTF source catalogs with PS1 calibrator stars.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import time
import pandas as pd
import numpy as np
import logging
from astropy.coordinates import SkyCoord 
from dataslicer.df_utils import check_col, downcast_df
logging.basicConfig(level = logging.INFO)

def init_catalog_query():
    from extcats import CatalogQuery
    return CatalogQuery.CatalogQuery(
            'ps1cal', 'ra', 'dec', dbclient = None, logger = None)

def match_to_PS1cal_fields(df, rs_arcsec, ra_key='ra', dec_key='dec', fieldid_col='FIELDID',
    rcid_col='RCID', col2rm = ['rcid', 'field', '_id', 'hpxid_16'], ps1cal_query = None, 
    clean_non_matches = True, logger = None):
    """
        match sources in the dataframe to those in the PS1Cal database 
        using field and RC Ids to speed up the process.
        
        Parameters:
        -----------
        
            df: `pd.DataFrame`
                dataframe with the sources you want to match
            
            rs_arcsec: `float`
                search radius for the match. The closest source within rs_arcsec is chosen.
            
            ra[dec]_key: `str`
                name of columns in df specifying RA and Dec of the sources (J2000).
            
            field[rc]id_col: `str`
                name of columns in df with field and RC id.
            
            col2rm: `list`
                which columns of the PS1Cal db you want to purge.
            
            ps1cal_query: `extcats.CatalogQuery`
                object that will take care of doing the PS1 matching.

            clean_non_matches: `bool`
                if True, sources with no match in the PS1cal db are removed.
            
            logger: `logging.logger`
                instance of logger to use.
        
        Returns:
        -------
            
            df with the matches or None.
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)

    check_col([fieldid_col, rcid_col, ra_key, dec_key], df)
    
    # then check that all the fields in this object df
    # are present in the PS1Cal database
    my_fields = df[fieldid_col].unique().astype(int).tolist()
    my_rcs = df[rcid_col].unique().astype(int).tolist()
    logger.info("found objtable sources for %d fields and %d RCs."%
        (len(my_fields), len(my_rcs)))
    
    # initialize the catalog query object and grab the database
    if ps1cal_query is None:
        ps1cal_query = init_catalog_query()
    
    # get the ones you have in the catalog:
    ps1cal_fields = ps1cal_query.src_coll.distinct('field')
    logger.info("PS1 calibrator database is indexed in %d fields"%
        (len(ps1cal_fields)))
    
    # see if the fields in the objtable are also in the database
    if not all([ff in ps1cal_fields for ff in my_fields]):
        logger.info("FIELDID and/or RCID missing from objtable dataframe.")
        return None

    # -------------------------------------------- #
    #               now the matching               #
    # -------------------------------------------- #

    # get the calibrators for those fields and RCs
    query = {'field': { "$in": my_fields}, 'rcid': {'$in': my_rcs}}
    proj = None #{ 'ra': 1, 'dec': 1 }      # TODO: first just get the coordinates, then, for matches, get the rest.
    ps1df = pd.DataFrame(
        [src for src in ps1cal_query.src_coll.find(query, proj)])
    logger.info("querying PS1Cal database for all sources in your fields and RCs: found %d sources."%
        len(ps1df))
    
    # use astropy coordinates to find the closest match
    ps1_coords = SkyCoord(ps1df['ra'], ps1df['dec'], unit = 'deg')
    my_coords = SkyCoord(df[ra_key], df[dec_key], unit = 'deg')
    logger.info("matching coordinates..")
    start = time.time()
    closest_ps1_ids, d2ps1, _ = my_coords.match_to_catalog_sky(ps1_coords)
    logger.info("done. Took %.2e seconds."%(time.time()-start))
    
    # add ID and distance of closest PS1 cp to source dataframe
    start = time.time()
    logger.info("appending PS1 info to source dataframe.")
    df['_id'] = [ps1df._id[ps1id] for ps1id in closest_ps1_ids]
    df['dist2ps1'] = d2ps1.arcsec
    
    # either remove sources with matches too far away or flag them
    if clean_non_matches:
        df = df[df['dist2ps1']<rs_arcsec]
    else:
        df.loc[df['dist2ps1']>rs_arcsec, 'dist2ps1'] = np.nan
    
    # merge the dataframes
    df = df.merge(ps1df, on = '_id', how='left',  suffixes=['', '_ps1'])
    logger.info("done. Took %.2e seconds."%(time.time()-start))
    
    # remove uselsess columns and return
    if not col2rm is None:
        df.drop(columns=col2rm, inplace=True)

    # remove non
    if clean_non_matches:
        df.dropna(subset = ['dist2ps1'], inplace = True)
        logger.info("dropping sources without match in PS1cal DB: %d retained."%(len(df)))
    
    return df


def match_to_PS1cal(ras, decs, rs_arcsec, ids, ps1cal_query = None,
    col2rm = ['rcid', 'field', '_id', 'hpxid_16'], show_pbar = True):
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
                pair in the input lists.
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
        ps1cal_query = init_catalog_query()
    
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

def stats(df, filter_col = 'FILTERID', ps1color_col = None, ps1mag_col = None, withzp_col = None, withcolor_col = None,
    airmass_col = 'airmass', logger = None):
    '''
        This method returns some statistics about a given dataframe, such as # of calibrator stars, mean color, min magnitude in each band etc.
        Also, the dataframe gets expanded with some useful columns if not supplied as parameter
        
        Pararameters:
        -------------

            df: `pd.DataFrame`
                dataframe you're interested in

            filtercol: `str`
                name of the column in the dataframe containing the filterid (1-3)

            ps1color_col: `str`
                name of the column in the dataframe containing the PS1 color

            ps1mag_col: `str`
                name of the column in the dataframe containing the PS1 magnitude

            withzp_col: `str`
                name of the column in the dataframe containing the ZTF magnitude + zeropoint from header

            withcolor_col: `str`
                name of the column in the dataframe containing the ZTF magnitude + zeropoint from header + colorterm from header 

        Returns:
        --------

            dict with the following entries:

                number_of_datapoints: number of entries

                number_of_stars: the number of unique stars

                summed_airmass: sum over the airmass

                min_magnitude_PS1: the smallest PS1 magnitude

                min_magnitude_ZTF_calibrated: the smallest ZTF magnitude (after applying zeropoint and color correction)

                min_magnitude_lightcurve_averages: the smallest of the lightcurve average magnitudes
    '''

    if logger is None:
        logger = logging.getLogger(__name__)

    # choose the right PS1 color (according to filterid)
    if ps1color_col is None and ps1mag_col is None:

        ps1color_col = 'PS1_color'
        ps1mag_col = 'PS1_mag'

        # create auxiliary columns
        df['PS1_color_g'] = df['gmag'].values - df['rmag'].values
        df['PS1_color_r'] = df['gmag'].values - df['rmag'].values
        df['PS1_color_i'] = df['rmag'].values - df['imag'].values

        # select right PS1 color column using masking on FILTERID
        gr_mask = np.logical_or(df['FILTERID'] == 1, df['FILTERID'] == 2)
        df[ps1color_col] = df['PS1_color_g'].where(df['FILTERID'] == 1, df['PS1_color_r'])
        df[ps1mag_col] = df['gmag'].where(df['FILTERID'] == 1, df['rmag'])
        df[ps1color_col] = df[ps1color_col].where(gr_mask, df['PS1_color_i'])
        df[ps1mag_col] = df[ps1mag_col].where(gr_mask, df['imag'])


        # drop the the auxiliary columns
        df.drop(columns = ['PS1_color_g', 'PS1_color_r', 'PS1_color_i'], inplace = True)

    if withzp_col is None:

        withzp_col = 'mag_with_zp'
        df[withzp_col] = df['mag'].values + df['MAGZP'].values

    if withcolor_col is None:

        withcolor_col = 'mag_color_corrected'
        df[withcolor_col] = df[withzp_col] + ( df['CLRCOEFF'] * df[ps1color_col] )

    # determine number of datapoints and stars
    datapoints_nr = df.shape[0]
    stars_nr = len(df['_id'].unique())
    airmass_sum = df['airmass'].sum()

    # calculate lightcurve averages
    lightcurve_ztf_avg = df.groupby('_id')[withcolor_col].mean()
    lightcurve_ps1_avg = df.groupby('_id')[ps1mag_col].mean()

    # calculate smallest magnitudes
    min_magnitude_ps1 = max(df[ps1mag_col].values)
    min_magnitude_ztf_calibrated = max(df[withcolor_col].values)
    min_magnitude_lc = max(lightcurve_ztf_avg)

    return_dict = {'number_of_datapoints': datapoints_nr, 'number_of_stars': stars_nr, 'summed_airmass': airmass_sum,
    'min_magnitude_PS1': min_magnitude_ps1, 'min_magnitude_ZTF_calibrated': min_magnitude_ztf_calibrated,
    'min_magnitude_lightcurve_averages': min_magnitude_lc}

    logger.info('number of datapoints: {}'.format(datapoints_nr))
    logger.info('number of stars: {}'.format(stars_nr))
    logger.info('airmass summed: {:.0f}'.format(airmass_sum))
    logger.info('smallest PS1 magnitude = {:2.2f}'.format(min_magnitude_ps1))
    logger.info('smallest calibrated star magnitude = {:2.2f}'.format(min_magnitude_ztf_calibrated))
    logger.info('smallest lightcurve average magnitude = {:2.2f}'.format(min_magnitude_lc))

    return return_dict
