#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# class to collect and manage the metadata (airmass, seeing, ccd temp, time, 
# zero point, ecc..) of a given dataset into a dataframe
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os, tqdm, logging
from astropy.io import fits
import pandas as pd

from dataslicer.dataset_base import dataset_base
from dataslicer.df_utils import downcast_df, check_col


def load_IRSA_meta(df, IRSA_meta_cols = ['airmass'], expid_col = 'EXPID', logger = None):
        """
            retrieve metadata for each file from the IRSA archieve. It uses the EXPID
            header keyword to search for the right metadata and then merge the dfs.
            
            Requires IRSA account and ztfquery.
            
            Parameters:
            -----------
            
                df: `pandas.DataFrame` or `dataslicer.srcdf`
                    dataframe-like object to which you want to add IRSA metadata.
            
                IRSA_meta_cols: `list`
                    names of the IRSA metadata you want to add to the fits file metadata.
                    In None, all the IRSA columns will be added.
                
                expid_col: `str`
                    name of metadata column containing the expid.
                
                logger: `logging.logger`
                    logger instance. If none, default will be used.
        """
        
        if logger is None:
            logging.basicConfig(level = logging.INFO)
            logger = logging.getLogger(__name__)
        
        # check which are the expid we have in this object
        check_col(expid_col, df)
        expids = pd.unique(df[expid_col])
        logger.info("found %d unique exposures (%s) in metadata."%
            (len(expids), expid_col))
        expids_str = ["%d"%expid for expid in expids]
        
        # query IRSA
        from ztfquery import query
        zquery = query.ZTFQuery()
        query_str = "expid+IN+(%s)"%(",".join(expids_str))
        logger.info("querying IRSA using: %s"%query_str)
        zquery.load_metadata(kind="sci", sql_query="%s"%query_str)
        logger.info("retrieved %d metadata"%len(zquery.metatable))

        # select which IRSA columns to add
        if not IRSA_meta_cols is None:
            logger.info("selecting IRSA meta columns: %s"%", ".join(IRSA_meta_cols))
            IRSA_meta_cols.append('expid')  # you need this to join the dfs
            metatable = zquery.metatable[IRSA_meta_cols]
        else:
            logger.info("using all IRSA meta columns.")
            metatable = zquery.metatable
        logger.info("adding the following columns to metadata dataframe: %s"%
            (", ".join(metatable.columns.values)))
        
        # join the dataframe
        metatable = metatable.rename(columns={'expid': expid_col})
        df = df.merge(metatable.drop_duplicates(), on = expid_col)
        logger.info("joined IRSA meta to dataframe. The following columns are now available: %s"%
            (", ".join(df.columns.values)))
        return df
        

class metadata(dataset_base):
    """
        class to collect and manage the metadata of a given dataset.
    """
    
    def to_csv(self, **args):
        """
        """
        self._to_csv(tag = 'metadata', **args)


    def read_csv(self, **args):
        """
        """
        self._read_csv(tag = 'metadata', **args)


    def load_header_meta(self, header_keys = None, downcast = True, **getheader_args):
        """
            go and read the header of the fits files and create a 
            dataframe with the desired header keywords. The dataframe
            will be stored in this object's metadata attribute. It will have 
            one row for each file, one column for each header key plus the
            absolute path of the file.
            
            Parameters:
            -----------
            
                header_keys: `list`, or None
                    list of header key names that will form the columns of the 
                    dataframe. If None, a default list of keywords will be used.
                    primitive wildchar support is possible (e.g: APCOR* will take
                    all the keywords with APCOR in the name).
                
                downcast: `bool`
                    if True, uses pd.to_numeric to downcast ints and floats columns
                    in order to reduce RAM usage.
                
                getheader_args: `kwargs`
                    arguments to be passed to astropy.io.fits.getheader.
                    You should have at least one specifying the extension!
        """
        # init the logger
        self._set_logger(__name__)
        self.logger.info("Reading headers for metadata..")
        
        # default keywords
        if header_keys is None:
            header_keys = [
                'NMATCHES', 'MAGZP', 'MAGZPUNC', 'MAGZPRMS', 'CLRCOEFF', 'CLRCOUNC',
                'ZPCLRCOV', 'PCOLOR', 'SATURATE', 'ZPMED', 'ZPAVG', 'ZPRMSALL',
                'CLRMED', 'CLRAVG', 'CLRRMS', 'FIXAPERS', 'APCOR*', 'APCORUN*',
                'FIELDID', 'CCDID', 'QID', 'FILTERID', 'RCID', 'OBSMJD', 'EXPID', 'PROGRMID'
                ]
        magic_keys = [k.replace('*', '') for k in header_keys]
        
        # loop on files and fill in the dataframe with header keywords
        rows = []
        for fitsfile in tqdm.tqdm(self.files):
            try:
                head, row = fits.getheader(fitsfile, **getheader_args), {}
                for key, val in head.items():
                    if key in header_keys or any([mk in key for mk in magic_keys]):
                        self.logger.debug("found key %s in desired header keys."%key)
                        row[key] = val
                row['PATH'] = fitsfile
                rows.append(row)
            except OSError:
                self.logger.warning("skipping corrupted file %s"%fitsfile)
        self.df = pd.DataFrame.from_records(rows)

        # check that you have all the keys you asked for
        df_cols = self.df.columns.values.tolist()
        for mk in magic_keys:
            if not any([mk in key for key in df_cols]):
                self.logger.warning("couldn't find requested key: %s in file headers."%mk)
        
        # add obsid column as unique identifier of the data product
        self.df['OBSID'] = (
                    self.df['EXPID'].astype(str) + 
                    self.df['RCID'].astype(str) ).astype(int)
        
        if downcast:
            self.df = downcast_df(self.df)
        
#        self.df.set_index('OBSID', inplace = True, drop = False)
        self.logger.info("loaded meta data from fits headers for %d files into metadata dataframe."%len(self.df))

    


#    def get_paths(self, **querydf_args):
#        """
#            query the dataframe and reuturn a list of file path matching the query
#            
#            Parameters:
#            -----------
#                
#                dfquery_args: 
#                    pandas.DataFrame.query arguments.
#            
#            Returns:
#            --------
#                list of paths
#        """
#        return self.query_df ['PATH'].values

