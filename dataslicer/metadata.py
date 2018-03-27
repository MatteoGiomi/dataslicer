#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# class to collect and manage the metadata (airmass, seeing, ccd temp, time, 
# zero point, ecc..) of a given dataset into a dataframe
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os, tqdm
from astropy.io import fits
import pandas as pd

from dataslicer.dataset_base import dataset_base


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


    def load_header_meta(self, header_keys = None, **getheader_args):
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
            head, row = fits.getheader(fitsfile, **getheader_args), {}
            for key, val in head.items():
                if key in header_keys or any([mk in key for mk in magic_keys]):
                    self.logger.debug("found key %s in desired header keys."%key)
                    row[key] = val
            row['PATH'] = fitsfile
            rows.append(row)
        self.df = pd.DataFrame.from_records(rows)
        
        # add obsid column as unique identifier of the data product
        self.df['OBSID'] = (
                    self.df['EXPID'].astype(str) + 
                    self.df['RCID'].astype(str) ).astype(int)
#        self.df.set_index('OBSID', inplace = True, drop = False)
        self.logger.info("loaded meta data from fits headers for %d files into metadata dataframe."%len(self.df))


    def load_IRSA_meta(self):
        """
            retrieve metadata for each file from the IRSA archieve. Uses 
            queryIRSA to download and collect metadata.
        """
        self._check_for_meta()
        raise NotImplementedError("still under development")


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

