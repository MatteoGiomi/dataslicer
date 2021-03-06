#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# from the data contained in a given directory, create and manage both
# the table data and the metadata of the files.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import tqdm, time, os
import pandas as pd
import numpy as np
from astropy.io import fits

from dataslicer.dataset_base import dataset_base, select_kwargs
from dataslicer.metadata import metadata
from dataslicer.objtable import objtable
from astropy.io import fits
from dataslicer.df_utils import strlist_in_strlist

class dataset(dataset_base):
    """
        class to describe a set of fits files in terms of both the data table
        and the associated metadata.
    """
    
    def __init__(self, name, datadir, fext  =  ".fits", logger = None):
        """
            Parameters
            ----------
                
                name: `str`
                    name of the dataset.
                
                datadir: `str`
                    path of directory where the data is
                
                fext: `str`
                    extension of the data files.
                
                logger: `logger.Logger`:
                    logger for the class. If None, a default one will be created.
        """
        
        # init the logger
        self._set_logger(__name__)
        if not logger is None:
            self.logger = logger
        
        # init parent class
        dataset_base.__init__(self, name, datadir, fext, logger = self.logger)
        self.logger.info("found %d %s files in directory: %s"%(len(self.files), self.fext, self.datadir))


    def load_metadata(self, metadata_file = None, force_reload = False, meta_tocsv = True, **args):
        """
            load the metadata pertaining to this dataset. If a csv file named metadata_file
            is found in the data directory, the metadata are read from that file if
            the columns names saved to csv are the same aas tose requested.
            
            Parameters:
            -----------
            
                metadata_file: `str`
                    path to a csv file containing the metadata for the dataset you
                    want to read.
                
                meta_tocsv: `bool`
                    if True, the metadata dataframe will be saved in self.datadir.
                
                force_reload: `bool`
                    if True, metadata will be reloaded from the fits files, even
                    if a metadata_file is found. This file will be overwritten.
        """
        
        self.metadata = metadata(self.name, self.datadir, self.fext, self.logger)
        if metadata_file is None:
            metadata_file = os.path.join(self.datadir, self.name+"_metadata.csv.gz")
        
        # check for the file and if the df all has the columns you requested, else reload it.
        read_from_fits = True
        if os.path.isfile(metadata_file) and (not force_reload):
            self.logger.info("found metadata file: %s"%metadata_file)
            self.metadata.read_csv(fname = metadata_file, **args)
            read_from_fits = False
            hk = args.get('header_keys', None)
#            if ( hk is not None and not set(hk).issubset(set(self.metadata.df.columns.values)) ):
            if (hk is not None) and (not strlist_in_strlist(hk, self.metadata.df.columns.values)):
                self.logger.info("requested columns are different from those found in file. Reloading it.")
                read_from_fits = True
        if read_from_fits:
            self.metadata.load_header_meta(**args)
            if meta_tocsv:
                self.metadata.to_csv(fname=metadata_file, **args)


    def load_objtable(self, objtable_file = None, force_reload = False, obj_tocsv = True, **args):
        """
            load the table data for this dataset eventually cut on metadata to select 
            only certains files to be loaded.
            
            Parameters:
            -----------
                
                objtable_file: `str` or None
                    path to a csv file containing an objtable dataframe you want to
                    load. If None, a default file named self.name+_objtable.csv is
                    searched for in self.datadir
                
                force_reload: `bool`
                    if True, objtable will be reloaded from the fits files, even
                    if a metadata_file is found. This file will then be overwritten.
                
                obj_tocsv: `bool`
                    if True, the objtable dataframe will be saved in self.datadir.
                
                args: dataset_base.query_df and/or objtable.load_data and/or objtable.to_csv args. 
                    If expr = str is provided, the string will be used to query 
                    the metadata and load just the selected objects.
        """
        self.objtable = objtable(self.name, self.datadir, self.fext, self.logger)
        if objtable_file is None:
            objtable_file = os.path.join(self.datadir, self.name+"_objtable.csv.gz")
        
        # check for the file and if the df all has the columns you requested, else reload it.
        read_from_fits = True
        if os.path.isfile(objtable_file) and (not force_reload):
            self.logger.info("found objtable file: %s"%objtable_file)
            self.objtable.read_csv(fname = objtable_file, **args)
            read_from_fits = False
            
            # check the columns (in case you want all of them, check what are those)
            req_cols = args.get('select_columns', None)
            if req_cols == 'all':
                head0 = fits.getheader(self.files[0], args['extension'])
                req_cols = [cname for key, cname in head0.items() if 'TTYPE' in key]
#            if ( req_cols is not None and not set(req_cols).issubset(set(self.objtable.df.columns.values)) ):
            if ( req_cols is not None) and (not strlist_in_strlist(req_cols, self.objtable.df.columns.values)):
                self.logger.info("requested columns are different from those found in file. Reloading it.")
                read_from_fits = True
        
        if read_from_fits:
            # eventually cut on metadata, or got them all.
            if 'expr' in args.keys():
                meta = self.metadata.query_df(**args)
            else:
                meta = self.metadata.df
            self.objtable.load_data(target_metadata_df = meta, **args)
            if obj_tocsv:
                self.objtable.to_csv(**args)
    
    def load(self, objtable_ext, metadata_ext, **args):
        """
            load metadata and objtable for this dataset.
            
            Parameters:
            -----------
                
                objtable_ext: `str` or `int`
                    extension ID of the table data in the fits files
                
                metadata_ext: `str` or `int`
                    extension ID for the header fits file containing the metadata.
                    This will be passed to astropy.io.fits.getheader as par ext
        """
        args['ext'] = metadata_ext
        self.load_metadata(**args)
        args['extension'] = objtable_ext
        self.load_objtable(**args)
        self._check_for_metadata()
        self._check_for_objtable()


    def _check_for_metadata(self):
        """
            check if the object contains a valid metadata description.
        """
        if not hasattr(self, 'metadata'):
            raise AttributeError("this object has no metadata attribute. Run 'load_header_meta' and retry.")
        elif len(self.metadata.df) == 0:
            raise RuntimeError("metadata dataframe is empty.")


    def _check_for_objtable(self):
        """
            check if the object contains a valid metadata description.
        """
        if not hasattr(self, 'objtable'):
            raise AttributeError("this object has no metadata attribute. Run 'load_header_meta' and retry.")
        elif len(self.objtable.df) == 0:
            raise RuntimeError("metadata dataframe is empty.")


    def set_plot_dir(self, plot_dir):
        """
            set the path of the directory used to store diagnostic plots for 
            both metdata and 
        """
        self.logger.info("diagnostic plots will be saved to %s"%(plot_dir))
        self._set_plot_dir(plot_dir)
        if hasattr(self, 'metatdata'):
            self.metadata._set_plot_dir(plot_dir)
        if hasattr(self, 'objtable'):
            self.objtable._set_plot_dir(plot_dir)


    def merge_metadata_to_sources(self, metadata_cols = None, join_on = 'OBSID'):
        """
            join metadata dataframe with objtable so that information such as
            the zero point and obsjd are attached to each source. NOTE that this 
            will modify the objtable dataframe.
            
            Parameters:
            -----------
            
                metadata_cols: list or None
                    list metadata column names to be added to the objtable dataframe.
                    if None, the entire metadata dataframe will be joined.
                
                join_on: `str`
                    name of the column used to join the metadata and source data.
        """
        
        # checks
        self._check_for_metadata()
        self._check_for_objtable()
        if not (join_on in self.objtable.df.columns):
            raise KeyError("Merge on column %s not present in objtable dataframe"%join_on)
        if not (join_on in self.metadata.df.columns):
            raise KeyError("Merge on column %s not present in metadata dataframe"%join_on)
        
        # skimm out metadata columns
        if (not metadata_cols is None) and (join_on not in metadata_cols):
            metadata_cols.append(join_on)
        if metadata_cols is None:
            meta_2_join = self.metadata.df
        else:
            for c in metadata_cols:
                if c not in self.metadata.df.columns.tolist():
                    self.logger.warning("Column %s not present in metadata df."%c)
            meta_2_join = self.metadata.df.ix[:, metadata_cols]
        
        # join and go
        self.objtable.df = pd.merge(
                self.objtable.df, meta_2_join, on = join_on)
        self.logger.info("Merged metadata with objtable sources. Columns used: %s"%
            (", ".join(meta_2_join.columns.values)))


    def select_objects(self, query):
        """
            discard objects from objtable if they don't satisfy the query
            
            Parameters:
            -----------
            
                query: `str`
                    query expression, see 
                    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html 
                    for syntax explanation.
        """
        self.objtable.query_df(expr = query)
        
