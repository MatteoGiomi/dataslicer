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

from dataslicer.dataset_base import dataset_base
from dataslicer.metadata import metadata
from dataslicer.objtable import objtable


class dataset(dataset_base):
    """
        class to describe a set of fits files in terms of both the data table
        and the associated metadata.
    """
    
    def __init__(self, name, datadir, fext  =  ".fits", logger = None, **args):
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
        self.logger.info("found %d .%s files in directory: %s"%(len(self.files), self.fext, self.datadir))
        
        # load metadata
        self.load_metadata(**args)
        self._check_for_metadata()


    def load_metadata(self, metadata_file = None, **args):
        """
            load the metadata pertaining to this dataset.
        """
        print ("ajcna", self.logger)
        self.metadata = metadata(self.name, self.datadir, self.fext, self.logger)
        if metadata_file is None:
            metadata_file = os.path.join(self.datadir, self.name+"_metadata.csv")
        if os.path.isfile(metadata_file):
            self.logger.info("found metadata file: %s"%metadata_file)
            self.metadata.read_csv(fname = metadata_file, **args)
        else:
            self.metadata.load_header_meta(**args)
            self.metadata.to_csv(**args)


    def _check_for_metadata(self):
        """
            check if the object contains a valid metadata description.
        """
        if not hasattr(self, 'metadata'):
            raise AttributeError("this object has no metadata attribute. Run 'load_header_meta' and retry.")
        elif len(self.metadata.df) == 0:
            raise RuntimeError("metadata dataframe is empty.")


    def load_objtable(self, **args):
        """
            load the table data for this dataset eventually cut on metadata to select 
            only certains files to be loaded.
            
        """
        
        # eventually cut on metadata
        meta = self.metadata.query_df(**args)
        
        # load the data
        self.objtable = objtable(self.name, self.datadir, self.fext, self.logger)
        self.objtable.load_data(**args)
        
        # load the corresponding files



