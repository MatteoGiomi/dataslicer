#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# base class to collect files contained in a given directory.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os, glob, inspect, logging
import pandas as pd


def select_kwargs(func, **kwargs):
    """
        return the subset of kwargs that are accepted by the function func
    """
    return {k:v for k, v in kwargs.items() if k in inspect.getargspec(func).args}


class dataset_base():
    """
        base class to collect files contained in a given directory.
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
        
        # and man gave names
        if not os.path.isdir(datadir):
            raise OSError("directory: %s, does not exist."%datadir)
        self.name = name
        self.datadir = datadir
        self.fext = fext
        if not logger is None:
            self.logger = logger
        
        # go and find the files
        self.files  = [ os.path.abspath(f) for f in glob.glob(datadir+"/*"+fext) ]
        if len(self.files)  == 0:
            self.logger.warning("no %s files found in %s"%(fext, datadir))


    def _to_csv(self, tag = None, fname = None, **to_csv_args):
        """
            save the metadata dataframe as a csv table.
            
            Parameters:
            -----------
                
                tag: `str` or None
                    string to be appended to the dataset name qualifying the content
                    of the csv file (e.g. 'metadata', 'tabledata')
                
                fname: `str` or None
                    name of the csv file to which thi object will be stored. If 
                    None, it will be saved to self.datadit/self.name+"_"+tag+".csv"
                
                to_csv_args: `kwargs`
                    to be passed to pandas.DataFrame.to_csv method. The path_or_buff
                    kwargs is overwritten by the fname argument.
        """
        
        self._check_for_df()
        
        if not tag is None:
            tagname = "_%s"%(tag.strip())
        else:
            tag = ""
            tagname = ""
        
        if fname is None:
            fname = os.path.join(self.datadir, self.name+"%s.csv"%tagname)
        to_csv_args['path_or_buf'] = fname
        if hasattr(self, 'df'):
            self.logger.info("saving %s dataframe to csv file: %s"%(tag, fname))
            true_args = select_kwargs(pd.DataFrame.to_csv, **to_csv_args)
            self.df.to_csv(**true_args)
        else:
            raise AttributeError("this object has no df attribute: nothing to write.")


    def _read_csv(self, tag = None, fname = None, **read_csv_args):
        """
            load a csv table into this object metadata dataframe.
        
            Parameters:
            -----------
                
                tag: `str` or None
                    string to be appended to the dataset name qualifying the content
                    of the csv file (e.g. 'metadata', 'tabledata')
                
                fname: `str` or None
                    name of the csv file to be read. If None, it is assumed that 
                    this is self.datadit/self.name+"_"+tag+".csv"
                
                read_csv_args: `kwargs`
                    to be passed to pandas.DataFrame.read_csv method. The path_or_buff
                    kwargs is overwritten by the fname argument.
        """
        
        if not tag is None:
            tagname = "_%s"%(tag.strip())
        else:
            tag = ""
            tagname = ""
            
        if fname is None:
            fname = os.path.join(self.datadir, self.name+"%s.csv"%tagname)
        read_csv_args['filepath_or_buffer'] = fname
        self.logger.info("reading %s dataframe from csv file: %s"%(tag, fname))
        true_args = select_kwargs(pd.read_csv, **read_csv_args)
        self.df = pd.read_csv(**true_args)


    def _check_for_df(self):
        """
            check if the object contains a dataframe attribute
        """
        if not hasattr(self, 'df'):
            raise AttributeError("this object has no DataFrame.")


    def _set_logger(self, logger_name):
        """
            if not logger attribute is found for this object, 
            initialize one with the given name.
            
            Parameters:
            -----------
            
                logger_name: `str`
                    name of the logger
        """
        if not hasattr(self, 'logger'):
            logging.basicConfig(level = logging.INFO)
            self.logger = logging.getLogger(logger_name)
            self.logger.debug("set logger named %s."%(logger_name))


    def query_df(self, **dfquery_args):
        """
            query the dataframe. By defalt the query will be inplace, modifying
            the self.df attribute of this class. Ref:
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html
            
            Parameters:
            -----------
                
                dfquery_args: 
                    pandas.DataFrame.query arguments.
            
            Returns:
            --------
                pandas.DataFrame with query results.
        """
        self._check_for_df()
        if not 'inplace' in dfquery_args.keys():
            dfquery_args['inplace'] = True
        
        # select valid kwargs
        true_args = select_kwargs(pd.DataFrame.query, **dfquery_args)
        self.logger.info("quering dataframe with: %s"%true_args['expr'])
        qdf = self.df.query(**true_args)
        if dfquery_args['inplace']:
            return self.df
        else:
            return qdf
