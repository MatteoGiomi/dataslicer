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
    return {k:v for k, v in kwargs.items() if k in inspect.getfullargspec(func).args}


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


    def _set_plot_dir(self, plot_dir):
        """
            set the path of the directory used to store diagnostic plots.
            
            Parameters:
            -----------
                
                plot_dir: `str`
                    path of the directory where diag plots will be saved. 
                    It will be created if not existsing.
        """
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        self.plot_dir = plot_dir


    def save_fig(self, fig, name, **savefig_kwargs):
        """
            save a plot to plotdir/name If plotdir is not defined, save to
            current directory.
            
            Parameters:
                
                fig: `matplotlib.figure.Figure`
                    figure to be saved.
                
                name: `str`
                    name of the plot.
                
                savefig_kwargs: `kwargs`
                    passed to matplotlib.figure.Figure.savefig
        """
        
        if not hasattr(self, 'plot_dir'):
            logging.warning("plot directory not set for this object. Saving figure to current dir.")
            pltdir = './'
        else:
            pltdir = self.plot_dir
        filename = os.path.join(pltdir, name)
        self.logger.info("saving plot to %s"%filename)
        fig.savefig(filename, **savefig_kwargs)
        
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
            fname = os.path.join(self.datadir, self.name+"%s.csv.gz"%tagname)
        to_csv_args['path_or_buf'] = fname
        if to_csv_args.get('index', None) is None:
            to_csv_args['index'] = False
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
            fname = os.path.join(self.datadir, self.name+"%s.csv.gz"%tagname)
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
        
        # add default
        if not 'inplace' in dfquery_args.keys():
            dfquery_args['inplace'] = True
        
        # select valid kwargs
        true_args = select_kwargs(pd.DataFrame.query, **dfquery_args)
        qdf = self.df.query(**true_args)
        if dfquery_args['inplace']:
            self.logger.info("quering dataframe with: %s. %d rows survived"%
                (true_args['expr'], len(self.df)))
            return self.df
        else:
            self.logger.info("quering dataframe with: %s. %d rows survived"%
                (true_args['expr'], len(qdf)))
            return qdf
