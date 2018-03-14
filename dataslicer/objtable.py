#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# class to collect and manage the tablular data of a given dataset into a dataframe.
# 
# Author: M. Giomi (matteo.giomi@desy.de)

import tqdm, time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from astropy.io import fits

from extcats import CatalogQuery
from dataslicer.dataset_base import dataset_base, select_kwargs


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


class objtable(dataset_base):
    """
        class to collect and manage the tablular data of 
        a given dataset into a dataframe.
    """


    def to_csv(self, **args):
        """
        """
        self._to_csv(tag = 'tabledata', **args)


    def read_csv(self, **args):
        """
        """
        self._read_csv(tag = 'tabledata', **args)


    def load_data(self, target_metadata_df = None, add_obs_id = True, **fits_to_df_args):
        """
            load the data contained into the files and corresponding to the given
            extension. Eventually apply a cut on the metadata table to select the 
            files you want.
            
            Parameters:
            -----------
                
                target_metadata_df: `pandas.DataFrame` or None
                    dataframe with the metadata for the files you want to load. 
                    If None, all the files in this object self.files attribute will be used.
                
                add_obs_id: `bool`
                    if True, a column identifying each data product is added to 
                    the dataframe, so that you can later join the data with the metadata.
                    This obs_id is made appending the RC ID (0..63) to the exposure ID.
                
                fits_to_pd_args: `kwargs`
                    arguments to be passed to the fits_to_df function that will
                    parse the fits file into a pandas dataframe. At least extension
                    should be specified. Another useful arguments is 'columns', 
                    allowing you to select which columns of the fits you want to read.
        """
        
        # init the logger
        self._set_logger(__name__)
        self.logger.info("loading files into object table..")
        
        # if you want, select based on metadata
        if (not target_metadata_df is None):
            self.logger.info("using target metadata to indetify the files.")
            files = target_metadata_df['PATH'].values
        else:
            files = self.files
        if len(files) == 0:
            raise RuntimeError("found no file to load.")
        
        # create the big dataframe
        true_args = select_kwargs(fits_to_df, **fits_to_df_args)
        start = time.time()
        if (not add_obs_id) or (target_metadata_df is None):
            frames  = [fits_to_df(ff, **true_args) for ff in tqdm.tqdm(files)]
        else:
            frames = []
            for ff in tqdm.tqdm(files):
                buff = fits_to_df(ff, **true_args)
                obsid = target_metadata_df[target_metadata_df['PATH'] == ff]['OBSID']#.iloc[0]
                buff['OBSID'] = obsid
                frames.append(buff)
        self.df = pd.concat(frames)
        end = time.time()
        self.logger.info("loaded %d files into a %d rows dataframe. Took %.2e sec"%
            (len(files), len(self.df), (end-start)))


    def cluster_sources(self, cluster_size_arcsec = 3, min_samples = 0,
                xname = 'ALPHAWIN_J2000', yname = 'DELTAWIN_J2000', purge_df = False):
        """
            cluster the data in this object dataframe depending on 
            their position. It uses the DBSCAN algorithm to get the job done.
            Builds a grouped version of the dataframe with the cluster ID as
            grouping key.
            
            Parameters:
            -----------
                
                cluster_size_arcsec: `float`
                    The maximum distance between two points for them to be 
                    considered as in the same neighborhood (esp parameter of sklearn.DBSCAN)
            
                min_samples: `int`
                    minimum number of points in a cluster for it to be retained. 
                    If 0, no restriction if applied.
                
                x[/y]name: `str`
                    name of df columns specifiyng the x,y coordinates to use.
                
                purge_df: `bool`
                    if True, points which are not in a retained cluster (with more than
                    min_samples) are removed from the dataframe.
        """
        
        # checking, always a nice feeling
        self._check_for_df()
        
        # spatial clustering of the points on a sphere
        self.logger.info("running DBSCAN to cluster %d sources into individual objects"%len(self.df))
        self.logger.info("using %s, %s as coordinates and a radius of %.2f arcsec"%(
            xname, yname, cluster_size_arcsec))
        coords = np.array([self.df[xname], self.df[yname]]).T
        db = DBSCAN(
            eps = np.radians(cluster_size_arcsec / 3600.),
            min_samples= min_samples,
            algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        
        # tag each source and remove noisy clusters (that is, with size smaller than minoccur)
        self.df['clusterID'] = db.labels_
        clean_df = self.df[self.df.clusterID  != -1]
        if purge_df:
            self.df = clean_df
        
        # group the dataframe
        self.gdf = clean_df.groupby('clusterID')
        self.logger.info(
            "found %d clusters with maximum size of %.2f arcsec and minimum number of entries: %d"%
            (len(self.gdf), cluster_size_arcsec, min_samples))


    def compute_cluster_centroid(self, xname = 'ALPHAWIN_J2000', yname = 'DELTAWIN_J2000', 
        wav = False, xerr = None, yerr = None):
        """
            compute the average position of each cluster centroid. 
            
            Parameters:
            -----------
                
                x[/y]name: `str`
                    name of df columns specifiyng the x,y coordinates to use.
                
                wav: `bool`
                    if True, the weighted average is computed, weighting each coordinate
                    with (1/err)**2. The names for the error columns have to be provided. 
                
                x[/y]err: `str`
                    name of df columns specifiyng the uncertainties x,y coordinates to use.
            
            Returns:
            --------
                
                average ra, average dec of each group as array.
        """
        
        if wav is True:
            raise NotImplementedError("cluster centroid with weighted average not implemented yet.")
        
        if not hasattr(self, 'gdf'):
            raise AttributeError("this object has no grouped dataframe.")
            
        return (self.gdf[xname].mean(), self.gdf[yname].mean())


    def match_to_PS1cal(self, min_samples = 0, 
            xname = 'ALPHAWIN_J2000', yname = 'DELTAWIN_J2000', sep = 3,
            returnobjs = False, dbclient = None):
            """
                split the data in this object table matching them to the PS1 calibrator stars. 
                
                Use the extcats package to do the matching. For this, the PS1 calibrators
                have to be arranged in a mongo database.
                
                Parameters:
                -----------
                    minoccur: [int], minimum number of times an object has to be present in the
                             data in order to be returned. If None, no restriction if applied.
                             
                    sep: [quantity, deg], maximum allowed radius for an object to be associated
                         with a PS1 calibrator.
                         
                    x[/y]name: [str], name for table columns specifiyng the x,y coordinates to use.
                    
                    returnobjs: [bool], if True, returns as a first argument the list of 
                               dataslice.source objects, else the data table, grouped by PS1 calibrator.
                               
                    skycoords_kwargs : [dict], kwargs for astropy.coordinates.SkyCoord.
                    
                    PS1skycoords_kwargs : [dict], kwargs for astropy.coordinates.SkyCoord when 
                                            reading the PS1 calibrator tables.
                Returns:
                --------
                objs, list of source objects found in the data that are assigned to PS1 stars.
                ps1cals, table with all te PS1 calibrators pertaining to the dataset.
            """
            
            # initialize the catalog query object
            ps1cal_query = CatalogQuery.CatalogQuery(
                'ps1cal', 'ra', 'dec', dbclient = dbclient, logger = self.logger)
            
            use_clusters = True
            if use_clusters:
                self.logger.info("Matching cluster centroids with the PS1 calibrators")
                av_x, av_y = self.compute_cluster_centroid(xname, yname)
                for icl in tqdm.tqdm(range(len(self.gdf))):
                    ps1match =  ps1cal_query.findclosest(ra = av_x[icl], dec = av_y[icl], 
                        rs_arcsec = sep, method = 'healpix')
                    if ps1match != (None, None):
                        pass
                        
                    
                    
            
#            logging.info("matching %d table entries to PS1 calibrator stars"%len(self.tab))
#            logging.info("using %s, %s as coordinates and a search radius of %.2f arcsec"%(
#                xname, yname, sep.to('arcsec').value))
            
            
            
            # do the matching source by source
#            logging.info("matching objects with the PS1 calibrators")
#            for isrc in tqdm.tqdm(range(len(self.tab))):
#                ps1match = ps1cal_query.findclosest(
#                    ra = self.tab[xname][isrc], dec = self.tab[yname][isrc], 
#                    rs_arcsec = sep.to('arcsec').value, method = 'healpix')
