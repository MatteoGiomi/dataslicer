#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# class to collect and manage the tablular data of a given dataset into a dataframe.
# 
# Author: M. Giomi (matteo.giomi@desy.de)

import time, tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

import dataslicer.df_utils as df_utils
from dataslicer.dataset_base import dataset_base, select_kwargs


# this class is getting huge, should try somethig like:
# http://www.qtrac.eu/pyclassmulti.html
#
# OR, EVEN BETTER:
# https://groups.google.com/forum/?hl=en#!topic/comp.lang.python/goLBrqcozNY
# sooner or later.

# here the hefty pieces of codes are kept
from dataslicer._objtable_methods import _objtable_methods

class objtable(dataset_base, _objtable_methods):
    """
        class to collect and manage the tablular data of 
        a given dataset into a dataframe.
    """

    def to_csv(self, **args):
        self._to_csv(tag = 'tabledata', **args)


    def read_csv(self, **args):
        self._read_csv(tag = 'tabledata', **args)


    def _check_for_gdf(self):
        if not hasattr(self, 'gdf'):
            raise AttributeError("this object has no grouped dataframe.")


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
        true_args = select_kwargs(df_utils.fits_to_df, **fits_to_df_args)
        start = time.time()
        if target_metadata_df is None:
            frames  = [df_utils.fits_to_df(ff, **true_args) for ff in tqdm.tqdm(files)]
        else:
            frames = []
            for ff in tqdm.tqdm(files):
                buff = df_utils.fits_to_df(ff, **true_args)
                if add_obs_id:
                    obsid = target_metadata_df[
                        target_metadata_df['PATH'] == ff]['OBSID'].values[0]
                    buff['OBSID'] = pd.Series( [obsid]*len(buff), dtype = int)
                frames.append(buff)
        self.df = pd.concat(frames)
        end = time.time()
        self.logger.info("loaded %d files into a %d rows dataframe. Took %.2e sec"%
            (len(files), len(self.df), (end-start)))


    def cluster_sources(self, cluster_size_arcsec, min_samples, xname, yname, n_jobs = 2, purge_df = False):
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
                
                n_jobs: `int`
                    The number of parallel jobs to run. If -1 use all CPUs.
                
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
            algorithm='ball_tree', metric='haversine', n_jobs = n_jobs).fit(np.radians(coords))
        
        # tag each source and remove noisy clusters (that is, with size smaller than minoccur)
        self.df['clusterID'] = db.labels_
        clean_df = self.df[self.df.clusterID  != -1]
        if purge_df:
            self.df = clean_df
        
        # group the dataframe
        self.gdf = clean_df.groupby('clusterID', sort = False)
        self.logger.info(
            "found %d clusters with maximum size of %.2f arcsec and minimum number of entries: %d"%
            (len(self.gdf), cluster_size_arcsec, min_samples))


    def compute_cluster_centroid(self, xname, yname, 
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
                
                average_ra, average_dec of each group as array.
        """
        
        if wav is True:
            raise NotImplementedError("cluster centroid with weighted average not implemented yet.")
        self._check_for_gdf()
        return (self.gdf[xname].mean(), self.gdf[yname].mean())


    def calmag(self, mag_col, err_mag_col, calmag_col = None, zp_name = 'MAGZP', 
        clrcoeff_name = 'CLRCOEFF', zp_err = 'MAGZPUNC', clrcoeff_err = 'CLRCOUNC',
        ps1_color1 = None, ps1_color2 = None, dropmag = False):
        """
            apply photometric calibration to magnitude. The formula used is
            
            
            Mcal = Minst + ZP_f + c_f*(M1_PS1 -M2_PS1)
            
            where Minst is the instrumental magnitude, ZP_f , c_f are the image-wise
            zero point and color coefficient (MAGZP,CLRCOEFF), and (M1_PS1-M2_PS1)
            is the color of the source in the PS1 system. The filter used are defined
            by the header key PCOLOR.
            
            IMPORTANT: the above formula can be applied ONLY to PSF-fit catalogs.
            
            Parameters:
            -----------
            
                mag_col: `str`
                    name of the magnitude column you want to calibrate.
                
                err_mag_col: `str` or None
                    name of the columns containing the error on mag_col. If None, 
                    error on the calibrated magnitudes will not be computed. 
                
                calmag_col/err_mag_col: `str` or None
                    name of the column in the dataframe which will host the
                    calibrated magnitude value and its error. If None the name will 
                    be cal_+magname and err_cal_+magname
                
                zp_name/clrcoeff_name: `str` or None
                    name of ZP and color coefficient term. If clrcoeff_name is None, 
                    color correction will be ignored.
                
                zp_err/clrcoeff_err: `str`
                    name of columns containing the error on the ZP and color coefficient.
                
                ps1_color1[2]: `str` or array-like
                    If strings, these are the names of the PS1cal magnitudes used
                    to calibrate (they should be consistent with PCOLOR).
                    If array-like they should have the same length of the dataframe.
                
                dropmag: `bool`
                    if True the df column magname will be dropped.
                    
        """
        
        self.logger.info("Applying photometric calibration.")
        
        # see what columns are needed
        needed_cols = [mag_col, zp_name]
        if clrcoeff_name is None:
            self.logger.warning("color correction will not be applied")
        else:
            needed_cols.extend([clrcoeff_name])
            if type(ps1_color1) == str and type(ps1_color2) == str:
                needed_cols.extend([ps1_color1, ps1_color2])
        if not err_mag_col is None:
            needed_cols.extend([zp_err, clrcoeff_err, err_mag_col])
        
        df_utils.check_col(needed_cols, self.df)
        for k in needed_cols:
            df_utils.check_col(k, self.df)
        
        # name the cal mag column and the one for the error
        if calmag_col is None:
            calmag_col = "cal_"+mag_col
        err_calmag_col = "err_"+calmag_col
        
        # fill them
        if clrcoeff_name is None:
            self.df[calmag_col] = self.df[mag_col] + self.df[zp_name]
            if not err_mag_col is None:
                self.df[err_calmag_col] = np.sqrt(
                                    self.df[err_mag_col]**2. +
                                    self.df[zp_err]**2.)
        else:
            self.df[calmag_col] = (
                self.df[mag_col] + 
                self.df[zp_name] +
                self.df[clrcoeff_name]*(self.df[ps1_color1] - self.df[ps1_color2])
                                   )
            if not err_mag_col is None:
                self.df[err_calmag_col] = np.sqrt(
                    self.df[err_mag_col]**2. +
                    self.df[zp_err]**2. +
#                    (self.df[clrcoeff_err] * (self.df[ps1_color1] - self.df[ps1_color2]))**2. + 
                    (self.df[clrcoeff_name]**2) * (self.df['e_'+ps1_color1]**2. + self.df['e_'+ps1_color2]**2.)
                                                  )
        
        # eventually get rid of the uncalibrated stuff
        if dropmag:
            logging.info("dropping non calibrated magnitude %s from dataframe"%mag_col)
            self.df.drop(columns = [mag_col])
        
        # TODO: diagnostic plot with the pool distribution


    def cluster_op(self, col, function):
        """
            apply a function to each cluster group in this dataframe and 
            return a dataframe with the results.
            
            Parameters:
            -----------
            
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
        self._check_for_gdf
        df_utils.check_col(col, self.gdf)
        
        # see if it's in df_utils
        if type(function) == str:
            try:
                func = getattr(df_utils, function)
                self.logger.info("using function %s from df_utils module"%function)
            except AttributeError:
                self.logger.info("using user defined function %s"%function.__name__)
        else:
            func = function
            
        # apply and return
        return self.gdf[col].apply(func).unstack()


    def compute_camera_coord(self, rc_x_name, rc_y_name, cam_x_name = 'cam_xpos', 
        cam_y_name = 'cam_ypos', xgap_pix = 7, ygap_pix = 10, rcid_name = 'RCID'):
        """
            compute the camera-wide x/y coordinates of the sources. The x,y position
            start at the bottom-left corner of the camera (RC 14)
            
            Parameters:
                
                rc_x[y]_name: `str`
                    name of dataframe column containg the position of the sources 
                    in pixel on the readout channel (RC)
                
                cam_x[y]_name: `str`
                    name of the columns that will contain the camera-wide coordinates.
                
                x[y]gap_pix: `int`
                    size of gap between CCDs, in pixels.
                
                rcid_name: `str`
                    name of column containing the ID of the readout-channels (0 to 63).
        """
        
        # dimension of a RC in pixels
        xsize, ysize = 3080, 3072
        
        # checks
        df_utils.check_col([rc_x_name, rc_y_name, rcid_name], self.df)
        
        # compute ccd and quadrant (1 to 4) from RC
        ccd = (self.df[rcid_name]//4 + 1).rename('ccd')
        q = (self.df[rcid_name]%4 + 1).rename('q')

        # arrange the rc in rows and cols based on ccd and q.
        # NOTE: the returned values are zero-indexed (from 0 to 7) and 
        # start from the bottom-left corner of the image, so that RC 14 is
        # at position (0, 0) and RC 48 at (7, 7).
        yrc= 2*((ccd-1)//4) + 1*np.logical_or(q==1, q==2)
        xrc= 2*( 4-(ccd-1)%4)-1 - 1*np.logical_or(q==2, q==3)
        
        # now add the gaps between the ccds, and the rc size in pixels 
        # so that you have the x/y camera position of the lower-left corner of the RCs
        # of the readout channels
        xll = (xrc // 2)*xgap_pix + xrc*xsize
        yll = (yrc // 2)*ygap_pix + yrc*ysize
        
        # finally add the x/y position inside each RC
        self.df[cam_x_name] = xll + self.df[rc_x_name]
        self.df[cam_y_name] = yll + self.df[rc_y_name]
        self.logger.info("computed camera-wide coordinates of the sources as columns: %s %s of the dataframe"%
            (cam_x_name, cam_y_name))
        
        # TODO: rotation?
        
        # eventually update the grouped dataframe
        if hasattr(self, 'gdf'):
            self.logger.info("updating grouped dataframe")
            self.gdf = self.df.groupby('clusterID', sort = False)
