#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# subclass pandas DataFrame to hold source tables. This class is meant both to
# offload some work from the dataslicer.objectable one, and to provide a more
# direct way to analyze fits file catalogs, without having to set up a dataset 
# object.
# 
# Author: M. Giomi (matteo.giomi@desy.de)

import time, tqdm
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level = logging.INFO)

from dataslicer.df_utils import match_to_PS1cal, fits_to_df

class srcdf(pd.DataFrame):
    """
        a dataframe with style. Eg, you can read in a fits file with:
        sdf = srcdf.read_fits(fitsfile, extension)
    """
    
    # dimension of a RC in pixels
    xsize, ysize = 3072, 3080
    
    @property
    def _constructor(self):
        return srcdf

    # TODO: port cluster_source methods from objtable
    
    
    @staticmethod
    def read_fits(fitsfile, extension, **kwargs):
        """
            read a fits file into a srcdf object and returns it.
            
            Parameters:
            -----------
                
                fitsfile: `str`
                valid path to the file you want to read.
                
                extension: `str` or `int`
                    extension to be read. This will be the second parameter
                    passed to astropy.io.fits.getdata
                
                kwargs: `kwargs`
                    other arguments to be passed directly to dataslicer.df_utils.fits_to_df
            
            Returns:
            -------
                srcdf.
        """
        return srcdf(fits_to_df(fitsfile, extension, **kwargs))
    

    def match_to_PS1Cal(self, rs_arcsec, ids = "sid", xname = 'ra', yname = 'dec', 
        clean_non_matches = True, logger = None, **kwargs):
        """
            for each source in this object, search for matching in the PS1Cal
            calibrator star catalog. This method is a wrapper around the 
            df_utils.math_to_PS1Cal function.
            
            Parameters:
            -----------
                
                rs_arcsec: `float`
                    search radius, in arcseconds, for the matching.
                
                ids: `str` or (str, ids) tuple
                    ID of the sources to makes it possible to attach, to each 
                    coordinate pair, the corresponding PS1cal.
                    If ids is string, the resulting datframe will have a column called
                    ids and the element in this column will be the index of the coordinate
                    pair in the input lists.
                    If ids is a (str, ids) tuple, the resulting dataframe will have a column
                    named ids[0] and will take the values given in ids[0].
                
                x[/y]name: `str`
                        name for table columns specifiyng the x,y (Equatorial, J2000)
                        coordinates to use.
                
                clean_non_matches: `bool`
                        if True, sources with no match in the PS1cal db are removed.
                
                logger: `logging.logger`
                    logger instance.
                
                **kwargs: `kwargs`
                    other arguments passed to df_utils.math_to_PS1Cal
        """
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # get a dataframe with the matches.
        logger.info("Matching %d sources to PS1Cal stars. Using %s and %s as coordinates."%
            (len(self), xname, yname))
        ps1cp_df = match_to_PS1cal(self[xname], self[yname], rs_arcsec, ids = ids, **kwargs)
        logger.info("Found PS1 calibrators for %d sources"%
            (len(ps1cp_df)))
        
        # merge the dataframe
        if type(ids) == str:
            idname = ids
        else:
            idname = ids[0]
        self = self.merge(
            ps1cp_df, on = idname, how='left',  suffixes=['', '_ps1'])
        
        # if requested, drop sources without PS1cp
        if clean_non_matches:
            self.dropna(subset = ['dist2ps1'], inplace = True)
            logger.info("dropping sources without match in PS1cal DB: %d retained."%
                (len(self.df)))

    def photometric_solution(self):
        """
            this is up to you Simeon! Here you implement your ZP and clr coeff fit.
            
            I've implemented the match_to_PS1 cal method, so you should have everything you need. 
            
            Keep in mind that depending on the filter ZTF is using, the PS1 colors can change, 
            in the ztf_pipeline_deliverables.pdf (pg 29) it is written:
            
            gPS1 − g = ZPg + cg (gPS1 − rPS1)
            rPS1 − r = ZPr + cr (gPS1 − rPS1)
            iPS1 −i = ZPi + ci(rPS1 −iPS1) 
            
            Remember to pass through all the parameters you need, and set reasonable defauls..
            
            You are free to decide weather this function should return the ZP and CLRCOEFF (and 
            errors), or add them to the dataframe as new columns, or both (depending on user choice).
            
            Enjoy, and feel free to ask as many questions as you want!
            
            matte
        """
        
        pass

    def calmag(self, mag_col, err_mag_col = None, calmag_col = None, zp_name = 'MAGZP', 
        clrcoeff_name = 'CLRCOEFF', zp_err = 'MAGZPUNC', clrcoeff_err = 'CLRCOUNC',
        ps1_color1 = None, ps1_color2 = None, dropmag = False, plot = True):
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
                
                zp_name/clrcoeff_name: `str`, `float`, or None
                    name (or value, if scalar) of ZP and color coefficient term. 
                    If clrcoeff_name is None, color correction will be ignored.
                
                zp_err/clrcoeff_err: `str`, `float`
                    name (or value, if scalar) of the error on the ZP and color coefficient.
                    If err_mag_col is not None, they must be given!
                
                ps1_color1[2]: `str` or array-like
                    If strings, these are the names of the PS1cal magnitudes used
                    to calibrate (they should be consistent with PCOLOR).
                    If array-like they should have the same length of the dataframe.
                
                dropmag: `bool`
                    if True the df column magname will be dropped.
        """
        raise NotImplementedError()
        
        
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
        
        check_col(needed_cols, self)
        for k in needed_cols:
            check_col(k, self)
        
        # name the cal mag column and the one for the error
        if calmag_col is None:
            calmag_col = "cal_"+mag_col
        err_calmag_col = "err_"+calmag_col
        
        # fill them
        if clrcoeff_name is None:
            self[calmag_col] = self[mag_col] + self[zp_name]
            
            if not err_mag_col is None:
                self[err_calmag_col] = np.sqrt(
                                    self[err_mag_col]**2. +
                                    self[zp_err]**2.)
        else:
            ps1_color = self[ps1_color1] - self[ps1_color2]
            self[calmag_col] = (
                self[mag_col] + 
                self[zp_name] +
                self[clrcoeff_name]*ps1_color)
            
            if not err_mag_col is None:
                d_ps1_color = np.sqrt( self['e_'+ps1_color1]**2. + self['e_'+ps1_color2]**2. )
                self[err_calmag_col] = np.sqrt(
                    self[err_mag_col]**2. +
                    self[zp_err]**2. +
                    (self[clrcoeff_err] * ps1_color)**2. + 
                    (self[clrcoeff_name] * d_ps1_color)**2)
        
        # eventually get rid of the uncalibrated stuff
        if dropmag:
            self.logger.info("dropping non calibrated magnitude %s from dataframe"%mag_col)
            self.drop(columns = [mag_col])
        
        if plot:
            fig, ax = plt.subplots()
            if err_mag_col is None:
                ax.hist(self[calmag_col], bins = 100)
                ax.set_xlabel("calibrated magnitude [mag]")
            else:
                ax.scatter(self[calmag_col], self[err_calmag_col])
                ax.set_xlabel("calibrated magnitude [mag]")
                ax.set_ylabel("error on calibrated magnitude [mag]")
            fig.tight_layout()
            self.save_fig(fig, '%s_calmag.png'%self.name)
            plt.close()


    def compute_camera_coord(self, rc_x_name, rc_y_name, cam_x_name = 'cam_xpos', 
        cam_y_name = 'cam_ypos', rotate = False, xgap_pix = 7, ygap_pix = 10, rcid_name = 'RCID', logger = None):
        """
            compute the camera-wide x/y coordinates of the sources. The x,y position
            start at the bottom-left corner of the camera (RC 14)
            
            Parameters:
            -----------
                
                rc_x[y]_name: `str`
                    name of dataframe column containg the position of the sources 
                    in pixel on the readout channel (RC)
                
                cam_x[y]_name: `str`
                    name of the columns that will contain the camera-wide coordinates.
                
                rotate: `bool`
                    if True, the individual readout channels will be rotated by 180 deg.
                
                x[y]gap_pix: `int`
                    size of gap between CCDs, in pixels.
                
                rcid_name: `str`
                    name of column containing the ID of the readout-channels (0 to 63).
                
                logger: `logging.logger`
                    logger instance.
        """
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # checks
        check_col([rc_x_name, rc_y_name, rcid_name], self)
        
        # compute ccd and quadrant (1 to 4) from RC
        ccd = (self[rcid_name]//4 + 1).rename('ccd')
        q = (self[rcid_name]%4 + 1).rename('q')

        # arrange the rc in rows and cols based on ccd and q.
        # NOTE: the returned values are zero-indexed (from 0 to 7) and 
        # start from the bottom-left corner of the image, so that RC 14 is
        # at position (0, 0) and RC 48 at (7, 7).
        yrc= 2*((ccd-1)//4) + 1*np.logical_or(q==1, q==2)
        xrc= 2*( 4-(ccd-1)%4)-1 - 1*np.logical_or(q==2, q==3)
        
        # now add the gaps between the ccds, and the rc size in pixels 
        # so that you have the x/y camera position of the lower-left corner of the RCs
        # of the readout channels
        xll = (xrc // 2)*xgap_pix + xrc*self.xsize
        yll = (yrc // 2)*ygap_pix + yrc*self.ysize
        
        # finally add the x/y position inside each RC
        if not rotate:
            self[cam_x_name] = xll + self[rc_x_name]
            self[cam_y_name] = yll + self[rc_y_name]
        else:
            self[cam_x_name] = xll - self[rc_x_name]
            self[cam_y_name] = yll - self[rc_y_name]
        self.logger.info("computed camera-wide coordinates of the sources as columns: %s %s of the dataframe"%
            (cam_x_name, cam_y_name))


    def trim_edges(self, x_name, y_name, trim_dist):
        """
            remove sources wich are close to the edge of a RC. The channels
            are 3070 x 3078 pixels. This will retain only the sources for which
            dist_x < x_name < 3070 - dist_x and dist_y < y_name < 3078 - dist_y
            
            Parameters:
            -----------
                
                x[y]_name: `str`
                    name of dataframe column containg the position of the sources 
                    in pixel on the readout channel (RC)
                
                trim_dist: `float` or array-like
                    minimum distance (in pixels) from the edge of the RC.
            
            Returns:
            --------
                
                pandas.DataFrame with the rejected sources.
        """
        check_col([x_name, y_name], self)
        
        if type(trim_dist) in [float, int]:
            dx = trim_dist
            dy = dx
        elif len(trim_dist) == 2:
            dx = trim_dist[0]
            dy = trim_dist[1]
        else:
            raise ValueError("parameter trim_dist must be float or [dist_x, dist_y].")

        query = "(@dx < %s < (%f - @dx)) and (@dy < %s < (%d - @dy))"%(
            x_name, srcdf.xsize, y_name, srcdf.ysize)
        self.query(expr = query, inplace = True)


    def tag_dust(self, df, dust_df_file, radius_multiply = 1., xname = 'xpos', yname = 'ypos',
        dust_df_query = None, remove_dust = False, logger = None):
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
                    
                    logger: `logging.logger`
                        logger instance.
                
                Returns:
                --------
                    
                    no_dust_df, dust_df: two dataframes containing the objects not contaminated
                    and contaminated by dust respectively.
            """
            
            if logger is None:
                logger = logging.getLogger(__name__)
            
            from shapely.geometry import Point
            from shapely import vectorized
            
            # read in the df file with the dust grains
            dust_df = pd.read_csv(dust_df_file)
            if not dust_df_query is None:
                dust_df.query(dust_df_query, inplace = True)
            logger.info("read position of %d dust grains from file: %s"%(len(dust_df), dust_df_file))
            
            # sort the dust dataframe by dust size, smaller first. This way, in case of
            # close-by dust grains, the source is assigned to the largest one.
            dust_df = dust_df.sort_values('r')
            
            # loop on the dust grains and match them to the sources
            logger.info("matching sources %d objects with %d dust grains.."%(len(df), len(dust_df)))
            for _, d in dust_df.iterrows():
                
                # create the geometrical object for this dust grain
                dust_g = Point(d['x'], d['y']).buffer( (radius_multiply * d['r']) )
                
                # find out all the sources that matches this dust grain
                flagin = vectorized.contains(dust_g, self[xname], self[yname])
                logger.debug("found %d sources inside a dust grain of radius: %.2f"%(sum(flagin), d['r']))
                
                # add the dust property to the dataframe
                self.loc[flagin, 'dust_x'] = d['x']
                self.loc[flagin, 'dust_y'] = d['y']
                self.loc[flagin, 'dust_r'] = d['r']
            
            dust_matches = self.count()['dust_r']
            logger.info("found %d sources on top of dust grains."%(dust_matches))
            
            # split the dataframe into dust and non-dust, eventually remove 
            # the dust from the source dataframe
            dusty = srcdf(self.dropna(axis=0, subset = ['dust_r']))
            clean = srcdf(self[self['dust_r'].isna()])
            if remove_dust:
                self = clean
                logger.info("removing sources affected by dust. %d retained."%(len(self)))
            return clean, dusty

