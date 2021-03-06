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
import matplotlib.pyplot as plt
import logging
from scipy.odr import ODR, Model, Data, RealData
logging.basicConfig(level = logging.INFO)

from dataslicer.df_utils import fits_to_df, check_col
from dataslicer.PS1Cal_matching import match_to_PS1cal
from dataslicer.metadata import load_IRSA_meta

class srcdf(pd.DataFrame):
    """
        a dataframe with style. Eg, you can read in a fits file with:
        sdf = srcdf.read_fits(fitsfile, extension)
    """

    # dimension of a RC in pixels
    xsize, ysize = 3072, 3080

    # a dataframe proper has no logger, so this can't be an attribute
    _class_logger = logging.getLogger(__name__)

    @property
    def _constructor(self):
        return srcdf

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


    def add(self, other_df, reindex = True, srcid_key = 'sourceid', inplace = True, logger = None):
        """
            add another dataframe of source dataframe to this object.

            Parameters:
            -----------

                other_df: `pandas.DataFrame` or `dataslicer.srcdf.srcdf`
                    dataframe to add.

                reindex: `bool`
                    weather or not the sourceid key has to be recomputed for the
                    new dataframe.

                srcid_key: `str`
                    key with the soucre identified that has to be re-indexed.

                inplace: `bool`
                    if True, the second df will be added to this object and None
                    will be returned. If False, return a new dataframe.

                logger: `logging.logger`
                    logger instance.

                Returns:
                --------

                    srcdf or None, depending on the value of inplace

        """

        if logger is None:
            logger = srcdf._class_logger

        merged = srcdf(pd.concat([self, other_df], ignore_index = True))
        if reindex:
            merged.reindex_sources(srcid_key, logger = logger)

        if inplace:
            self = merged
        else:
            return merged


    def add_IRSA_meta(self, IRSA_meta_cols = ['airmass'], expid_col = 'EXPID', logger = None):
        """
            wrapper aound dataslicer.metadata.load_IRSA_meta
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
            logger = srcdf._class_logger
        self = load_IRSA_meta(self, IRSA_meta_cols = IRSA_meta_cols, expid_col = expid_col, logger = logger)
        return self

    def reindex_sources(self, srcid_key = 'sourceid', logger = None):
        """
            Assign to the srcid column the values of the pandas dataframe index.
            In this way, the sources remains uniquely identified even if multiple
            catalogs are loaded into the same object.

            Parameters:
            -----------

                srcid_key: `str`
                    name of the column containing the sourceid that you need to re-index.

                logger: `logging.logger`
                    logger instance.
        """

        if logger is None:
            logger = srcdf._class_logger
        logger.info("re-indexing %s column using the DataFrame index values."%(srcid_key))
        self[srcid_key] = self.index.values


    def match_to_PS1Cal(self, rs_arcsec, ids = 'sourceid', xname = 'ra', yname = 'dec',
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
            logger = srcdf._class_logger

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
                (len(self)))
        return self

    def photometric_solution(self, ztf_filter = 'g', mag_col = 'mag', mag_err_col = 'sigmag',
        gmag_col = 'gmag', rmag_col = 'rmag', imag_col = 'imag',
        gmag_err_col = 'e_gmag', rmag_err_col = 'e_rmag', imag_err_col = 'e_imag',
        logger = None, return_values = True, append_columns = True, plot = False, plotfile = ""):
        """
            Fit for zero-point and color-coefficient using Orthogonal Distance Regression

            The Model used is Mcal = Minst + ZP + c_f*(M1_PS1 - M2_PS1)

            The parameters fitted are ZP (the zero-point) and c_f (the color-coefficient)

            Depending on which ZTF-Filter is used the M1_PS1 and M2_PS1 are different:
            for g: g_PS1 - r_PS1
            for r: g_PS1 - r_PS1
            for i: r_PS1 - i_PS1

            Parameters:
            -----------

                ztf_filter: `str`
                    name of the ZTF-filter that was used for the fits
                    (calibration changes depending on filter)
                    Accepted values: 'g', 'r' or 'i'

                mag_col: 'str'
                    name of the column of the instrumental magnitude in the dataframe

                mag_err_col: 'str'
                    name of the column of the instrumental magnitude error in the dataframe

                gmag_col: 'str'
                    name of the column of PS1-magnitude in g band --> LIKEWISE FOR error as well as r and i band

                return_values: 'bool'
                    if True, the zero-point and color-coefficient
                    (and the errors) will be returned by the method as two arrays

                append_columns: 'bool'
                    if True, the zero-point and color-coefficient
                    (and the errors) will be appended as columns to the dataframe

                plot: 'bool'
                    if True, the instrumental magnitude will be plotted vs. PS1 magnitude (of chosen filter) and
                    the PS1 magnitude delta as color

                plotfile: 'str'
                    the filename of the plot

        """

        if logger is None:
            logger = srcdf._class_logger
        logger.info("Fitting (Orthogonal Distance Regression) for zero-point and color-coefficient using PS1 calibrators")

        # TODO: Find out why some values in the gmag/rmag/imag_err-columns are exactly 0

        # exclude 0 values of uncertainty to avoid divide-by-zero-error in fitting
        if ztf_filter == 'g':
            df = self.query(gmag_err_col + ' != 0')
        if ztf_filter == 'r':
            df = self.query(rmag_err_col + ' != 0')
        if ztf_filter == 'i':
            df = self.query(imag_err_col + ' != 0')

        gmag = df[gmag_col].values
        gmag_err = df[gmag_err_col].values

        rmag = df[rmag_col].values
        rmag_err = df[rmag_err_col].values

        imag = df[imag_col].values
        imag_err = df[imag_err_col].values

        inst_mag = df[mag_col].values
        inst_mag_err = df[mag_err_col].values

        if ztf_filter == 'g':
            logger.info("Ftting zero-point and color-coefficient for g-filter")
            mag_delta = gmag - rmag
            mag_delta_err = np.sqrt(gmag_err**2 + rmag_err**2)
            yvalues = gmag
            yvalues_err = gmag_err


        elif ztf_filter == 'r':
            logger.info("Fitting zero-point and color-coefficient for r-filter")
            mag_delta = gmag - rmag
            mag_delta_err = np.sqrt(gmag_err**2 + rmag_err**2)
            yvalues = rmag
            yvalues_err = rmag_err

        elif ztf_filter == 'i':
            logger.info("Fitting zero-point and color-coefficient for i-filter")
            mag_delta = rmag - imag
            mag_delta_err = np.sqrt(rmag_err**2 + imag_err**2)
            yvalues = imag
            yvalues_err = imag_err

         # create matrix from instrumental magnitude and PS1 magnitude delta (M1_PS1 - M2_PS2) for fit-methods
        xvalues = np.vstack([inst_mag, mag_delta])
        xvalues_err = np.vstack([inst_mag_err, mag_delta_err])

        def func(B, x):
            return B[0] + x[0] + (B[1]*x[1])

        # data is the measured magnitudes and their error, the model is the function
        data = RealData(x = xvalues, y = yvalues, sx = xvalues_err, sy = yvalues_err)
        model = Model(func)

        # do the actual fit (beta0 are starting points for the regression)
        odr_fit = ODR(data, model, beta0 = [20,-0.01])
        output = odr_fit.run()
        logger.info("Reason(s) for halting:")
        logger.info(output.stopreason)
        logger.info("Reduced Chi-square: ")
        logger.info(output.res_var)

        # plot instrumental magnitude vs. PS1 magnitude with PS1 magnitude delta as color
        if plot:
            cm = plt.cm.get_cmap('viridis')
            fig1 = plt.figure()
            ax = plt.axes()
            ax.set_title("Magnitude comparison")
            sc = ax.scatter(inst_mag, yvalues, c=mag_delta, vmin=0, vmax=2, cmap=cm)
            ax.set_xlabel("ZTF instrumental magnitude [mag]")
            ax.set_ylabel("PS1 magnitude [mag]")
            cbar = fig1.colorbar(sc)
            cbar.set_label('PS1 magnitude delta [mag]', rotation=270, labelpad = 20)
            fig1.savefig(plotfile)

        # append the fitted parameters and their sigmas (!) as columns
        if append_columns:
            zp_col = "fit_zp_" + ztf_filter
            zp_sig_col = "sig_" + "fit_zp_" + ztf_filter
            clrcoeff_col = "fit_clrcoeff_" + ztf_filter
            clrcoeff_sig_col = "sig_" + "fit_clrcoeff_" + ztf_filter
            self[zp_col] = output.beta[0]
            self[zp_sig_col] = output.sd_beta[0]
            self[clrcoeff_col] = output.beta[1]
            self[clrcoeff_sig_col] = output.sd_beta[1]

        # return the fitted parameters and their sigmas (!)
        if return_values:
            return output.beta, output.sd_beta


    def calmag(self, mag_col, zp, zp_err = None, err_mag_col = None, calmag_col = None,
        clrcoeff = None, clrcoeff_err = None, ps1_color1 = None, ps1_color2 = None,
        e_ps1_color1 = None, e_ps1_color2 = None, logger = None):
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

                zp/clrcoeff: `float`, or `array-like`
                    value(s) of ZP and color coefficient term. If clrcoeff_name is None,
                    color correction will be ignored.

                zp_err/clrcoeff_err: `float`, or `array-like`
                    value(s) of the errors on ZP and the color coefficient term.

                calmag_col: `str` or None
                    name of the column in the dataframe which will host the
                    calibrated magnitude value and its error. If calmag_col is None,
                    defaults to: cal_+mag_col. If the error is computed, the name of
                    the columns containting it will be err_calmag_col

                ps1_color1[2]: `array-like` or None
                    Contain, for each source in the dataframe, the PS1Cal colors
                    for the respective stars.

                e_ps1_color1[2]: `array-like` or None
                    Contain, for each source in the dataframe, the errors on the
                    PS1Cal colors for the respective stars.
        """

        if logger is None:
            logger = srcdf._class_logger
        logger.info("Applying photometric calibration.")
        if clrcoeff is None:
            logger.warning("color correction will not be applied")

        # name the cal mag column and the one for the error
        if calmag_col is None:
            calmag_col = "cal_"+mag_col
        err_calmag_col = "err_"+calmag_col

        # fill them
        if clrcoeff is None:
            self[calmag_col] = self[mag_col] + zp
            if not err_mag_col is None:
                self[err_calmag_col] = np.sqrt(
                                    self[err_mag_col]**2. +
                                    zp_err**2.)
        else:
            ps1_color = (ps1_color1 - ps1_color2)
            self[calmag_col] = (
                self[mag_col] +
                zp +
                clrcoeff*ps1_color)
            if not err_mag_col is None:
                d_ps1_color = np.sqrt( e_ps1_color1**2. + e_ps1_color2**2. )
                self[err_calmag_col] = np.sqrt(
                    self[err_mag_col]**2. +
                    zp_err**2. +
                    (clrcoeff_err * ps1_color)**2. +
                    (clrcoeff * d_ps1_color)**2)


    def compute_camera_coord(self, rc_x_name, rc_y_name, cam_x_name = 'cam_xpos',
        cam_y_name = 'cam_ypos', rotate = False, xgap = 7, ygap = 10, rcid_name = 'RCID', logger = None):
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

                x[y]gap: `int`
                    size of gap between CCDs, in pixels.

                rcid_name: `str`
                    name of column containing the ID of the readout-channels (0 to 63).

                logger: `logging.logger`
                    logger instance.
        """

        if logger is None:
            logger = srcdf._class_logger

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
        xll = (xrc // 2)*xgap + xrc*self.xsize
        yll = (yrc // 2)*ygap + yrc*self.ysize

        # finally add the x/y position inside each RC
        if not rotate:
            self[cam_x_name] = xll + self[rc_x_name]
            self[cam_y_name] = yll + self[rc_y_name]
        else:
            self[cam_x_name] = xll - self[rc_x_name]
            self[cam_y_name] = yll - self[rc_y_name]
        logger.info("computed camera-wide coordinates of the sources as columns: %s %s of the dataframe"%
            (cam_x_name, cam_y_name))


    def compute_ccd_coord(self, rc_x_name, rc_y_name, ccd_x_name='ccd_xpos', ccd_y_name='ccd_ypos', 
        rotate=True, rcid='RCID', logger=None):
        """
            Compute the CCD-wide x/y coordinates of the sources. The CCD coordinates 
            spans the range 3072*2 pix in the x direction and 3080*2 pix in the y direction.
            
            compute the CCD-wide x/y coordinates of the sources in the table. The x,y 
            position start at the bottom-left corner of q3 and the upper-right corner of q1 
            is at position (2*3072, 2*3080).

            Parameters:
            -----------

                rc_x[y]_name: `str` or `int`
                    name of dataframe column containg the position of the sources
                    in pixel on the readout channel (RC). If scalar, indicate the
                    RC for all the sources in this dataframe.

                cam_x[y]_name: `str`
                    name of the columns that will contain the camera-wide coordinates.

                rotate: `bool`
                    if True, the individual readout channels will be rotated by 180 deg.

                x[y]gap: `int`
                    size of gap between CCDs, in pixels.

                rcid_name: `str`
                    name of column containing the ID of the readout-channels (0 to 63).

                logger: `logging.logger`
                    logger instance.
        """

        if logger is None:
            logger = srcdf._class_logger
        check_col([rc_x_name, rc_y_name], self)
        
        # compute quadrant (1 to 4) ID from RC ID. Parse the info
        # on the RC ID, depending if column name, single value, or other array.
        if type(rcid) is str:
            check_col(rcid, self)
            rcid_arr = self[rcid_name]
        elif np.isscalar(rcid):
            rcid_arr = np.array([rcid]*len(self))
        else:
            rcid_arr = np.array(rcid)
        q = pd.Series((rcid_arr.astype(int)%4 + 1), index=self.index, name='q')

        # extract readout channel positions
        x_rc, y_rc = [np.array(_) for _ in zip(*self[[rc_x_name, rc_y_name]].values)]
        
        # rotate 180 deg: change sign to both local x and y
        if rotate:
            x_rc *= -1
            y_rc *= -1
        
        # compute the position the lower-left corner of the different RC as 
        # arranged on the camera  and add the corner to the ccd coordinates
        xll = 0 + self.xsize*np.logical_or(q==1, q==4)
        yll = 0 + self.ysize*np.logical_or(q==1, q==2)
        xccd, yccd = x_rc + xll, y_rc + yll
        self[ccd_x_name] = xccd
        self[ccd_y_name] = yccd
        logger.info("computed CCD-wide coordinates of the sources as columns: %s %s of the dataframe"%
            (ccd_x_name, ccd_y_name))
        return (xccd, yccd)


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
        rej_df = self.query(expr = query, inplace = True)
        return rej_df


    def tag_dust(self, dust_df_file, radius_multiply = 1., xname = 'xpos', yname = 'ypos',
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
            logger = srcdf._class_logger

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
