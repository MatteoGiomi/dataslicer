#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# utility class that will gather some (long-coded) methods of the objtable class.
# 
# Author: M. Giomi (matteo.giomi@desy.de)
import os
import tqdm, time, jenkspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extcats import CatalogQuery
from dataslicer.df_utils import check_col
from dataslicer.PS1Cal_matching import match_to_PS1cal, match_to_PS1cal_fields

class _objtable_methods():
    
    # ------------------------------------------------------------- #
    #                                                               #
    #           mathods that plays with the PS1 calibrators         #
    #                                                               #
    # ------------------------------------------------------------- #
    
    def match_to_PS1cal(self, rs_arcsec, use, xname, yname,
            clean_non_matches = True, plot = True, **match_to_PS1_kwargs): 
            """
                Match the sources in the objtable to the PS1 calibrator stars. To each
                row in the dataframe the catalog entry of the found PS1 cp is added.
                
                Use the extcats package to do the matching. For this, the PS1 calibrators
                have to be arranged in a mongo database.
                
                Parameters:
                -----------
                     
                    rs_arcsec: `float`
                        search radius, in arcseconds for matching with the PS1 calibrators.
                    
                    clean_non_matches: `bool`
                        if True, sources with no match in the PS1cal db are removed.
                     
                    x[/y]name: `str`
                        name for table columns specifiyng the x,y (Equatorial, J2000) coordinates to use.
                    
                    match_to_PS1_kwargs: `kwargs`
                        to be passed to df_utils.match_to_PS1. These includes:
                            * col2rm: `list`
                                names of columns in the PS1 calibrator catalogs to be 
                                excluded from the resulting dataframe. Default are:
                                ['rcid', 'field', '_id', 'hpxid_16']
                            
                            *dbclient: `pymongo.MongoClient`
                                pymongo client that manages the PS1 calibrators databae.
                                This is passed to extcats CatalogQuery object. Default is None
                    
                    use: `str`
                        how the matching is done. Allowed values are:
                            * clusters:
                                match each of this object clusters to the calibrators, rather
                                than individual sources. In this case, the cluster centroid position is 
                                used. 
                            * fieldid:
                                the matching will use FIELDID and RCID to pre-select PS1 sources
                                and then match the two coordinate lists.
                            * srcs:
                                Do the matching source bey source. This is the fallback option.
                    
                    plot: `bool`
                        if True, a disganostic plot showing the distribution of the 
                        PS1cal - cluster distance will be created and saved.
            """
            
            self.logger.info("matching objtable entries to PS1 calibrator stars")
            self.logger.info("using %s, %s as coordinates and a search radius of %.2f arcsec"%(
                xname, yname, rs_arcsec))
            
            # make room for defaults
            if match_to_PS1_kwargs is None:
                match_to_PS1_kwargs = {}
            
            # initilaize the query object and pass it to the function.
            ps1cal_query = CatalogQuery.CatalogQuery(
                'ps1cal', 'ra', 'dec', dbclient = None, logger = self.logger)
            match_to_PS1_kwargs['ps1cal_query'] = ps1cal_query
            
            if use == "clusters":
                self.logger.info("Matching cluster centroids with the PS1 calibrators")
                av_x, av_y = self.compute_cluster_centroid(xname, yname)
                
                # do the matching
                match_to_PS1_kwargs['ids'] = 'clusterID'
                ps1cp_df = match_to_PS1cal(av_x, av_y, rs_arcsec, **match_to_PS1_kwargs)
                self.logger.info("Found PS1 calibrators for %d clusters"%
                    (len(ps1cp_df)))
                
                # merge the dataframe
                self.df = self.df.merge(
                    ps1cp_df, on = 'clusterID', how='left',  suffixes=['', '_ps1'])
                
                # if requested, drop sources without PS1cp
                if clean_non_matches:
                    self.df.dropna(subset = ['dist2ps1'], inplace = True)
                    self.logger.info("dropping sources without match in PS1cal DB: %d retained."%
                        (len(self.df)))
                
                # update the grouped dataframe
                self.logger.info("updating cluster dataframe.")
                self.gdf = self.df.groupby('clusterID', sort = False)
                
            elif use == "fieldid":
                self.logger.info("Matching to PS1 cal using FIELDID and RC information")
                self.df = match_to_PS1cal_fields(
                                                self.df, 
                                                rs_arcsec,
                                                ra_key=xname,
                                                dec_key=yname, 
                                                clean_non_matches=clean_non_matches,
                                                ps1cal_query=ps1cal_query,
                                                logger=self.logger)
            elif use == "srcs":
                    # TODO: Add logic so that if there are no clusters, or no FIELDID use the src-by-source
                    self.logger.info("Matching each of the %d sources to the PS1 cal database."%(len(self.df)))
                    
                    # do the matching
                    match_to_PS1_kwargs['ids'] = ('srcID', self.df['srcID'])
                    ps1cp_df = match_to_PS1cal(self.df[xname], self.df[yname], rs_arcsec, **match_to_PS1_kwargs)
                    self.logger.info("Found PS1 calibrators for %d sources"%
                        (len(ps1cp_df)))
                        
                    # merge the dataframe
                    self.df = self.df.merge(
                        ps1cp_df, on = 'srcID', how='left',  suffixes=['', '_ps1'])
            else:
                raise RuntimeError("Parameter 'use' must be in ['clusters', 'fieldid', 'srcs'], got %s."%use)
            
            # if requested, drop sources without PS1cp
            if clean_non_matches:
                self.df.dropna(subset = ['dist2ps1'], inplace = True)
                self.logger.info("dropping sources without match in PS1cal DB: %d retained."%
                    (len(self.df)))
            
            # create diagnostic plot
            if plot:
                
                fig, ax = plt.subplots()
                if hasattr(self, 'gdf'):
                    h = ax.hist(self.gdf['dist2ps1'].max(), bins = 50, log = True)
                    ax.set_xlabel("cluster distance to PS1 calibrator [arcsec]")
                else:
                    h = ax.hist(self.df['dist2ps1'], bins = 50, log = True)
                    ax.set_xlabel("source distance to PS1 calibrator [arcsec]")
                fig.tight_layout()
                self.save_fig(fig, '%s_match_to_PS1cal.png'%self.name)
                plt.close()


    def add_bandwise_PS1mag_for_filter(self, bandwise_ps1mag_name='ps1mag_band', filterid_col='FILTERID'):
        """
            add a new column to the dataframe with the right PS1 magnitude depending 
            on the value of filterid_col.
            
            Parameters:
            -----------
                
                bandwise_ps1mag_name: `str`
                    name of new column to be added.
                
                filterid_col: `str` or None
                    name of the columns containing the ID of the FTZ filter. 
                    The following rule applies:
                    to:     
                            FILTER ID      BAND
                                1           g
                                2           r
                                3           i
        """
        
        check_col(filterid_col, self.df)
        
        self.logger.info("adding band-wise PS1 magnitude column %s to the df depending on %s"%
            (bandwise_ps1mag_name, filterid_col))
        self.df.loc[self.df[filterid_col] == 1, bandwise_ps1mag_name] = self.df['gmag']
        self.df.loc[self.df[filterid_col] == 2, bandwise_ps1mag_name] = self.df['rmag']
        self.df.loc[self.df[filterid_col] == 3, bandwise_ps1mag_name] = self.df['imag']
        
        # now propagate the changes to the grouped dataframe and remember to clean up
        self.update_gdf()

    def calculate_quality(self, cal_mag_name = 'cal_mag', ps1mag_name = 'p1mag_band', df_to_append = None, save_dir = None):
        """
            calculates the calibration bias for a given dataframe as a measure of calibration quality. It retains only PS1 matched calibrator
            stars lying in a magnitude bin of [18.5,17.5] (calibrated instrumental magnitude, including zero point
            correction and color correction) and is calculated for each readout channel and exposure independently. The quantity is defined as follows:

                bias(readout_channel, exposure) = < abs(mag_calibrated(readout_channel, exposure) - mag_PS1) >
                spread(readout_channel, exposure) = stddev(abs(mag_calibrated(readout_channel, exposure) - mag_PS1))

            Both values are given in millimag!

            Parameters:
            -----------
            
                cal_mag_name: `str`
                    name of column with the calibrated ZTF magnitudes.

                ps1mag_name: `str`
                    name of PS1 cal magnitude column to compare to cal_mag_name.

                df_to_append: `pandas Dataframe`
                    if one wishes to append the results to an existing dataframe, it can be passed here.

                save_dir: 'str'
                    name of the directory the resulting dataframe should be saved to as csv. If None is given, the
                    dataframe will only be returned, but not saved

            Returns:
            --------
                
                pandas.DataFrame with calibration quality for each readout channel and exposure
                The DataFrame contains the following columns: EXPID, FILTERID, FIELDID, OBSMJD, RCID, bias, spred, # calibrators
        """

        self.logger.info('calculating quality of calibration for given dataframe')

        if df_to_append is None:
            quality_df = pd.DataFrame(columns = ['EXPID', 'FIELDID', 'OBSMJD', 'RCID', 'bias', 'median', '# calibrators', 'spread'])
        else:
            quality_df = df_to_append

        # cut to magnitude bin
        df_temp = self.df
        df_temp.drop(df_temp[df_temp[cal_mag_name] < 17.5].index, inplace = True)
        df_temp.drop(df_temp[df_temp[cal_mag_name] > 18.5].index, inplace = True)

        # calculate absolute magnitude difference between ZTF measurement and PS1 calibrators
        # covert to millimag
        df_temp['abs_millimag_diff'] = np.abs(df_temp[cal_mag_name] - df_temp[ps1mag_name])*1000

        grouped = df_temp.groupby(by = ['OBSMJD', 'EXPID', 'FIELDID', 'FILTERID'], group_keys = True)

        ## calculate mean, stddev and number of calibrator stars for each RC and exposure
        def cutfunc1(x, y):
            mask = (x['RCID'] == y)
            return pd.DataFrame([{'bias': x.abs_millimag_diff[mask].abs().mean(), 'spread': x.abs_millimag_diff[mask].std(), 
                'median': x.abs_millimag_diff[mask].abs().median(), '# calibrators': x.abs_millimag_diff[mask].count(), 'RCID': y}])

        for rcid in df_temp['RCID'].unique():
            grouped_cut = grouped.apply(cutfunc1, rcid).reset_index(level = ['OBSMJD', 'EXPID', 'FIELDID', 'FILTERID'])
            quality_df = pd.concat([quality_df, grouped_cut], sort = True)
        quality_df.sort_values(by = ['EXPID', 'RCID'], inplace = True)
        quality_df.reset_index(drop = True, inplace = True)

        if save_dir:
            save_file = os.path.join(save_dir, 'quality.csv')
            quality_df.to_csv(save_file)
            self.logger.info('quality of calibration has been calculated and saved to {}'.format(save_file))

        else:
            self.logger.info('quality of calibration has been calculated')

        return quality_df


    def ps1based_outlier_rm_iqr(self, cal_mag_name, norm_mag_diff_cut, filterid_col='FILTERID', ps1mag_name=None, n_mag_bins=10, plot=True):
        """
            remove clusters based on the normalized difference between the cluster average
            magnitude and that of the associated PS1 calibrator star. The algorithm
            works as follows:
                
                - the average magnitude is computed for each cluster and compared 
                to the PS1 magnitude for the cluster counterpart.
                
                - the dataset is divided in 'natural' magnitude bins using the Jenks algorythm.
                
                - for each bin the inter quantile range (IQR) and the median of the 
                (cluster - PS1) magnitude difference is computed.
                
                - the (source - PS1) magnitude distance is computed and divided by the 
                IQR. That's the quantity you cut on.
            
            Parameters:
            -----------
            
                cal_mag_name: `str`
                    name of columns with the calibrated ZTF magnitudes.
                
                filterid_col: `str` or None
                    name of the columns containing the ID of the FTZ filter. If None, 
                    ps1mag_name must be given. The filter ID translates in the following bands:
                    to:     
                            FILTER ID      BAND
                                1           g
                                2           r
                                3           i
                    
                norm_mag_diff_cut: `float`
                    cut on the normalized magnitude difference to isolate outliers
                    (norm_mag_diff > norm_mag_diff_cut) from the rest.
                
                ps1mag_name: `str`
                    if filterid_col is None, name of PS1 cal magnitude to compare to cal_mag_name.
                
                n_mag_bins: `int`
                    number of magnitude bins to use.
                
                plot: `bool`
                    if you want to visualize the result of the procedure.
            
            Returns:
            --------
                
                pandas.DataFrame with the rejected outliers.
        """
        
        ## TODO: implement this also for single sources.
        self.logger.info(
            "rejecting outliers based on the IQR normalized magnitude difference wrt PS1 cal")
        
        # if filterid_col is given and it is present in the dataframre
        # use it to select the right PS1 magnitude to compare with
        cleanup, aux_ps1mag_name = False, 'aux_ps1mag'
        if not ps1mag_name is None:
            self.logger.info("using PS1 magnitude from column: %s"%ps1mag_name)
            aux_ps1mag_name = ps1mag_name
        elif filterid_col in self.df.columns.values:
            self.add_bandwise_PS1mag_for_filter(aux_ps1mag_name, filterid_col)
            cleanup = True  # remember to cleanup afterwards
        else:
            raise RuntimeError("either 'ps1mag_name' of 'filterid_col' must be specified.")
        
        self._check_for_gdf()
        check_col([cal_mag_name, aux_ps1mag_name], self.gdf)
        
        # create smaller df with cluster average mag and it's difference wrt ps1 cal
        start = time.time()
        mag_av = self.gdf[cal_mag_name].mean()
        mag_diff = self.gdf[cal_mag_name].mean() - self.gdf[aux_ps1mag_name].mean()
        scatter_df = pd.concat(
                        [mag_av.rename('av_mag'),
                        mag_diff.rename('mag_diff')], axis = 1).reset_index()
        
        # bin the data using natural bins in mag
        mag_av_bins = jenkspy.jenks_breaks(scatter_df['av_mag'].values, n_mag_bins)
        binned_df = scatter_df.groupby(
                pd.cut(scatter_df['av_mag'].rename('av_mag_bin'), mag_av_bins, include_lowest = True))
        
        # compute the IQR of the magnitude difference in each av_mag bin.
        iqr = (binned_df['mag_diff'].quantile(0.75) - binned_df['mag_diff'].quantile(0.25))
        
        # for each cluster, compute the distance of the magnitude difference
        # wrt to the median for the bin, and normalize it with the IQR
        cluster_id, norm_mag_dist = [], []
        for mag_bin, iqr_val in iqr.iteritems():
            group = binned_df.get_group(mag_bin)
            norm_dist = np.abs(group['mag_diff'] - group['mag_diff'].median()) / iqr_val
            cluster_id.extend(group['clusterID'].values)
            norm_mag_dist.extend(norm_dist.values)
        
        # cast into dataframe and merge it with the sources using cluster ID
        norm_mag_dist_df = pd.DataFrame(
            {'clusterID': cluster_id, 'norm_mag_dist': norm_mag_dist})
        self.df = self.df.merge(norm_mag_dist_df, on = 'clusterID', how = 'left')
        
        # separate outliers from core samples and regroup
        clean_df = self.df.query('norm_mag_dist < @norm_mag_diff_cut')
        outl_df = self.df.query('not (norm_mag_dist < @norm_mag_diff_cut)')
        clean_gdf = clean_df.groupby('clusterID', sort = False)
        end = time.time()
        
        # print some stats
        self.logger.info("rejected %d clusters based on IRQ normalized residuals. Took %.2e sec"%(
            (len(self.gdf) - len(clean_gdf)), (end-start)))
        
        if plot:
            # grab some more quantities
            full_gdf = self.df.groupby('clusterID', sort = False)
            mag_dist = full_gdf['norm_mag_dist'].mean()
            bin_x = binned_df['av_mag'].median()
            bin_y = binned_df['mag_diff'].median()
            
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize = (15, 6), gridspec_kw = {'width_ratios':[2,1]})
            
            # scatter plot of the cluster magnitudes vs the PS1 ones
            sc = ax1.scatter(mag_av, mag_diff, c=mag_dist)
            
            # tag the outliers
            if len(outl_df)>0:
                grp_outl_df = outl_df.groupby('clusterID', sort = False)
                out_mag_av = grp_outl_df[cal_mag_name].mean()
                out_mag_diff = grp_outl_df[cal_mag_name].mean() - grp_outl_df[aux_ps1mag_name].mean()
                ax1.scatter(out_mag_av, out_mag_diff, c= 'r', marker = 'x')
                
            # plot the bins
            ax1.errorbar(bin_x.values, bin_y.values, xerr = 0, yerr = iqr.values, c ='r')
            for x in mag_av_bins: ax1.axvline(x, c = 'k', lw = 0.7, ls = '--')
            
            cb = plt.colorbar(sc, ax = ax1)
            cb.set_label("normalized mag distance")
            ax1.set_xlabel("cluster calibrated mag [mag]")
            ax1.set_ylabel("cal mag - PS1 mag [mag]")

            ax2.hist(full_gdf['norm_mag_dist'].apply(lambda x: x).values, bins = 50, log = True)
            ax2.axvline(norm_mag_diff_cut, c = 'r')
            ax2.set_xlabel("normalized mag distance")
            fig.tight_layout()
            self.save_fig(fig, '%s_ps1based_outlier_rm.png'%self.name)
            plt.close()
        
        # remove aux_ps1_magniute
        if cleanup:
            self.logger.info("removing auxiliary column: %s"%aux_ps1mag_name)
            self.df.drop(columns=[aux_ps1mag_name], inplace=True)
            self.update_gdf()
        
        # replace the clean dfs
        self.df = clean_df
        self.gdf = clean_gdf
        self.logger.info("%d sources and %d clusters retained."%(len(self.df), len(self.gdf)))
        return (outl_df)


    def select_clusters(self, cond, plot_x = None, plot_y = None, **plt_kwargs):
        """
            remove the clusters and corresponding sources from the dataframe unless
            a given condition is satisfied by all the member of the cluster.
            
            Parameters:
            -----------
            
                cond: `str`
                    condition that all the member of a cluster have to satisfy in order
                    not to be rejected.
                
                plot_x/plot_y: `str` or None
                    if you want to visualize the result of the procedure as a 1d hist
                    (plot_x != None and plot_y == None) or a scatter plot (both plot_x and 
                    plot_y not None).
                
                plt_kwargs:
                    kwargs for plot. 
            
            Returns:
            --------
                
                pandas.DataFrame with the rejected outliers.
        """
        
        # TODO: extend so that you can ask for cluster in which AT LEAST n sources satisfy
        # the condition
        
        self.logger.info("selecting clusters for which %s is satified by all the members"%cond)
        
        # select sources with magnitude difference greater than the cut
        rejected = self.df.query("not (%s)"%cond)
        self.logger.info("found %d sources that do not satisfy the condition"%len(rejected))
        
        # find out which clusters are affected
        bad_cluster_ids = rejected['clusterID'].unique()
        self.logger.info("%d sources do not satisfy the condition. They belong to these %d clusters: %s"%
            (len(rejected), len(bad_cluster_ids), ", ".join(["%d"%cid for cid in bad_cluster_ids])))
        
        # remove all the sources contained in these clusters and go
        self.df.query("clusterID not in @bad_cluster_ids", inplace = True)
        self.update_gdf()
        self.logger.info("%d sources and %d clusters retained."%(len(self.df), len(self.gdf)))
        
        if not (plot_x is None and plot_y is None):
            fig, ax = plt.subplots()
            if plot_y is None:
                pngname = "select_clusters_hist_%s"%plot_x
                bins=np.histogram(np.hstack((self.df[plot_x],rejected[plot_x])), bins=50)[1]
                ax.hist(self.df[plot_x], bins=bins, label="accepted", **plt_kwargs)
                ax.hist(rejected[plot_x], bins=bins, label="rejected", **plt_kwargs)
                ax.set_xlabel(plot_x)
            else:
                pngname = "select_clusters_scatter_%s_vs_%s"%(plot_x, plot_y)
                ax.scatter(self.df[plot_x], self.df[plot_y], label = "accepted", **plt_kwargs)
                ax.scatter(rejected[plot_x], rejected[plot_y], label = "rejected", **plt_kwargs)
                ax.set_xlabel(plot_x)
                ax.set_ylabel(plot_y)
            ax.set_title(cond)
            ax.legend()
            fig.tight_layout()
            self.save_fig(fig, '%s_%s'%(self.name, pngname))
            plt.close()
        return rejected


# --------------------------------------- #
# boneyard. I'm too cicken to delete them #
# these were the objtable method before   #
# the srcdf revolution.                   #
# --------------------------------------- #


#    def calmag(self, mag_col, err_mag_col = None, calmag_col = None, zp_name = 'MAGZP', 
#        clrcoeff_name = 'CLRCOEFF', zp_err = 'MAGZPUNC', clrcoeff_err = 'CLRCOUNC',
#        ps1_color1 = None, ps1_color2 = None, dropmag = False, plot = True):
#        """
#            apply photometric calibration to magnitude. The formula used is
#            
#            
#            Mcal = Minst + ZP_f + c_f*(M1_PS1 -M2_PS1)
#            
#            where Minst is the instrumental magnitude, ZP_f , c_f are the image-wise
#            zero point and color coefficient (MAGZP,CLRCOEFF), and (M1_PS1-M2_PS1)
#            is the color of the source in the PS1 system. The filter used are defined
#            by the header key PCOLOR.
#            
#            IMPORTANT: the above formula can be applied ONLY to PSF-fit catalogs.
#            
#            Parameters:
#            -----------
#            
#                mag_col: `str`
#                    name of the magnitude column you want to calibrate.
#                
#                err_mag_col: `str` or None
#                    name of the columns containing the error on mag_col. If None, 
#                    error on the calibrated magnitudes will not be computed. 
#                
#                calmag_col: `str` or None
#                    name of the column in the dataframe which will host the
#                    calibrated magnitude value and its error. If calmag_col is None, 
#                    defaults to: cal_+mag_col. If the error is computed, the name of
#                    the columns containting it will be err_calmag_col
#                
#                zp_name/clrcoeff_name: `str` or None
#                    name of ZP and color coefficient term. If clrcoeff_name is None, 
#                    color correction will be ignored.
#                
#                zp_err/clrcoeff_err: `str`
#                    name of columns containing the error on the ZP and color coefficient.
#                
#                ps1_color1[2]: `str` or array-like
#                    If strings, these are the names of the PS1cal magnitudes used
#                    to calibrate (they should be consistent with PCOLOR).
#                    If array-like they should have the same length of the dataframe.
#                
#                dropmag: `bool`
#                    if True the df column magname will be dropped.
#                
#                plot: `bool`
#                    if True, a diagnostic plot showing the histogram of the magnitudes
#                    and their errors is created in plotdir.
#        """
#        
#        self.logger.info("Applying photometric calibration.")
#        
#        # see what columns are needed
#        needed_cols = [mag_col, zp_name]
#        if clrcoeff_name is None:
#            self.logger.warning("color correction will not be applied")
#        else:
#            needed_cols.extend([clrcoeff_name])
#            if type(ps1_color1) == str and type(ps1_color2) == str:
#                needed_cols.extend([ps1_color1, ps1_color2])
#        if not err_mag_col is None:
#            needed_cols.extend([zp_err, clrcoeff_err, err_mag_col])
#        
#        check_col(needed_cols, self.df)
#        for k in needed_cols:
#            check_col(k, self.df)
#        
#        # name the cal mag column and the one for the error
#        if calmag_col is None:
#            calmag_col = "cal_"+mag_col
#        err_calmag_col = "err_"+calmag_col
#        
#        # fill them
#        if clrcoeff_name is None:
#            self.df[calmag_col] = self.df[mag_col] + self.df[zp_name]
#            
#            if not err_mag_col is None:
#                self.df[err_calmag_col] = np.sqrt(
#                                    self.df[err_mag_col]**2. +
#                                    self.df[zp_err]**2.)
#        else:
#            ps1_color = self.df[ps1_color1] - self.df[ps1_color2]
#            self.df[calmag_col] = (
#                self.df[mag_col] + 
#                self.df[zp_name] +
#                self.df[clrcoeff_name]*ps1_color)
#            
#            if not err_mag_col is None:
#                d_ps1_color = np.sqrt( self.df['e_'+ps1_color1]**2. + self.df['e_'+ps1_color2]**2. )
#                self.df[err_calmag_col] = np.sqrt(
#                    self.df[err_mag_col]**2. +
#                    self.df[zp_err]**2. +
#                    (self.df[clrcoeff_err] * ps1_color)**2. + 
#                    (self.df[clrcoeff_name] * d_ps1_color)**2)
#        
#        # eventually get rid of the uncalibrated stuff
#        if dropmag:
#            self.logger.info("dropping non calibrated magnitude %s from dataframe"%mag_col)
#            self.df.drop(columns = [mag_col])
#        
#        if plot:
#            
#            fig, ax = plt.subplots()
#            if err_mag_col is None:
#                ax.hist(self.df[calmag_col], bins = 100)
#                ax.set_xlabel("calibrated magnitude [mag]")
#            else:
#                ax.scatter(self.df[calmag_col], self.df[err_calmag_col])
#                ax.set_xlabel("calibrated magnitude [mag]")
#                ax.set_ylabel("error on calibrated magnitude [mag]")
#            fig.tight_layout()
#            self.save_fig(fig, '%s_calmag.png'%self.name)
#            plt.close()


#    def compute_camera_coord(self, rc_x_name, rc_y_name, cam_x_name = 'cam_xpos', 
#        cam_y_name = 'cam_ypos', xgap_pix = 7, ygap_pix = 10, rcid_name = 'RCID'):
#        """
#            compute the camera-wide x/y coordinates of the sources. The x,y position
#            start at the bottom-left corner of the camera (RC 14)
#            
#            Parameters:
#            -----------
#                
#                rc_x[y]_name: `str`
#                    name of dataframe column containg the position of the sources 
#                    in pixel on the readout channel (RC)
#                
#                cam_x[y]_name: `str`
#                    name of the columns that will contain the camera-wide coordinates.
#                
#                x[y]gap_pix: `int`
#                    size of gap between CCDs, in pixels.
#                
#                rcid_name: `str`
#                    name of column containing the ID of the readout-channels (0 to 63).
#        """
#        
#        # dimension of a RC in pixels
#        xsize, ysize = 3072, 3080
#        
#        # checks
#        check_col([rc_x_name, rc_y_name, rcid_name], self.df)
#        
#        # compute ccd and quadrant (1 to 4) from RC
#        ccd = (self.df[rcid_name]//4 + 1).rename('ccd')
#        q = (self.df[rcid_name]%4 + 1).rename('q')

#        # arrange the rc in rows and cols based on ccd and q.
#        # NOTE: the returned values are zero-indexed (from 0 to 7) and 
#        # start from the bottom-left corner of the image, so that RC 14 is
#        # at position (0, 0) and RC 48 at (7, 7).
#        yrc= 2*((ccd-1)//4) + 1*np.logical_or(q==1, q==2)
#        xrc= 2*( 4-(ccd-1)%4)-1 - 1*np.logical_or(q==2, q==3)
#        
#        # now add the gaps between the ccds, and the rc size in pixels 
#        # so that you have the x/y camera position of the lower-left corner of the RCs
#        # of the readout channels
#        xll = (xrc // 2)*xgap_pix + xrc*xsize
#        yll = (yrc // 2)*ygap_pix + yrc*ysize
#        
#        # finally add the x/y position inside each RC
#        self.df[cam_x_name] = xll + self.df[rc_x_name]
#        self.df[cam_y_name] = yll + self.df[rc_y_name]
#        self.logger.info("computed camera-wide coordinates of the sources as columns: %s %s of the dataframe"%
#            (cam_x_name, cam_y_name))
#        
#        # TODO: rotation?
#        
#        # update grouped df
#        self.update_gdf()

