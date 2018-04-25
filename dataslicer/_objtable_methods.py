#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# utility class that will gather some (long-coded) methods of the objtable class.
# 
# Author: M. Giomi (matteo.giomi@desy.de)

import tqdm, time, jenkspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extcats import CatalogQuery
from dataslicer.df_utils import check_col, match_to_PS1cal, match_to_PS1cal_fields

class _objtable_methods():
    
    # ------------------------------------------------------------- #
    #                                                               #
    #           mathods that plays with the PS1 calibrators         #
    #                                                               #
    # ------------------------------------------------------------- #
    
    def match_to_PS1cal(self, rs_arcsec, use_clusters, xname, yname,
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
                    
                    use_clusters: `bool`
                        if True, match each of this object clusters to the calibrators, rather
                        than individual sources. In this case, the cluster centroid position is 
                        used. 
                        If False, the matching will try to use FIELDID and RCID to speed up the query.
                        If this information is not included in this object dataframe, or if the 
                        fields are not the 'standard' ones, the matching will be done source by source.
                     
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
            
            if use_clusters:
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
                
            else:
                # if you have the field & RC id for every object, use them if possible.
                dfcols = self.df.columns.tolist()
                match_src_by_src = False
                if 'FIELDID' in dfcols and 'RCID' in dfcols:
                    
                    fields = self.df.FIELDID.unique().astype(int).tolist()
                    rcids = self.df.RCID.unique().astype(int).tolist()
                    self.logger.info("found objtable sources for %d fields and %d RCs."%
                        (len(fields), len(rcids)))
                    
                    # get the ones you have in the catalog:
                    ps1cal_fields = ps1cal_query.src_coll.distinct('field')
                    self.logger.info("PS1 calibrator database is indexed in %d fields"%
                        (len(ps1cal_fields)))
                    
                    # see if the fields in the objtable are also in the database
                    if all([ff in ps1cal_fields for ff in fields]):
                        self.logger.info(
                            "Matching to PS1 cal using FIELDID and RC information")
                        match_to_PS1cal_fields()
                        
                    else:
                        self.logger.warning("cannot find objtable fields in PS1 cal.")
                        match_src_by_src = True
                else:
                    self.logger.info("FIELDID and/or RCID missing from objtable dataframe.")
                    match_src_by_src = True
                
                if match_src_by_src:
                    self.logger.info("Matching each of the %d sources to the PS1 cal database."%(len(self.df)))
                    
                    # do the matching
                    match_to_PS1_kwargs['ids'] = ('srcID', self.df['srcID'])
                    ps1cp_df = match_to_PS1cal(self.df[xname], self.df[yname], rs_arcsec, **match_to_PS1_kwargs)
                    self.logger.info("Found PS1 calibrators for %d sources"%
                        (len(ps1cp_df)))
                        
                    # merge the dataframe
                    self.df = self.df.merge(
                        ps1cp_df, on = 'srcID', how='left',  suffixes=['', '_ps1'])
                    
                    # if requested, drop sources without PS1cp
                    if clean_non_matches:
                        self.df.dropna(subset = ['dist2ps1'], inplace = True)
                        self.logger.info("dropping sources without match in PS1cal DB: %d retained."%
                            (len(self.df)))
            if plot:
                # create diagnostic plot
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
        
        
    def ps1based_outlier_rm_iqr(self, cal_mag_name, ps1mag_name, norm_mag_diff_cut, n_mag_bins=10, plot = True):
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
                
                ps1mag_name: `str`
                    name of PS1 cal magnitude to compare to cal_mag_name.
                    
                norm_mag_diff_cut: `float`
                    cut on the normalized magnitude difference to isolate outliers
                    (norm_mag_diff > norm_mag_diff_cut) from the rest.
                
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
        
        # checks
        self._check_for_gdf()
        check_col([cal_mag_name, ps1mag_name], self.gdf)
        
        # create smaller df with cluster average mag and it's difference wrt ps1 cal
        start = time.time()
        mag_av = self.gdf[cal_mag_name].mean()
        mag_diff = self.gdf[cal_mag_name].mean() - self.gdf[ps1mag_name].mean()
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
                out_mag_diff = grp_outl_df[cal_mag_name].mean() - grp_outl_df[ps1mag_name].mean()
                ax1.scatter(out_mag_av, out_mag_diff, c= 'r', marker = 'x')
                
            # plot the bins
            ax1.errorbar(bin_x, bin_y, xerr = 0, yerr = iqr.values, c ='r')
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
        
        # replace the clean dfs
        self.df = clean_df
        self.gdf = clean_gdf
        self.logger.info("%d sources and %d clusters retained."%(len(self.df), len(self.gdf)))
        return (outl_df)


    def select_clusters(self, cond):
        """
            remove the clusters and corresponding sources from the dataframe unless
            a given condition is satisfied by all the member of the cluster.
            
            Parameters:
            -----------
            
                cond: `str`
                    condition that all the member of a cluster have to satisfy in order
                    not to be rejected.
                
                plot: `bool`
                    if you want to visualize the result of the procedure.
            
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
        self.gdf = self.df.groupby('clusterID', sort = False)
        self.logger.info("%d sources and %d clusters retained."%(len(self.df), len(self.gdf)))
        return rejected
