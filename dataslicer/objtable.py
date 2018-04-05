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


    def apply_zp(self, metadata, zp_col_name = 'MAGZP', mag_col = 'MAG_AUTO', 
        zp_mag_col = None, join_on = 'OBSID'):
        """
            apply ZP from the fits header stored in 
            the metadata to the specified magnitude.
            
            Parameters:
            -----------
            
                metadata: `pandas.DataFrame` or None
                    dataframe with the metadata for the files you want to load. 
                    If None, all the files in this object self.files attribute will be used.
                
                zp_col_name: `str`
                    name of metadata column containing the zero point.
                
                mag_col: `str`
                    name of this object dataframe colum you want to apply the ZP to.
                
                zp_mag_col: `str`
                    name of the colum that will contain the zp corrected magnitude.
                    If None, it will default to 'ZPC_'+mag_col
                
                join_on: `str`
                    name of the column used to join the metadata and source data.
        """
        
        # check you have everything you need
        self._check_for_df()
        if (not join_on in self.df.columns):
            raise KeyError("Join on column %s not present in objtable dataframe"%join_on)
        if (not join_on in metadata.columns):
            raise KeyError("Join on column %s not present in metadata dataframe"%join_on)
        
        self.logger.info("Applying ZP correction to %s. Results will be parsed in %s"%
            (mag_col, zp_mag_col))
        self.df[zp_mag_col] = (
                    self.df[mag_col] + 
                    metadata[metadata[join_on] == self.df[join_on]][zp_col_name])


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
        if target_metadata_df is None:
            frames  = [fits_to_df(ff, **true_args) for ff in tqdm.tqdm(files)]
        else:
            frames = []
            for ff in tqdm.tqdm(files):
                buff = fits_to_df(ff, **true_args)
                if add_obs_id:
                    obsid = target_metadata_df[
                        target_metadata_df['PATH'] == ff]['OBSID'].values[0]
                    buff['OBSID'] = pd.Series( [obsid]*len(buff), dtype = int)
                frames.append(buff)
        self.df = pd.concat(frames)
        end = time.time()
        self.logger.info("loaded %d files into a %d rows dataframe. Took %.2e sec"%
            (len(files), len(self.df), (end-start)))


    def cluster_sources(self, cluster_size_arcsec, min_samples,
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
        self.gdf = clean_df.groupby('clusterID', sort = False)
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
                
                average_ra, average_dec of each group as array.
        """
        
        if wav is True:
            raise NotImplementedError("cluster centroid with weighted average not implemented yet.")
        
        if not hasattr(self, 'gdf'):
            raise AttributeError("this object has no grouped dataframe.")
        return (self.gdf[xname].mean(), self.gdf[yname].mean())


    def get_cluster_sizes(self):
        """
            return a list of the size of each cluster in this object
            grouped dataframe.
        """
        if not hasattr(self, 'gdf'):
            raise AttributeError("this object has no grouped dataframe.")
        return [len(group) for _, group in self.gdf]


    def match_to_PS1cal(self, rs_arcsec, use_clusters, clean_non_matches = True,
            xname = 'ALPHAWIN_J2000', yname = 'DELTAWIN_J2000', 
            col2rm = ['rcid', 'field', '_id', 'hpxid_16'], dbclient = None): 
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
                    
                    col2rm: `list`
                        names of columns in the PS1 calibrator catalogs to be excluded from the
                        resulting dataframe.
                    
                    dbclient: `pymongo.MongoClient`
                        pymongo client that manages the PS1 calibrators databae.
                        This is passed to extcats CatalogQuery object.
            """
            
            self.logger.info("matching objtable entries to PS1 calibrator stars")
            self.logger.info("using %s, %s as coordinates and a search radius of %.2f arcsec"%(
                xname, yname, rs_arcsec))
            
            # initialize the catalog query object and grab the database
            ps1cal_query = CatalogQuery.CatalogQuery(
                'ps1cal', 'ra', 'dec', dbclient = dbclient, logger = self.logger)
            
            # gather the found counterparts into a list of dictionary
            ps1cps = []
            if use_clusters:
                self.logger.info("Matching cluster centroids with the PS1 calibrators")
                av_x, av_y = self.compute_cluster_centroid(xname, yname)
                
                # search cp for each cluster centroid
                for icl, grp in tqdm.tqdm(self.gdf):
                    ps1match =  ps1cal_query.findclosest(ra = av_x[icl], dec = av_y[icl], 
                        rs_arcsec = rs_arcsec, method = 'healpix')
                    
                    # if you have a cp, add it to the list
                    if ps1match != (None, None):
                        buff = {}
                        for c in ps1match[0].colnames:
                            if (not col2rm is None) and (c not in col2rm):     # remove unwanted columns
                                buff[c] = ps1match[0][c]
                        buff['dist2ps1'] = ps1match[1]
                        buff['clusterID'] = icl        # this will be used to join the df
                        ps1cps.append(buff)
                
                # merge the dataframe
                self.df = self.df.merge(pd.DataFrame(ps1cps), on = 'clusterID', how='left')
                
                # if requested, drop sources without PS1cp
                if clean_non_matches:
                    print ("jackancanckanca", len(self.df))
                    self.df.dropna(subset = ['dist2ps1'], inplace = True)
                    self.logger.info("dropping sources without match in PS1cal DB: %d retained."%
                        (len(self.df)))
                self.logger.info("Created dataframe with PS1 calibrators for %d clusters"%
                    (len(ps1cps)))
            else:
                raise NotImplementedError("matching based on field / rc not fully implemeted yet.") # TODO
                # if you have the field & RC id for every object, use them if possible.
                dfcols = self.df.columns.tolist()
                if 'FIELDID' in dfcols and 'RCID' in dfcols:
                    self.logger.info(
                    "Matching source by sources using FIELDID and RC information")
                    
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
                        pass
#                        #fields = [246, 247, 248]
#                        #rcids = [38, 55]
#                        # get the coordinates of all the calibrators for those fields
#                        ps1srcs = [src for src in ps1cal_query.src_coll.find(
#                            {'field': { "$in": fields}, 'rcid': {'$in': rcids}},
#                            { '_id': 1, 'ra': 1, 'dec': 1 })]
##                        # match the object by coordinates: 
##                        # first find the closest calibrator for each object
##                        coords = SkyCoord(self.tab[xname], self.tab[yname], **skycoords_kwargs)
##                        ps1coords = SkyCoord(
##                            ra = ps1cals['ra']*u.deg, dec = ps1cals['dec']*u.deg, **PS1skycoords_kwargs)
##                        idps1, d2ps1, _  =  coords.match_to_catalog_sky(ps1coords)
##                        
##                        # add these indexes and distances to the table
##                        newcols = [Column(name = "PS1_ID", dtype = int, data = idps1),
##                                 Column(name = "D2PS1", data = d2ps1.to('arcsec').value, unit = 'arcsec')]
##                        self.tab.add_columns(newcols)
##                        
##                        # then select the matches that are more apart than sep
##                        matches = self.tab[self.tab['D2PS1']<sep.to('arcsec').value]
##                        logging.info("%d objects are within %.2f arcsec to a PS1 calibrator"%(
##                            len(matches), sep.to('arcsec').value))
##                        
##                        # finally group them accoring to the PS1 id.
##                        cals = matches.group_by('PS1_ID')
##                        
##                        # check that, no more than one object per dataset is
##                        # associated to any PS1 calibrator
##                        if np.any(np.diff(cals.indices)>len(self.hdf5paths)):
##                            logging.warning(
##                                "some PS1 calibrator has more than %d matches."%len(self.hdf5paths))
##                            logging.warning("there are probably problems with the matching.")
                else:
                    self.logger.info(
                    "FIELDID and/or RCID missing from objtable dataframe. Matching source by source.")
                    
                    # loop over the df rows
                    # find the match
                    # create the df
                    
                    pass

            
