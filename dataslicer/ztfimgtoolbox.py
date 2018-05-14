# collection of functions to help working with ZTF images.

import glob, os, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
import scipy.ndimage as ndimage
from astropy.io import fits

def rqid(ccd, q):
    """
        given the ccd ID number (from 1 to 16)
        and the CCD-wise readout channel number (from 1 to 4)
        compute the ID of that readout quadrant (from 1 to 64)
    """
    if type(ccd) in [str, float, int]:
        ccd=int(ccd)
    else:
        ccd=ccd.astype(int)
    if type(q) in [str, float, int]:
        q=int(q)
    else:
        q=q.astype(int)
    rqid=(ccd-1)*4 + q - 1
    return rqid

def ccdqid(rc):
    """
        given the readout quadrant ID (0 to 63), 
        computes the ccd (1 to 16) and quadrant (1 to 4) ID
    """
    
    ccd_id = rc//4 + 1
    q = rc%4 + 1
#    print ("RC %d: CCD: %d, q: %d"%(rc, ccd_id, q))
    return ccd_id, q

def getaxesxy(ccd, q):
    """
        given the ccd number (from 1 to 16) and the 
        CCD-wise readout channel number (from 1 to 4), 
        this function return the x-y position of this 
        readout quadrant on the 8 x 8 grid of the full ZTF field.
    """
    
    yplot=7-( 2*((ccd-1)//4) + 1*(q==1 or q==2) )
    xplot=2*( 4-(ccd-1)%4)-1 - 1*(q==2 or q==3) 
    return int(xplot), int(yplot)


def rm_edge(img, plotname = None):
    """
        given a flat field image remove the brigth edge.
        
        Parameters:
        -----------
        
            img: `numpy.array`
                the flat image you start with.
            
            plotname: `str` or None
                if not None, produce a diagnostic plot and save it there.
        
        Returns:
        -------
            
            image without the edge feature.
    """
    from scipy.stats.mstats import winsorize
    wins = winsorize(img, limits=(None, 0.05), axis = 1)
    if not plotname is None:
        fig, axes = plt.subplots(1, 3, figsize = (10, 3))
        ax = axes[0]
        imshow_args = {'origin': 'lower', 'aspect': 'auto'}
        im = ax.imshow(img, norm = ImageNormalize(img, interval=ZScaleInterval()), **imshow_args)
        ax.set_title("Original")
        plt.colorbar(im, ax = ax)
        
        ax = axes[1]
        imshow_args = {'origin': 'lower', 'aspect': 'auto'}
        im = ax.imshow(wins, norm = ImageNormalize(wins, interval=ZScaleInterval()), **imshow_args)
        ax.set_title("Winsorized")
        plt.colorbar(im, ax = ax)
        
        ax = axes[2]
        diff = img - wins
        imshow_args = {'origin': 'lower', 'aspect': 'auto'}
        im = ax.imshow(diff, norm = ImageNormalize(diff, interval=ZScaleInterval()), **imshow_args)
        ax.set_title("Original - Winsorized")
        plt.colorbar(im, ax = ax)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(plotname)
        plt.close(fig)
    
    return wins




def get_dust_shapes(img, gsmooth_std = 50, gsmooth_trunc = 2, perc_th = 1, 
    opening_disk_rad = 5, closing_disk_rad = 15, rotate = True, plotname = None):
    """
        given a flat field image, apply a combination of filters
        that will isolate dust grains.
        
        The idea is to compute the difference between the original image
        and a smoothed version, obtained with a large gaussian kernel.
        Prior to smoothing, a median filter with a small orizonthal kernel
        is applied in order to remove the hot stripes.
        
        The difference between the smoothed image and the raw one will contain 
        all the artifacts, including dust. To isolate the dust we then proceed
        to thresholding the image and then opening/closing this binary mask
        to remove high frequency noise. 
        
        Finally, we apply a blob_finding algorithm to get the center and radius 
        of the dust. To summarize, these are the steps:
        
            * median filter to remove hot stripes
            * gaussian_filter to smooth noise
            * subtract smoothed & filtered from raw image and take abs
            * threshold at 30% percentile
            * opening & closing
            * blob finding
        
        Parameters:
        -----------
        
            img: `numpy.array`
                the flat image you start with.
            
            gsmooth_std: `float` or `tuple`
                std of the gaussian kernel used for smoothing.
            
            gsmooth_trunc: `float`
                after how many std you want to truncate the gaussian kernel
            
            perc_th: `float`
                percentile level (between 0 and 100) to use for thresholding the image.
                Everything below will be set to 1.
            
            opening[closing]_disk_rad: `int`
                radius of the disk-shaped structuring element used for closing/opening
                the thresholded image. The opeing radius should be a bit smaller than 
                the size of the smaller dust grains you want to resolve, while the 
                closing one should be a bit larger.
            
            rotate: `bool`
                weather or not the images (and all the subsequent processing) should
                be rotated by 180 deg. Default is False.
            
            plotname: `str` or None
                if not None, produce a diagnostic plot and save it there.
            
            bobs_doh_kwargs: `kwargs`
                arguments to be passed to skimage.feature.blob_doh.
    """
    import time
    from scipy.stats.mstats import winsorize
    from skimage.morphology import opening, closing, disk, square, erosion, dilation
    from skimage.feature import blob_dog, blob_log, blob_doh
    
    if rotate:
        print ("rotating the image by 180 deg.")
        img  = np.rot90(img, 2)
#    img = img[:1000, :1000]
    
    # remove the bright edge
    no_edge = rm_edge(img)
    
    # first remove the hot stripes with median filter and then 
    # smooth the filtered one with a broad-truncated gaussian kernel
    print ("Filtering and smoothing the image..", end="", flush=True)
    start = time.time()
    no_stripes = ndimage.median_filter(no_edge, size = (1, 5))
    smooth = ndimage.gaussian_filter(
        no_stripes, sigma=gsmooth_std, truncate=gsmooth_trunc, order=0)
    end = time.time()
    print ("done, took: %.2e sec"%(end-start))

    # now compute the difference and it's absolute value.
    diff = no_edge - smooth
    smooth_diff = ndimage.gaussian_filter(diff, sigma=1, truncate = 2, order=0)
    
    # threshold the image
    th = np.percentile(smooth_diff, 1.)
    thresh =  np.copy(smooth_diff)
    thresh[thresh>th] = 0
    thresh[thresh<th] = 1
    
    # first remove bright pixels (opening) and then fill holes in bright spots
    print ("Opening and closing the image..", end="", flush=True)
    start = time.time()
    eroded = opening(thresh, disk(opening_disk_rad))
    dilated = closing(eroded, disk(closing_disk_rad))
    binary =  dilated
    end = time.time()
    print ("done, took: %.2e sec"%(end-start))
    
    # dinally fin the blobs
    print ("Finding blobs..", end="", flush=True)
    start = time.time()
    blobs = blob_doh(binary, min_sigma = 20, max_sigma = 500, 
        overlap = 0.5, num_sigma=20, log_scale = False)
    end = time.time()
    print ("done, took: %.2e sec"%(end-start))
    
    if not plotname is None:
        # compute the differences
        stripes =  img - no_stripes
        
        fig, axes = plt.subplots(2, 4, figsize = (15, 15), sharex = True, sharey = True)
        axes = axes.flatten()
        imshow_args = {'origin': 'lower', 'aspect': 'auto'}
        norm=ImageNormalize(img, interval=ZScaleInterval())
        
        # plot the raw
        ax = axes[0]
        im = ax.imshow(img, norm = norm, **imshow_args)
        ax.set_title("Original")
        plt.colorbar(im, ax = ax)
        
        # plot the one without the edge
        ax = axes[1]
        im = ax.imshow(no_edge, norm = ImageNormalize(no_edge, interval=ZScaleInterval()), **imshow_args)
        ax.set_title("No Edge")
        plt.colorbar(im, ax = ax)
        
        # plot the difference
        ax = axes[2]
        norm = ImageNormalize(diff, interval=ZScaleInterval())
        im = ax.imshow(diff, norm = norm, **imshow_args)
        ax.set_title("Original - Processed")
        plt.colorbar(im, ax = ax)
        
        # plot the smoothed abs difference
        ax = axes[3]
        norm = ImageNormalize(smooth_diff, interval=ZScaleInterval())
        im = ax.imshow(smooth_diff, norm = norm, **imshow_args)
        ax.set_title("Smoothed diff")
        plt.colorbar(im, ax = ax)
        
        # plot the thresholded one
        ax = axes[4]
        im = ax.imshow(thresh, cmap = 'binary', **imshow_args)
        ax.set_title("Thresholded")
        plt.colorbar(im, ax = ax)
        
        # plot the eroded one
        ax = axes[5]
        im = ax.imshow(eroded, cmap = 'binary', **imshow_args)
        ax.set_title("Eroded")
        plt.colorbar(im, ax = ax)
        
        # plot the closed one
        ax = axes[6]
        im = ax.imshow(dilated, cmap = 'binary', **imshow_args)
        ax.set_title("Eroded + dilated")
        plt.colorbar(im, ax = ax)
        
        # plot dust the
        ax = axes[7]
        im = ax.imshow(binary, cmap = 'binary', **imshow_args)
        ax.set_title("Binary mask")
        plt.colorbar(im, ax = ax)
        for b in blobs:
            y, x, r = b
            c = plt.Circle((x, y), r, color='r', linewidth=1.5, fill=False)
            ax.add_patch(c)
#            axes[3].add_patch(c)
        fig.tight_layout()
        fig.show()
        exit()
        fig.savefig(plotname)
        plt.close()
    
    # return a dataframe
    buff = []
    for b in blobs:
        y, x, r = b
        buff.append({'x': x, 'y': y, 'r': r})
    df = pd.DataFrame(buff)
    return df


def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)


def get_dust_shapes_fft(img, gsmooth_std = 150, gsmooth_trunc = 2, perc_th = 40, 
    opening_disk_rad = 5, closing_disk_rad = 15, rotate = True, plotname = None):
    """
        given a flat field image, apply a combination of filters
        that will isolate dust grains.
        
        The idea is to compute the difference between the original image
        and a smoothed version, obtained with a large gaussian kernel.
        Prior to smoothing, a median filter with a small orizonthal kernel
        is applied in order to remove the hot stripes.
        
        The difference between the smoothed image and the raw one will contain 
        all the artifacts, including dust. To isolate the dust we then proceed
        to thresholding the image and then opening/closing this binary mask
        to remove high frequency noise. 
        
        Finally, we apply a blob_finding algorithm to get the center and radius 
        of the dust. To summarize, these are the steps:
        
            * median filter to remove hot stripes
            * gaussian_filter to smooth noise
            * subtract smoothed & filtered from raw image and take abs
            * threshold at 30% percentile
            * opening & closing
            * blob finding
        
        Parameters:
        -----------
        
            img: `numpy.array`
                the flat image you start with.
            
            gsmooth_std: `float` or `tuple`
                std of the gaussian kernel used for smoothing.
            
            gsmooth_trunc: `float`
                after how many std you want to truncate the gaussian kernel
            
            perc_th: `float`
                percentile level (between 0 and 100) to use for thresholding the image.
                Keep it low (default is 30%), since the noise will be removed by opening
                and closing.
            
            opening[closing]_disk_rad: `int`
                radius of the disk-shaped structuring element used for closing/opening
                the thresholded image. The opeing radius should be a bit smaller than 
                the size of the smaller dust grains you want to resolve, while the 
                closing one should be a bit larger.
            
            rotate: `bool`
                weather or not the images (and all the subsequent processing) should
                be rotated by 180 deg. Default is False.
            
            plotname: `str` or None
                if not None, produce a diagnostic plot and save it there.
            
            bobs_doh_kwargs: `kwargs`
                arguments to be passed to skimage.feature.blob_doh.
    """
    import time
    from skimage.morphology import opening, closing, disk
    from skimage.feature import blob_dog, blob_log, blob_doh
    
    if rotate:
        print ("rotating the image by 180 deg.")
        img  = np.rot90(img, 2)
    
    # compute the fft and shift it
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)
    
    # use gaussian filter
    ncols, nrows = img.shape
    sigmax, sigmay = 10, 10
    cy, cx = nrows/2, ncols/2
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    gmask = gmask/np.sum(gmask)
    
    fft_filter = fft*gmask
    filtered_image = np.abs(np.fft.ifft2(fft_filter))
    
    fig, axes = plt.subplots(2, 2, sharex = True, sharey = True)
    axes = axes.flatten()
    imshow_args = {'origin': 'lower', 'aspect': 'auto'}
    norm=ImageNormalize(img, interval=ZScaleInterval())
    
    ax = axes[0]
    ax.imshow(img, norm = norm, **imshow_args)
    ax.set_title("Original")
    
    ax = axes[1]
    ax.imshow(np.log10(abs(fft)), **imshow_args)
    ax.set_title("FFT")
    
    ax = axes[2]
    ax.imshow(filtered_image, **imshow_args)
    ax.set_title("Filtered Image")
    
    ax = axes[3]
    ax.imshow(np.log10(abs(fft_filter)), **imshow_args)
    ax.set_title("Filtered FFT")
    
    
    
    
    plt.show()
    exit()
#    f_ishift = np.fft.ifftshift(fshift1)  #inverse shift
#    img_back = np.fft.ifft2(f_ishift)     #inverse fourier transform
#    img_back = np.abs(img_back)
#    
    
    
    
    # first remove the hot stripes with median filter and then 
    # smooth the filtered one with a broad-truncated gaussian kernel
    print ("Filtering and smoothing the image..", end="", flush=True)
    start = time.time()
    no_stripes = ndimage.median_filter(img, size = (1, 5))
    smooth = ndimage.gaussian_filter(
        no_stripes, sigma=gsmooth_std, truncate=gsmooth_trunc, order=0)
    end = time.time()
    print ("done, took: %.2e sec"%(end-start))

    # now compute the difference and it's absolute value
    diff = img - smooth
    dust = abs(diff)
    
    # threshold the image
    thresh = np.percentile(dust, perc_th)
    dust[dust<thresh] = 0
    dust[dust>thresh] = 1
    
    # first remove bright pixels (opening) and then fill holes in bright spots
    print ("Opening and closing the image..", end="", flush=True)
    start = time.time()
    binary = opening(dust, disk(opening_disk_rad))
    binary = closing(binary, disk(closing_disk_rad))
    end = time.time()
    print ("done, took: %.2e sec"%(end-start))
    
    # dinally fin the blobs
    print ("Finding blobs..", end="", flush=True)
    start = time.time()
    blobs = blob_doh(binary, min_sigma = 15, max_sigma = 2000, overlap = 0.5, num_sigma=20, log_scale = True)
    end = time.time()
    print ("done, took: %.2e sec"%(end-start))
    
    if not plotname is None:
        # compute the differences
        stripes =  img - no_stripes
        
        fig, axes = plt.subplots(2, 2, figsize = (15, 15), sharex = True, sharey = True)
        axes = axes.flatten()
        imshow_args = {'origin': 'lower', 'aspect': 'auto'}
        norm=ImageNormalize(img, interval=ZScaleInterval())
        
        # plot the raw
        ax = axes[0]
        im = ax.imshow(img, norm = norm, **imshow_args)
        ax.set_title("Original")
        plt.colorbar(im, ax = ax)
        
        # plot the difference
        ax = axes[1]
        norm = ImageNormalize(diff, interval=ZScaleInterval())
        im = ax.imshow(diff, norm = norm, **imshow_args)
        ax.set_title("Original - filtered & smoothed")
        plt.colorbar(im, ax = ax)
        
        # plot the thresholded one
        ax = axes[2]
        im = ax.imshow(dust, norm = norm, **imshow_args)
        ax.set_title("Thresholded")
        plt.colorbar(im, ax = ax)
        
        # plot the dust
        ax = axes[3]
        norm = ImageNormalize(binary, interval=ZScaleInterval())
        im = ax.imshow(binary, **imshow_args)
        ax.set_title("Binary mask")
        plt.colorbar(im, ax = ax)
        for b in blobs:
            y, x, r = b
            c = plt.Circle((x, y), r, color='r', linewidth=1.5, fill=False)
            ax.add_patch(c)
        fig.tight_layout()
        fig.savefig(plotname)
        plt.close()
    
    # return a dataframe
    buff = []
    for b in blobs:
        y, x, r = b
        buff.append({'x': x, 'y': y, 'r': r})
    df = pd.DataFrame(buff)
    return df



def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    taken from:
    https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    print (compression_pairs)
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def combineimages(files, outfile, plot=True, gapX = 462, gapY = 645, 
    fill_value = 0, newshape=None, overwrite=False):  #, newshape=(616, 512))
    """ 
        Combine CCD images into a full frame and save the output to a file. 
        Adapted from Matthew's combine_cal_files
  
        Parameters
        ----------
        files: `list`
            list of 16 fits files (one for ccd), that has to be combined together.
            IMPORTANT: this list has to be sorted according to ccd ID, otherwise the 
            trick won't work!
        outfile: `str`
            name of the output file that will be produced.
        plot: `bool`
            if true, a plot will be created. Only works if newshape is not None, 
            that is, if the data has been downsampled.
        gapX : int
            The separation between CCDs in the x (RA) direction
        gapY : int
            The separation between CCDs in the y (Dec) direction
        fill_value : int
            The value for pixels in the CCD gaps
        newshape : tuple
            if not None, the data array is downsampled to the newshape using bin_ndarray.
    """
    
    # here is the modified Matthew's function
    if not os.path.isfile(outfile) or overwrite:
        print ("combinig images...")
        rows = []
        for ccdrow in tqdm.tqdm(range(4)):
            for qrow in range(2):
                chunks = []
                for ccd in range(4): 
                    hdulist = fits.open(files[4 * ccdrow + (4 - ccd) - 1])
                    if qrow == 0:
                        img_data_1 = hdulist[3].data   
                        img_data_2 = hdulist[4].data
                    else:
                        img_data_1 = hdulist[2].data
                        img_data_2 = hdulist[1].data
                    
                    if not newshape is None:
                        img_data_1=bin_ndarray(img_data_1, newshape, 'mean')
                        img_data_2=bin_ndarray(img_data_2, newshape, 'mean')
                    
                    # Rotate by 180 degrees
                    img_data_1 = np.rot90(img_data_1, 2)
                    img_data_2 = np.rot90(img_data_2, 2)
                    x_gap = np.zeros((img_data_1.shape[0], gapX)) + fill_value
                    chunks.append(np.hstack((img_data_1, img_data_2)))
                row_data = np.hstack((chunks[0], x_gap, chunks[1], x_gap, chunks[2], x_gap, chunks[3]))
                rows.append(row_data)
            if ccdrow < 3: rows.append(np.zeros((gapY, row_data.shape[1])) + fill_value)
        array = np.vstack(rows)
        fits.writeto(
            outfile, array, header = fits.getheader(files[0], 0), overwrite=overwrite)
        print ("combined image saved to: %s"%outfile)

    # eventually plot it
    if (plot is True) and os.path.isfile(outfile):  #not (newshape is None) 
    
        fig, ax=plt.subplots(figsize=(10, 10))
        try:
            norm=ImageNormalize(array, interval=ZScaleInterval())
        except UnboundLocalError:
            array=fits.getdata(outfile, 0)
            norm=ImageNormalize(array, interval=ZScaleInterval())
        im=ax.imshow(
            array, origin='lower', aspect='auto', 
            cmap='nipy_spectral', norm=norm, 
            interpolation='none')
        cb=plt.colorbar(im, ax=ax)
        cb.set_label("mean $\Delta$t [s] in each pixel group")
        ax.set_title(outfile.split("/")[-1].replace(".fits.fz", ""))
        outplot=outfile.replace(".fits", "").replace(".fz", "")+".png"
        fig.savefig(outplot)
        print ("plot saved to: %s"%outplot)


def plot_ccd_image(ccdfile, outfile, rotate = True, cmap = "Greys", clabel = ""):
    """
        make a plot with the 4 rc in one raw image file
    """
    
    # read in the data and eventually rotate it
    data = []
    for irc in range(1, 5):
        if rotate:
            rc_img = np.rot90(fits.getdata(ccdfile, irc), 2)
        else:
            rc_img = fits.getdata(ccdfile, irc)
        data.append(np.float32(rc_img))
    
    # prepare the plot
    fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(8, 8), sharex=True, sharey=True)
    
    # combine all the data to have a global scaling
    flatdata = np.array(data).flatten()
    norm=ImageNormalize(flatdata, interval=ZScaleInterval())
    
    # plot 
    for irc in range(1, 5):
        # pick the right axes
        if irc==1:
            ax=ax1
        elif irc==2:
            ax=ax2
        elif irc==3:
            ax=ax3
        else:
            ax=ax4
        
        im=ax.imshow(data[irc -1], origin='lower', norm=norm, aspect='auto', cmap = cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        
    # add colorbar and save plot
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    cb=fig.colorbar(im, ax=[ax1, ax2, ax3, ax4], pad=0.02)
    cb.set_label(clabel)
    fig.savefig(outfile)
    print ("plot saved to:", outfile)
    plt.close(fig)


def check_rawfile_list(filelist, ngroups, pattern, nfiles_4group, start_at0 = False):
    """
        given a list of files, check that they represent an integer
        number of ZTF exposures  by counting them and looking for matches
        of the file naming scheme.
        
        Parameters:
        -----------
            
            filelist: `list`
                list of fits files to check
            
            ngroups: `int`
                number of file groups, e.g. 16 for CCD groups, 64 for RQ groupings.
            
            pattern: `str`
                template string to match the files to and assign them to the groups.
                it has to be possible to format it with a single int parameter. E.g.:
                    "c%02d", "rq%02", ecc.
            
            nfiles_4group: `int`
                number of files in each of the groups defined by the pattern
            
            start_at0: `bool`
                specifies if int identifying the file groups starts at zero (for RC)
                or at 1 (for CCD)
    """
    start = 1 if start_at0 else 0
    end = ngroups + start
    fgroups = {}
    for igroup in range(start, end):
        files = [f for f in filelist if pattern%igroup in f]
        if len(files) != nfiles_4group:
            raise RuntimeError("there should be %s files with %s in name. Got %d instead."
                %(pattern%igroup, len(files)))
        fgroups[pattern%igroup] = files
    return fgroups


def split_in_rc(infile, outfile_tmpl = None, overwrite = False, 
        dtype  = 'float32', rm_original = False, compress = True, **writeto_kwargs):
    """
        split a ZTF raw image file (CCD wise) into the 4 files correspoding to the 
        readout channels.  
    
        Parameters:
        -----------
        
            infile: `str`
                path to the input CCD, raw image file.
            
            outile_tmpl: `str` 
                name template for the output files. It has to contain a sequence 
                that can be formatted via a single integer, specifiyng the readout channel.
                If None defaults to infile+'_rc%02d_'.
            
            overwrite: `bool`
                self explaining
            
            dtype: `str` or built-in data type
                data type to case the images to
            
            rm_original: `bool`
                weather or not the original file has to be removed.
            
            compress: `bool`
                if True, write the file as a CompressedHDU, docs at:
                http://docs.astropy.org/en/stable/io/fits/api/images.html#astropy.io.fits.CompImageHDU
            
            writeto_kwargs: `astropy.io.fits.writeto kwargs`
                additional arguments to pass the astropy.io.fits.writeto method
            
    """
    
    if outfile_tmpl is None:
        pieces = os.path.basename(infile).split(".fits")
        pieces.insert(len(pieces)-1, "_rc%02d.fits")
        outfile_tmpl =  os.path.join(os.path.dirname(infile), "".join(pieces))
    
    hudl = fits.open(infile)
    iccd = hudl[0].header['CCD_ID']
    for irq in range(1, 5):
        rcid =  rqid(iccd, irq)
        outfile = outfile_tmpl%rcid
        if os.path.isfile(outfile) and not overwrite:
            continue
        hudl[irq].header['EXTNAME'] = str(1)
        if compress:
            chdu = fits.CompImageHDU(hudl[irq].data.astype(dtype), hudl[irq].header)
            chdu.writeto(outfile, overwrite = overwrite, **writeto_kwargs)
        else:
            fits.writeto(
            outfile, 
            data = hudl[irq].data.astype(dtype), 
            header = hudl[irq].header, overwrite = overwrite, **writeto_kwargs)
            
    hudl.close()
    if rm_original:
        os.remove(infile)
    

def split4rc(infiles, outdir, outfile_tmpl, **split_in_rc_args):
    """
        given a list of 16 files arranged for CCD, split them into 64 readout
        quadrant wise files.  
    
        Parameters:
        -----------
        
            infiles: `list`
                list of files names to CCD wise fits files each containing 4 separate image
                extensions for each readout quadrant. The list have to be sorted according
                to the CCD ID number.
            
            outdir: `str`
                path to directory where the RC wise files have to be saved.
            
            outile_tmpl: `str`
                name template for the output files. It has to contain the sequence 'RCID%02d'.
            
            split_in_rc_args: `kwargs`
                additional arguments passed to split_in_rc.
            
    """
    
#    check_rawfile_list(
#        infiles, ngroups = 16, pattern = "c%02d", nfiles_4group = 1, start_at0 = True)
    
    outfile_tmpl = os.path.join(outdir, outfile_tmpl)
    for iccd in tqdm.tqdm(range(0, 16)):
        split_in_rc(infiles[iccd], outfile_tmpl, split_in_rc_args)
        
