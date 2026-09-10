import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits
import sunpy.map as smap
import astropy.units as u
from astropy.coordinates import SkyCoord

# generall codes
def readfits(f, ext=0, xr=None, yr=None):
    with fits.open(f) as hdul:
        if xr is not None and yr is not None:
            data = hdul[ext].section[yr[0]:yr[1],xr[0]:xr[1]]
        elif xr is not None:
            data = hdul[ext].section[:,xr[0]:xr[1]]
        elif yr is not None:
            data = hdul[ext].section[yr[0]:yr[1],:]
        else:
            data = hdul[ext].data
        hdr = hdul[ext].header
    hdul.close()
    return data, hdr

def removenan(img0, missing=0.1):
    img = np.zeros_like(img0, dtype=np.float32)+missing
    idx = np.isfinite(img0)
    img[idx] = img0[idx]
    return img

def norm0_1(x, vmin=None, vmax=None):
    if vmin is None: vmin=x.min()
    if vmax is None: vmax=x.max()
    y = (x-vmin)/(vmax-vmin)
    y[y<0] = 0
    y[y>1] = 1
    return y

def norm8(im0,vmin=None,vmax=None,log=None):
    '''
    Normalize an image (np.array type) to uint8 datatype.
    '''
    if vmin is None: vmin=im0.min()
    if vmax is None: vmax=im0.max()
    im = removenan(im0)
    im[im<vmin] = vmin
    im[im>vmax] = vmax
    if log is not None:
        if vmin<0:
            im2 = im.copy()
            idx = im<0
            tmp = im[idx]
            im2[idx] = -np.log10(-im2[idx])
            im2[im>0] = np.log10(im2[im>0])
            vmin = -np.log10(-vmin)
            vmax = np.log10(vmax)
            im = im2.copy()
        else:
            im[im<1.e-6] = 1.e-6
            im = np.log10(im)
            vmin = im.min()
            vmax = np.log10(vmax)
    im = (im-vmin)/(vmax-vmin)*255
    return np.uint8(im)

def dist_circle(n, xcen=None, ycen=None, dtype=None):
    # Create an array where each value is its distance to a given center.
    if isinstance(n, (int, float, np.number)):
        n = [n, n]
    yy,xx=np.mgrid[0:n[0],0:n[1]]
    if xcen is None:
        xcen = (n[0]-1)/2.
    if ycen is None:
        ycen = (n[1]-1)/2.
    rr = np.sqrt((xx-xcen)**2.+(yy-ycen)**2.)
    if dtype is not None:
        rr = np.array(rr, dtype=dtype)
    return rr

def getMapEdge(m0):
    # for lasco, m0.bottom_left_coord may give a wrong result
    ny, nx, dy, dx, xc, yc = m0.meta['naxis2'], m0.meta['naxis1'], m0.meta['cdelt2'], m0.meta['cdelt1'], m0.meta['crpix1']-0.5, m0.meta['crpix2']-0.5
    return [-xc*dx,(nx-xc)*dx,-yc*dy,(ny-yc)*dy]

def immove(image, dy, dx):
    import scipy.fftpack as fft
    from scipy.ndimage import fourier_shift
    if dx == 0 and dy == 0:
        offset_image = image
    else:
        shift = (dy, dx)
        offset_image = fourier_shift(fft.fft2(image), shift)
        offset_image = np.real(fft.ifft2(offset_image))
    return offset_image

def imrot(im,ang,center=None):
    # ang: in degree, clockwise
    # center: (pix_x,pix_y). Default is the center of the image
    from PIL import Image
    im0 = Image.fromarray(im)
    im1 = np.array(im0.rotate(ang, center=center))
    return im1

def immove_rot(im0,dy,dx,ang):
    '''
    Shift an image in subpix then rotate
    
    Parameters:
    -----------
    dy, dx: float, in pixel.
    ang: float, in degree. Clockwise.
    '''
    from PIL import Image # rotation
    im0b = immove(im0,dy,dx)
    im1 = Image.fromarray(im0b)
    im1 = im1.rotate(ang) # in degree
    im1 = np.array(im1)
    return im1

def mapUpRecenter(m0):
    # for LASCO and SCIUV, m0.rotate(recenter=True, missing=1) changes the data dimension, which is not what we want.  
    if m0.meta['crval1'] != 0 or m0.meta['crval2'] != 0:
        print("m0.meta['crval1'] or m0.meta['crval2'] is not 0")
        return -1
    ny, nx = m0.data.shape
    dy = (ny+1)/2-m0.meta['crpix2']
    dx = (nx+1)/2-m0.meta['crpix1']
    imb = immove_rot(m0.data, dy, dx, -m0.meta['crota2'])
    hdr = m0.meta
    hdr['crpix1'] = (nx+1)/2
    hdr['crpix2'] = (ny+1)/2
    hdr['pc1_1'] = 1; hdr['pc1_2'] = 0; hdr['pc2_1'] = 0; hdr['pc2_2'] = 1
    hdr['crota2'] = 0
    try:
        del hdr['crota1']
    except:
        pass
    m1 = smap.Map(imb,hdr)
    return m1

def iris_despike(im, int_lim=30, medratio=20, n_isolate=9):
    '''
    Purpose: 
        remove bright pixels caused by energetic particles for IRIS raster image. The pixels with spikes are replaced
        with nearby pixels at the same wavelength.
    Input:
        im: input image.
        int_lim: if intensities are larger than this value (intensity limit), these pixels will be candidate with spikes.
        medratio: pixels whose ratios of original values to median-filtered values larger than medratio are treated as spikes.
        n_isolate: if the number of connected pixels is less than that value, they will be treated as spikes.
    Output:
        im2: output image.
    example:
        # for a LASCO-C2 difference image im0
        im1 = iris_despike(im0,600,20,12)
        im2 = -iris_despike(-im1,600,20,12)
    Possible improvement:
        use local median value to replace int_lim.
    History:
        2025-6-16: add a line before returning: im2[im2<-10] = 0. Because extraordinary negative values were found.
    '''
    import skimage.morphology as sm
    from skimage.measure import label
    from scipy.ndimage import median_filter
    ind = np.nonzero(im>-100)
    iy0 = np.min(ind[0])
    iyt = np.max(ind[0])
    ix0 = np.min(ind[1])
    ixt = np.max(ind[1])
    imc = im[iy0:iyt+1, ix0:ixt+1].copy()
    ny, nx = imc.shape
    # criterion 1
    im0 = imc.copy()
    im0[im0<0.1] = 0.1
    #imm = sf.median(im0, sm.disk(9))
    imm = np.abs(median_filter(im0, size=7))
    tmp = int_lim/medratio/2
    imm[imm<tmp] = tmp
    imr = im0/imm
    simg1 = np.zeros_like(imr, dtype=np.int8)
    simg1[np.logical_and(imr>medratio, im0>int_lim)] = 1
    im0 = imc.copy()
    labelf = label(simg1,connectivity=2, background=0)
    for i in range(np.max(labelf)):
        ind = np.nonzero(labelf == i+1)
        iymax = np.max(ind[0])
        iymin = np.min(ind[0])
        if iymin == 0:
            #print(ind[0], '\n', ind[1])
            #return
            for i in range(ind[0].size):
                im0[ind[0][i], ind[1][i]] = imc[iymax+1, ind[1][i]]
        elif iymax == ny-1:
            for i in range(ind[0].size):
                im0[ind[0][i], ind[1][i]] = imc[iymin-1, ind[1][i]]
        else:
            iymid = (iymax+iymin)*0.5
            for i in range(ind[0].size):
                if ind[0][i] <= iymid:
                    im0[ind[0][i], ind[1][i]] = imc[iymin-1, ind[1][i]]
                else:
                    im0[ind[0][i], ind[1][i]] = imc[iymax+1, ind[1][i]]
    
    # criterion 2
    simg1 = np.zeros_like(im0, dtype=np.int8)
    simg1[im0>=int_lim] = 1
    simg2 = sm.closing(simg1, sm.footprint_rectangle((3, 3)))
    labelf = label(simg2,connectivity=2, background=0)
    nlabel = np.zeros(np.max(labelf)+1, dtype=np.uint32)
    for i in range(np.max(labelf)+1):
        ind = np.nonzero(labelf == i)
        nlabel[i] = len(ind[0])
    spilab = np.nonzero(nlabel<=n_isolate)
    im1 = im0.copy()
    for i in list(spilab[0]):
        ind = np.nonzero(labelf == i)
        iymax = np.max(ind[0])
        iymin = np.min(ind[0])
        if iymin == 0:
            for i in range(ind[0].size):
                im1[ind[0][i], ind[1][i]] = im0[iymax+1, ind[1][i]]
        elif iymax == ny-1:
            for i in range(ind[0].size):
                im1[ind[0][i], ind[1][i]] = im0[iymin-1, ind[1][i]]
        else:
            iymid = (iymax+iymin)*0.5
            for i in range(ind[0].size):
                if ind[0][i] <= iymid:
                    im1[ind[0][i], ind[1][i]] = im0[iymin-1, ind[1][i]]
                else:
                    im1[ind[0][i], ind[1][i]] = im0[iymax+1, ind[1][i]]
    im2 = im.copy()
    im2[iy0:iyt+1, ix0:ixt+1] = im1
    # im2[im2<-10] = 0
    return im2

# SCIUV codes
def scihdr(hdr0, cdelt=2.4, crvals=[0,0], crpixs=[1024.5,1029.5], rotrad=-0.04, aiahdr=None):
    '''
    Purpose: (1) Regularize header for SCIUV data; (2) Adjust image position and rotation through fitsheader. The header is constructed from the original header and additional parameters.
    hdr0: `np.record`
        Header restored from published SCIUV file (.sav file during the commission phase). If the type is `astropy.io.fits.header.Header`, the rows before "hdr['naxis'] = 2" are not necessary.
    cdelt, crvals, crpixs, rotrad: `float` or `list`
        Parameters for image scaling, center coordinates, reference pixel, and rotation angle (in radians).
    aiahdr: `astropy.io.fits.header.Header` or None
        Header containing additional astronomical information.
    '''
    hdr = fits.Header()
    if type(hdr0) is fits.header.Header:
        names = hdr0.keys()
    else: # if type(hdr0) is np.record:
        names = hdr0.dtype.names
    for i in names:
        if i == 'COMMENT' or i == 'HISTORY':
            break
        if type(hdr0[i]) is bytes:
            hdr[i] = hdr0[i].decode()
        else:
            hdr[i] = hdr0[i]
    hdr['naxis'] = 2
    hdr['cunit1'] = 'arcsec'
    hdr['cunit2'] = 'arcsec'
    hdr['ctype1'] = 'HPLN-TAN'
    hdr['ctype2'] = 'HPLT-TAN'
    hdr['crval1'] = crvals[0]
    hdr['crval2'] = crvals[1]
    hdr['crpix1'] = crpixs[0]
    hdr['crpix2'] = crpixs[1]
    hdr['cdelt1'] = cdelt
    hdr['cdelt2'] = cdelt
    hdr['crota2'] = rotrad*180/np.pi
    hdr['pc1_1'] = hdr['pc2_2'] = np.cos(rotrad)
    hdr['pc1_2'] = -np.sin(rotrad)
    hdr['pc2_1'] = np.sin(rotrad)
    if aiahdr is not None:
        for i in ['r_sun', 'dsun_ref', 'dsun_obs', 'rsun_ref', 'rsun_obs', 'obs_vr', 'obs_vw', 'obs_vn', 'crln_obs', 'crlt_obs', 'hgln_obs', 'hglt_obs']:
            hdr[i] = aiahdr[i]
    return hdr

def scimap(img0, hdr, cmap=None, vmin=10, vmax=1500):
    # hdr = scihdr(hdr0)
    img = removenan(img0)
    img[img<1] = 1
    msci = smap.Map(img, hdr)
    if cmap is not None:
        msci.plot_settings['cmap'] = cmap
    if vmin is not None:
        msci.plot_settings['norm'] = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return msci

# LASCO codes
def lascoMap(f, fbkg=None, despike=False, int_lim=200, medratio=20, n_isolate=50, expNorm=False):
    # int_lim, medratio, are n_isolate are three parameters for despike.
    m0 = smap.Map(f)
    m1 = mapUpRecenter(m0)
    if expNorm is True:
        m1 = m1/m1.exposure_time
    if fbkg is not None:
        m0 = smap.Map(fbkg)
        mb = mapUpRecenter(m0)
        if expNorm is True:
            mb = mb/mb.exposure_time
        im0 = m1.data-mb.data
        if despike is True:
            # im0 = iris_despike(im0,600,20,n_isolate)
            im1 = iris_despike(im0,int_lim,medratio,n_isolate)
            im0 = -iris_despike(-im1,int_lim,medratio,n_isolate)
        m1 = smap.Map(im0, m1.meta)
    else:
        if despike is True:
            im0 = iris_despike(m1.data,int_lim,medratio,n_isolate)
            m1 = smap.Map(im0,m1.meta)
    return m1

def lascoMapb(f, fbkg=None, despike=False, int_lim=200, medratio=20, n_isolate=50, expNorm=False):
    # In lascoMap function, the target map f and bkg map fbkg are rotated firstly then subtracted. However, the different transformation parameters lead to residual errors in the difference image. 
    # In this function, the background is subtracted directly, then conduct northup and recenter.
    # int_lim, medratio, are n_isolate are three parameters for despike.
    m0 = smap.Map(f)
    if expNorm is True:
        m0 = m0/m0.exposure_time
    if fbkg is not None:
        mb0 = smap.Map(fbkg)
        if expNorm is True:
            mb0 = mb0/mb0.exposure_time
        m0b = smap.Map(m0.data - mb0.data, m0.meta)
        m1 = mapUpRecenter(m0b)
        if despike is True:
            im1 = iris_despike(m1.data,int_lim,medratio,n_isolate)
            im0 = -iris_despike(-im1,int_lim,medratio,n_isolate)
            m1 = smap.Map(im0, m1.meta)
    else:
        m1 = mapUpRecenter(m0)
        if despike is True:
            im0 = iris_despike(m1.data,int_lim,medratio,n_isolate)
            m1 = smap.Map(im0,m1.meta)
    return m1

def combineDisk_lasco(mdisk, mc2, vc2=[1,1500], vdisk=[1,200], rmask=2, cmapdisk=None, cmapc2=None, rsun_obs=969.8933):
    '''
    Combine disk and c2 maps to a new image, used for difference image.
    '''
    if abs(mdisk.meta['crota2'])>0.1 or abs(mc2.meta['crota2'])>0.1:
        print('Please make both maps NorthUp and recentered before using this code')
        return -1,-1
    nya, nxa = mdisk.data.shape
    nxn = int(nxa/(mc2.meta['cdelt1']/mdisk.scale.axis1.value))
    nyn = int(nya/(mc2.meta['cdelt2']/mdisk.scale.axis2.value))
    if nxn%2 == 1: nxn+=1
    if nyn%2 == 1: nyn+=1
    mdiskb = mdisk.resample(dimensions=(nxn, nyn)*u.pix) #
    # mc2b = mc2.rotate(recenter=True) # The input mc2 has been rotated
    nys, nxs = mc2.data.shape
    if nys<1024 or nxs<1024:
        print('mc2 should be complete images without cropping')
    imc2 = removenan(mc2.data)
    imdisk = removenan(mdiskb.data)
    imc2 = norm0_1(imc2, vc2[0], vc2[1])
    imdisk = norm0_1(imdisk, vdisk[0], vdisk[1])
    rrdisk = dist_circle([nyn, nxn])*mc2.meta['cdelt1']
    rrc2 = dist_circle([nys,nxs])*mc2.meta['cdelt1']
    idxdisk = rrdisk<rsun_obs*rmask
    idxc2 = rrc2<rsun_obs*rmask
    if cmapdisk is not None:
        imout = cmapc2(imc2)
        tmp = cmapdisk(imdisk)
        for i in range(4):
            tmpa = imout[:,:,i].copy()
            tmpb = tmp[:,:,i].copy()
            tmpa[idxc2] = tmpb[idxdisk]
            imout[:,:,i] = tmpa
        imout = imout.astype(np.float32)
    else:
        imout = imc2.astype(np.float32)
        imout[idxc2] = imdisk[idxdisk]
    edg = getMapEdge(mc2)
    return imout, edg

def combineDisk_SCI(mdisk, msci, vsci=[1,1500], vdisk=[1,200], logsci=True, logdisk=True, rmask=1.1, scaleSCI=2.4, cmapdisk=None, cmapsci=None):
    '''
    Combine disk and SCI maps to a new image.
    
    Parameters:
    -----------
    mdisk :  `sunpy.map.Map`
        Disk map.
    msci : `sunpy.map.Map`
        SCI map. All pixel numbers should be finite.
    vsci : `list` with 2 elements
        Minimum and maximum values of SCI data for plotting.
    vdisk : `list` with 2 elements
        Minimum and maximum values of disk data for plotting.
    log : `bool`
        Whether the output image is in log-scale.
    rmask : `float` in unit of solar radius.
        Boundary of disk and SCI image. Data of region beyond rmask are from SCI.
    cmapdisk: `matplotlib.colors.LinearSegmentedColormap`
        Color map of disk data. If both cmapdisk and cmapsci are set, the output image is a 3-D array of [ny, nx, 4].
    cmapsci: `matplotlib.colors.LinearSegmentedColormap`
        Color map of SCI data. If both cmapdisk and cmapsci are set, the output image is a 3-D array of [ny, nx, 4].
    
    Example:
    --------
        imc, edgs = combineDisk_SCI(m304, msci, vsci=[1,1500], vdisk=[1,200], log=True, rmask=1.1, scaleSCI=2.4, cmapdisk=plt.get_cmap('sdoaia304'), cmapsci=plt.get_cmap('gist_heat'))
        plt.imshow(imc, extent=edgs)
    '''
    try:
        rsun_obs = mdisk.meta['rsun_obs']
    except:
        rsun_obs = msci.meta['rsun_obs']
    if abs(mdisk.meta['pc1_2'])>1.e-3:
        mdisk = mapUpRecenter(mdisk)
    nya, nxa = mdisk.data.shape
    scaleDisk = (mdisk.scale.axis1.to(u.arcsec/u.pix)).value
    nxn = int(nxa/(scaleSCI/scaleDisk))
    nyn = int(nya/(scaleSCI/scaleDisk))
    mdiskb = mdisk.resample(dimensions=(nxn, nyn)*u.pix) # scale to 2.4 arcsec/pix
    nys, nxs = msci.data.shape
    if abs(msci.meta['pc1_2'])>1.e-3:
        msci = mapUpRecenter(msci)
    if nys!=2048 or nxs!=2048:
        print('msci should be complete images without cropping')
        return -1,-1
    # nx = ny = 2048
    # ixa = int((nxs-nx)/2); ixb = ixa+nx
    # iya = int((nys-ny)/2); iyb = iya+ny
    imsci = removenan(msci.data) # [iya:iyb,ixa:ixb])
    imdisk = removenan(mdiskb.data)
    if logsci is True:
        imsci[imsci<0.1] = 0.1
        imsci = np.log10(imsci)
        vsci = np.log10(np.array(vsci))
    if logdisk is True:
        imdisk[imdisk<0.1] = 0.1
        imdisk = np.log10(imdisk)
        vdisk = np.log10(np.array(vdisk))
    imsci = norm0_1(imsci, vsci[0], vsci[1])
    imdisk = norm0_1(imdisk, vdisk[0], vdisk[1])
    rrdisk = dist_circle([nyn, nxn])*scaleSCI
    rrsci = dist_circle(2048)*scaleSCI
    idxdisk = rrdisk<rsun_obs*rmask
    idxsci = rrsci<rsun_obs*rmask
    if cmapdisk is None or cmapsci is None:
        imout = imsci.astype(np.float32)
        imout[idxsci] = imdisk[idxdisk]
    else:
        imout = cmapsci(imsci)
        tmp = cmapdisk(imdisk)
        for i in range(4):
            tmpa = imout[:,:,i].copy()
            tmpb = tmp[:,:,i].copy()
            tmpa[idxsci] = tmpb[idxdisk]
            imout[:,:,i] = tmpa
        imout = imout.astype(np.float32)
    edg = 1024*scaleSCI
    edgs = [-edg, edg, -edg, edg]
    # msci.fits_header is possibly updated if rotation is applied.
    return imout, edgs, msci.fits_header