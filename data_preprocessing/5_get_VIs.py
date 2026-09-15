import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageSequence
import imageio.v3 as iio
from tqdm import tqdm

dir_path = '/share/data_share/UAV_MSI_data/CV_Experiments/MSI_NYH2_NYH3_2020-24'
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

labels_vi = pd.DataFrame(np.nan, index=range(len(onlyfiles)), columns=('file_name', 'blue', 'green', 'red', 're', 'nir', 
                                                                       'bcc', 'bgi', 'bi', 'cive', 'exg', 
                                                                       'exg2', 'exgr', 'exr', 'gmb', 'gmr', 
                                                                       'gdb', 'gdr', 'gcc', 'gli', 'mexg', 
                                                                       'mgvri', 'msrgr', 'mrcc', 'ndi', 'ndrbi', 
                                                                       'ngbdi', 'ngrdi', 'nrmbi', 'rmb', 'rdb', 
                                                                       'rcc', 'rgbvi', 'tgi', 'vari', 'tndgr', 
                                                                       'veg', 'com1', 'com2', 
                                                                       'psri', 'ndvi', 'gndvi', 'rvi', 'ndre', 
                                                                       'tvi', 'cvi', 'evi', 'cig', 'cire', 'dvi'))

for i in tqdm(range(len(onlyfiles))):
    file_name = onlyfiles[i]
    img_pth = f"{dir_path}/{file_name}"
    # for pngs
    # img_pil = Image.open(img_pth) # load PIL image object. if you just use this it'll be a single channel
    # img_npy = np.empty((*img_pil.size[::-1], 0), dtype=np.int64) # this is where we'll store all 5 channels
    # for im in ImageSequence.Iterator(img_pil): # iterate through 5 channels
        # img_npy = np.concatenate([img_npy, np.array(im)[..., None]], axis=-1) # add it to our numpy array
    # for tifs
    img_npy = iio.imread(img_pth) # transparent values set to NA
    if np.nanmax(img_npy) > 1: print("Clipping upper:", np.sum(img_npy > 1))
    if np.nanmin(img_npy) < 0: print("Clipping lower:", np.sum(img_npy < 0))
    if np.isnan(img_npy).sum() != 0: print("NAs:", np.isnan(img_npy).sum())
    img_npy = np.clip(img_npy, 0, 1) # set values greater than 1 equal to 1
    # get distribution of pixel values & threshold
    # lower_thresh = np.quantile(img_npy, 0.25)
    # upper_thresh = np.quantile(img_npy, 0.75)
    # img_npy[np.logical_and.reduce([img_npy >= lower_thresh, img_npy <= upper_thresh])] = 0
    # get bands
    B = img_npy[:,:,0].astype(float)
    G = img_npy[:,:,1].astype(float)
    R = img_npy[:,:,2].astype(float)
    RE = img_npy[:,:,3].astype(float)
    NIR = img_npy[:,:,4].astype(float)
    # get soil pixels
    sci = np.divide((R-G), (R+G), out=np.zeros_like(R-G), where=(R+G)!=0)
    hi = np.divide((2*R-G-B), (G-B), out=np.zeros_like(2*R-G-B), where=(G-B)!=0)
    si = np.divide((R-B), (R+B), out=np.zeros_like(R-B), where=(R+B)!=0)
    hue = np.arctan(np.divide(2*(B-G-R), 30.5*(G-R), out=np.zeros_like(2*(B-G-R)), where=(30.5*(G-R))!=0))
    # values to keep
    mask_dist = np.logical_and.reduce([NIR >= np.quantile(NIR, 0.25), NIR <= np.quantile(NIR, 0.75)])
    mask_hue = hue < 0
    # flag to ignore the mask
    ignore_mask = True
    # use all elements if ignore_mask is True, otherwise apply the mask
    mask = mask_dist if ignore_mask == False else np.ones_like(B, dtype=bool)
    # get VIs
    blue = np.nanmean(B, where = mask)
    green = np.nanmean(G, where = mask)
    red = np.nanmean(R, where = mask)
    re = np.nanmean(RE, where = mask)
    nir = np.nanmean(NIR, where = mask)
    bcc = np.nanmean(np.divide(B, R + G + B, out=np.zeros_like(B), where=(R + G + B)!=0), where = mask)
    bgi = np.nanmean(np.divide(B, G, out=np.zeros_like(B), where=G!=0), where = mask)
    bi = np.nanmean(np.sqrt((np.square(R) + np.square(G) + np.square(B))/3), where = mask)
    cive = np.nanmean(0.441*R - 0.811*G + 0.385*B + 18.78745, where = mask)
    exg = np.nanmean(2*G - R - B, where = mask)
    exg2 = np.nanmean(np.divide(2*G - R - B, G + R + B, out=np.zeros_like(2*G - R - B), where=(G + R + B)!=0), where = mask)
    exgr = np.nanmean(3*G - 2.4*R - B, where = mask)
    exr = np.nanmean(1.4*R - G, where = mask)
    gmb = np.nanmean(G - B, where = mask)
    gmr = np.nanmean(G - R, where = mask)
    gdb = np.nanmean(np.divide(G, B, out=np.zeros_like(G), where=B!=0), where = mask)
    gdr = np.nanmean(np.divide(G, R, out=np.zeros_like(G), where=R!=0), where = mask)
    gcc = np.nanmean(np.divide(G, R + G + B, out=np.zeros_like(G), where=(R + G + B)!=0), where = mask)
    gli = np.nanmean(np.divide(2*G - R - B, 2*G + R + B, out=np.zeros_like(2*G - R - B), where=(2*G + R + B)!=0), where = mask)
    mexg = np.nanmean(1.262*G - 0.884*R - 0.311*B, where = mask)
    mgvri = np.nanmean(np.divide(np.square(G) - np.square(R), np.square(G) + np.square(R), out=np.zeros_like(np.square(G) - np.square(R)), where=(np.square(G) + np.square(R))!=0), where = mask)
    msrgr = np.nanmean(np.sqrt(np.divide(G, R, out=np.zeros_like(G), where=R!=0)), where = mask)
    mrcc = np.nanmean(np.divide(np.power(R, 3), R + G + B, out=np.zeros_like(np.power(R, 3)), where=(R + G + B)!=0), where = mask)
    ndi = np.nanmean(128*(np.divide(G - R, G + R, out=np.zeros_like(G - R), where=(G + R)!=0) + 1), where = mask)
    ndrbi = np.nanmean(np.divide(R - B, R + B, out=np.zeros_like(R - B), where=(R + B)!=0), where = mask)
    ngbdi = np.nanmean(np.divide(G - B, G + B, out=np.zeros_like(G - B), where=(G + B)!=0), where = mask)
    ngrdi = np.nanmean(np.divide(G - R, G + R, out=np.zeros_like(G - R), where=(G + R)!=0), where = mask)
    nrmbi = np.nanmean(np.divide(R - B, G, out=np.zeros_like(R - B), where=G!=0), where = mask)
    rmb = np.nanmean(R - B, where = mask)
    rdb = np.nanmean(np.divide(R, B, out=np.zeros_like(R), where=B!=0), where = mask)
    rcc = np.nanmean(np.divide(R, R + G + B, out=np.zeros_like(R), where=(R + G + B)!=0), where = mask)
    rgbvi = np.nanmean(np.divide(np.square(G) - R*B, np.square(G) + R*B, out=np.zeros_like(np.square(G) - R*B), where=(np.square(G) + R*B)!=0), where = mask)
    tgi = np.nanmean(G - 0.39*R - 0.69*B, where = mask)
    vari = np.nanmean(np.divide(G - R, G + R - B, out=np.zeros_like(G - R), where=(G + R - B)!=0), where = mask)
    tndgr = np.nanmean(np.sqrt(np.divide(G - R, G + R, out=np.zeros_like(G - R), where=(G + R)!=0) + 0.5), where = mask)
    veg = np.nanmean(np.divide(G, np.power(R, 0.667)*np.power(B, 0.334), out=np.zeros_like(G), where=(np.power(R, 0.667)*np.power(B, 0.334))!=0), where = mask)
    com1 = exg + cive + exgr + veg
    com2 = 0.36*exg + 0.47*cive + 0.17*veg
    psri = np.nanmean(np.divide(R - G, RE, out=np.zeros_like(R - G), where=RE!=0), where = mask)
    ndvi = np.nanmean(np.divide(NIR - R, NIR + R, out=np.zeros_like(NIR - R), where=(NIR + R)!=0), where = mask)
    gndvi = np.nanmean(np.divide(NIR - G, NIR + G, out=np.zeros_like(NIR - G), where=(NIR + G)!=0), where = mask)
    rvi = np.nanmean(np.divide(NIR, R, out=np.zeros_like(NIR), where=R!=0), where = mask)
    ndre = np.nanmean(np.divide(NIR - RE, NIR + RE, out=np.zeros_like(NIR - RE), where=(NIR + RE)!=0), where = mask)
    tvi = np.nanmean(0.5*(120*(NIR - G) - 200*(R - G)), where = mask)
    cvi = np.nanmean(np.divide(NIR*R, np.square(G), out=np.zeros_like(NIR*R), where=(np.square(G))!=0), where = mask)
    evi = np.nanmean(np.divide(2.5*(NIR - R), NIR + 6*R - 7.5*B + 1, out=np.zeros_like(2.5*(NIR - R)), where=(NIR + 6*R - 7.5*B + 1)!=0), where = mask)
    cig = np.nanmean(np.divide(NIR, G, out=np.zeros_like(NIR), where=G!=0) - 1, where = mask)
    cire = np.nanmean(np.divide(NIR, RE, out=np.zeros_like(NIR), where=RE!=0) - 1, where = mask)
    dvi = np.nanmean(NIR - RE, where = mask)
    labels_vi.loc[i, 'file_name'] = file_name
    labels_vi.loc[i, 'blue'] = blue
    labels_vi.loc[i, 'green'] = green
    labels_vi.loc[i, 'red'] = red
    labels_vi.loc[i, 're'] = re
    labels_vi.loc[i, 'nir'] = nir
    labels_vi.loc[i, 'bcc'] = bcc
    labels_vi.loc[i, 'bgi'] = bgi
    labels_vi.loc[i, 'bi'] = bi
    labels_vi.loc[i, 'cive'] = cive
    labels_vi.loc[i, 'exg'] = exg
    labels_vi.loc[i, 'exg2'] = exg2
    labels_vi.loc[i, 'exgr'] = exgr
    labels_vi.loc[i, 'exr'] = exr
    labels_vi.loc[i, 'gmb'] = gmb
    labels_vi.loc[i, 'gmr'] = gmr
    labels_vi.loc[i, 'gdb'] = gdb
    labels_vi.loc[i, 'gdr'] = gdr
    labels_vi.loc[i, 'gcc'] = gcc
    labels_vi.loc[i, 'gli'] = gli
    labels_vi.loc[i, 'mexg'] = mexg
    labels_vi.loc[i, 'mgvri'] = mgvri
    labels_vi.loc[i, 'msrgr'] = msrgr
    labels_vi.loc[i, 'mrcc'] = mrcc
    labels_vi.loc[i, 'ndi'] = ndi
    labels_vi.loc[i, 'ndrbi'] = ndrbi
    labels_vi.loc[i, 'ngbdi'] = ngbdi
    labels_vi.loc[i, 'ngrdi'] = ngrdi
    labels_vi.loc[i, 'nrmbi'] = nrmbi
    labels_vi.loc[i, 'rmb'] = rmb
    labels_vi.loc[i, 'rdb'] = rdb
    labels_vi.loc[i, 'rcc'] = rcc
    labels_vi.loc[i, 'rgbvi'] = rgbvi
    labels_vi.loc[i, 'tgi'] = tgi
    labels_vi.loc[i, 'vari'] = vari
    labels_vi.loc[i, 'tndgr'] = tndgr
    labels_vi.loc[i, 'veg'] = veg
    labels_vi.loc[i, 'com1'] = com1
    labels_vi.loc[i, 'com2'] = com2
    labels_vi.loc[i, 'psri'] = psri
    labels_vi.loc[i, 'ndvi'] = ndvi
    labels_vi.loc[i, 'gndvi'] = gndvi
    labels_vi.loc[i, 'rvi'] = rvi
    labels_vi.loc[i, 'ndre'] = ndre
    labels_vi.loc[i, 'tvi'] = tvi
    labels_vi.loc[i, 'cvi'] = cvi
    labels_vi.loc[i, 'evi'] = evi
    labels_vi.loc[i, 'cig'] = cig
    labels_vi.loc[i, 'cire'] = cire
    labels_vi.loc[i, 'dvi'] = dvi

labels_vi.to_csv('labels_VI_NYH2_NYH3_2020-24.csv')

