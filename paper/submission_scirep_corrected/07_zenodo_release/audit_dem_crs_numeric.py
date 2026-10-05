#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared-dir',type=Path,required=True)
    ap.add_argument('--dem-dir',type=Path,required=True)
    ap.add_argument('--out-json',type=Path,required=True)
    ap.add_argument('--crs',default='EPSG:32648')
    a=ap.parse_args()
    gpkg=a.prepared_dir/'dem_slope_analysis'/'output'/'DLTB_with_slope.gpkg'
    g=gpd.read_file(gpkg)
    field=next(c for c in g.columns if c.lower() in ('slope_mean','slope','slope_deg','slope_degree'))
    geographic=g[field].astype(float).to_numpy()
    tif=sorted(a.dem_dir.glob('Copernicus_DSM_COG_10_N*_00_DEM.tif'))
    if len(tif)<2: raise RuntimeError(f'expected 2 DEM tiles, found {tif}')
    vrts=[]
    for p in tif:
        src=rasterio.open(p)
        vrts.append(WarpedVRT(src,crs=a.crs,resampling=Resampling.bilinear,resolution=30.0))
    dem,tr=merge(vrts,res=30.0)
    arr=dem[0].astype('float32')
    nodata=vrts[0].nodata
    if nodata is not None: arr[arr==nodata]=np.nan
    dx=abs(tr.a); dy=abs(tr.e)
    gy,gx=np.gradient(arr,dy,dx)
    projected=np.degrees(np.arctan(np.sqrt(gx*gx+gy*gy))).astype('float32')
    projected[~np.isfinite(arr)]=np.nan
    # rasterize parcel ids in projected grid
    gp=g.to_crs(a.crs)
    shapes=((geom,i+1) for i,geom in enumerate(gp.geometry) if geom is not None and not geom.is_empty)
    ids=rasterize(shapes,out_shape=projected.shape,transform=tr,fill=0,dtype='int32',all_touched=False)
    valid=np.isfinite(projected)&(ids>0)
    sums=np.bincount(ids[valid],weights=projected[valid],minlength=len(g)+1)
    counts=np.bincount(ids[valid],minlength=len(g)+1)
    proj_mean=np.full(len(g),np.nan,dtype='float64'); ok=counts[1:]>0; proj_mean[ok]=sums[1:][ok]/counts[1:][ok]
    # representative-point fallback for projected no-cell parcels
    missing=~np.isfinite(proj_mean)
    if missing.any():
        pts=gp.geometry.iloc[np.flatnonzero(missing)].representative_point()
        cols=((pts.x.to_numpy()-tr.c)/tr.a).astype(int); rows=((pts.y.to_numpy()-tr.f)/tr.e).astype(int)
        good=(rows>=0)&(rows<projected.shape[0])&(cols>=0)&(cols<projected.shape[1])
        vals=np.full(len(rows),np.nan); vals[good]=projected[rows[good],cols[good]]
        proj_mean[missing]=vals
    m=np.isfinite(geographic)&np.isfinite(proj_mean)
    d=proj_mean[m]-geographic[m]
    out={'source_gpkg':str(gpkg),'dem_tiles':[str(x) for x in tif],'projected_crs':a.crs,'n_parcels':int(len(g)),'n_compared':int(m.sum()),'n_geographic_fallback_or_missing':int((~np.isfinite(geographic)).sum()),'n_projected_fallback':int(missing.sum()),'geographic_mean_deg':float(np.mean(geographic[m])),'projected_mean_deg':float(np.mean(proj_mean[m])),'mean_delta_projected_minus_geographic_deg':float(np.mean(d)),'median_delta_deg':float(np.median(d)),'mae_deg':float(np.mean(np.abs(d))),'rmse_deg':float(np.sqrt(np.mean(d*d))),'pearson_r':float(stats.pearsonr(geographic[m],proj_mean[m]).statistic),'spearman_rho':float(stats.spearmanr(geographic[m],proj_mean[m]).statistic),'delta_q05_q95_deg':[float(x) for x in np.quantile(d,[.05,.95])],'projected_raster_shape':[int(x) for x in projected.shape],'projected_pixel_size_m':[float(dx),float(dy)]}
    a.out_json.parent.mkdir(parents=True,exist_ok=True); a.out_json.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
