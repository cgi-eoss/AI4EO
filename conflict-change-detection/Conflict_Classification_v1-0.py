import pyeo
import gdal
import os
import shutil, imp, sys
import osr, ogr
from xml.dom.minidom import parse
import time
import datetime
from datetime import datetime as dt

import numpy
import time
from  scipy import ndimage
import matplotlib.pyplot as plt
from pyeo import raster_manipulation
from zipfile import ZipFile
from pyeo.filesystem_utilities import init_log
init_log("test.log")

import geopy.distance

from sklearn.ensemble import RandomForestClassifier
from osgeo import gdal_array
import joblib
import scipy
from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as etree
from skimage.morphology import binary_opening
import glob
import logging
import subprocess
import re
from tempfile import TemporaryDirectory
import numpy as np
import faulthandler

from pyeo.filesystem_utilities import sort_by_timestamp, get_sen_2_tiles, get_l1_safe_file, get_sen_2_image_timestamp, \
    get_sen_2_image_tile, get_sen_2_granule_id, check_for_invalid_l2_data, get_mask_path, get_sen_2_baseline, \
    get_safe_product_type
from pyeo.exceptions import CreateNewStacksException, StackImagesException, BadS2Exception, NonSquarePixelException
log = logging.getLogger("pyeo")
import pyeo.windows_compatability
faulthandler.enable()

#Input from the terminal pre post fire image and sec2cor paths
fpre=sys.argv[1]
fpos=sys.argv[2]

#Output path where the results are generated
output=sys.argv[3]

#Ensures the directory exists, or else create one
def ensure_dir(f):
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
            return d
        return f

for x in os.listdir(fpre+'/files/'):
    if x.endswith('.tif') & x.__contains__('harm'):
        raspre0=fpre+'/files/'+x
        print('Pre-image found')
        pre_image_tag=x.split('_SR')[0]
        break
    else:
        print('Searching Pre-image')

for y in os.listdir(fpos+'/files/'):

    if y.endswith('.tif') & y.__contains__('harm'):
        raspos0=fpos+'/files/'+y
        print('Post-image found')
        post_image_tag=y.split('_SR')[0]
        break
    else:
        print('Searching Post-image')

raster_paths = [raspre0, raspos0]
print('raster_paths',raster_paths)
#out_raster_path=ensure_dir(fpre+'/'+'Output')

output_path = ensure_dir(output+'/'+pre_image_tag+'_'+post_image_tag)
print(output_path)

out_raster_path=output_path+'/'+pre_image_tag+'_'+post_image_tag+'.tif'
ensure_dir(output+'/'+pre_image_tag+'_'+post_image_tag+'/training')
ensure_dir(output+'/'+pre_image_tag+'_'+post_image_tag+'/manual')

pyeo.raster_manipulation.stack_images(raster_paths, out_raster_path, geometry_mode='intersect', format='GTiff', datatype=5)

#Creating empty shape files in the training and manual directories
def creat_emptyshapefile_fromrasterfile(rfin,sfout):
  pp=gdal.Open(rfin)
  srs = osr.SpatialReference()
  srs.ImportFromWkt(pp.GetProjection())
  ensure_dir(os.path.dirname(sfout)+"/")
  shpDriver = ogr.GetDriverByName("ESRI Shapefile")
  if os.path.exists(sfout)==False:
    outDataSource = shpDriver.CreateDataSource(sfout)
    outLayer = outDataSource.CreateLayer(sfout,srs,geom_type=ogr.wkbPolygon)
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    outLayer.CreateField(idField)
    outDataSource.Destroy()

creat_emptyshapefile_fromrasterfile(out_raster_path,output+'/'+pre_image_tag+'_'+post_image_tag+'/training/training.shp')
creat_emptyshapefile_fromrasterfile(out_raster_path,output+'/'+pre_image_tag+'_'+post_image_tag+'/manual/manual.shp')
#Selection of AI/ML alogorithm (RF or SVM)
#
#
#

#Selection of appropriate RF trained model for the tile
pkls=os.path.dirname(output_path)+"/"+"pkls"
pkldir=pkls
original = pkldir + "/" + [x for x in os.listdir(pkldir) if x.endswith(".pkl")][-1]#pkldir + "/" + finall
target = output+'/'+pre_image_tag+'_'+post_image_tag + "/training/rf.pkl"
shutil.copyfile(original, target)

#(or) Selection of appropriate SVM trained model for the tile
#
#
#

yourname="JohnSmith"
userefl=True
rmsmall=False
project="AI4EO"

def save_array_as_image(array, path, geotransform, projection, format = "GTiff"):
   
    driver = gdal.GetDriverByName(format)
    type_code = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    # If array is 2d, give it an extra dimension.
    if len(array.shape) == 2:
        array = numpy.expand_dims(array, axis=0)
    out_dataset = driver.Create(
        path,
        xsize=array.shape[2],
        ysize=array.shape[1],
        bands=array.shape[0],
        eType=type_code
    )
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)
    out_array = out_dataset.GetVirtualMemArray(eAccess=gdal.GA_Update).squeeze()
    out_array[...] = array
    out_array = None
    out_dataset = None
    return path

def find_indices(x,y):
  xsorted = numpy.argsort(x)
  ypos = numpy.searchsorted(x[xsorted], y)
  return xsorted[ypos]

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
	    os.makedirs(d)

def remove_smallobjects(objective,size=10):
    labels, n_labels = scipy.ndimage.label(objective)
    sizes = scipy.ndimage.sum(objective, labels, range(n_labels + 1))
    mask_size = sizes < size
    remove_pixel = mask_size[labels]
    reclass=remove_pixel&objective
    return reclass


def write_metadata(authorstr="Bob",datestr="10/1/2014",datasourcestr="names of images",institutionstr="University of Leicester",outmetadafile=""):
    #ISO 19139:2007 Geographic information -- Metadata -- XML schema implementation
    #https://www.iso.org/obp/ui/#iso:std:iso:ts:19139:ed-1:v1:en
    #you have to buy the document!!
    root=Element('metadata')
    author=Element('author')
    root.append(author)
    author.text=authorstr
    institution=Element('institution')
    root.append(institution)
    institution.text=institutionstr
    modified=Element('modified')
    root.append(modified)
    modified.text=datestr
    datasource=Element('input_datasource')
    root.append(datasource)
    datasource.text=datasourcestr
    tree=ElementTree(root)
    tree.write(open(outmetadafile,"wb"))

def convert_shapefile2array(trainingfnm,img_ds,nodata_value,options):
    trng_ds=ogr.Open(trainingfnm)
    trng_layer=trng_ds.GetLayer()
    trng_rds = gdal.GetDriverByName("MEM").Create("",img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_Int16)
    trng_rds.SetGeoTransform(img_ds.GetGeoTransform())
    trng_rds.SetProjection(img_ds.GetProjection())
    band = trng_rds.GetRasterBand(1)
    band.SetNoDataValue(-1)
    tra = numpy.zeros((img_ds.RasterYSize,img_ds.RasterXSize), dtype = numpy.int)
    tra[:]=nodata_value
    trng_rds.GetRasterBand(1).WriteArray(tra)
    gdal.RasterizeLayer(trng_rds, [1], trng_layer, burn_values=[nodata_value],options=options)
    tra=trng_rds.GetRasterBand(1).ReadAsArray()
    trng_ds,trng_rds=None,None
    return tra

def trainrf(rffname):
    rf=None
    rf = RandomForestClassifier()
    rf = rf.fit(X, Y)
    joblib.dump(rf,traindir+"/rf.pkl")
    return rf

indir=output_path
inputImg=indir+"/"+[x for x in os.listdir(indir) if x.endswith(".tif")][0]
traindir=indir+"/training"
manualdir=indir+"/manual"
img_ds=gdal.Open(inputImg)
img=img_ds.ReadAsArray()

img=img.transpose(1,2,0)
li=("pre1","pre2","pre3","pre4", "pos1","pos2","pos3", "pos4") 
mask2d=(img==(-9999)).any(axis=2)|(img==(0)).any(axis=2)
# spectral indices
spec=numpy.zeros((img_ds.RasterYSize, img_ds.RasterXSize, 9))
numpy.seterr(divide='ignore', invalid='ignore')
spec[:, :, 0]=(img[:, :, li.index("pre4")].astype("float")-img[:, :, li.index("pre3")])/(img[:, :, li.index("pre4")]+img[:, :, li.index("pre3")])
spec[:, :, 1]=(img[:, :, li.index("pos4")].astype("float")-img[:, :, li.index("pos3")])/(img[:, :, li.index("pos4")]+img[:, :, li.index("pos3")])
spec[:, :, 2]=spec[:, :, 0]-spec[:, :, 1]
array_spec=spec[:, :, 0]-spec[:, :, 1]


specmask=(numpy.isnan(spec)|numpy.isinf(spec)).copy()
spec[specmask]=0
ls=("prenbr","posnbr","inbr")
mask2d=mask2d|specmask.any(axis=2)
if mask2d.all():
    bamap=numpy.zeros(mask2d.shape,"int8")
    bamap[:,:]=2
else:
    tra=convert_shapefile2array(traindir+"/training.shp",img_ds,nodata_value=-1,options=['ATTRIBUTE=id'])
    maa=convert_shapefile2array(manualdir+"/manual.shp",img_ds,nodata_value=0,options=['ATTRIBUTE=id'])
    mask2d[(maa==2)|(maa<0)|(mask2d==True)]=True
    if userefl:
      toclass=numpy.concatenate((spec,img),axis=2)
    else:
      toclass=spec

Xtp=toclass.reshape(toclass.shape[0]*toclass.shape[1],toclass.shape[2])
train2d=(tra!=(-1))&(mask2d==False)
X=toclass[train2d]
train2d=(tra!=(-1))&(mask2d==False)
Y=tra[train2d]
if os.path.isfile(traindir+"/rf.pkl"):
    if os.path.getmtime(traindir+"/training.shp")>os.path.getmtime(traindir+"/rf.pkl"):
	    rf=trainrf(traindir+"/rf.pkl")
    else:
	    rf=joblib.load(traindir+"/rf.pkl")
else:
    rf=trainrf(traindir+"/rf.pkl")
time.sleep(0.1)
clp=rf.predict(Xtp)
bamap=clp.reshape(img.shape[:2]).astype("int16")
bamap[maa!=0]=maa[maa!=0]
mask2d[maa==2]=maa[maa==2]
bamap[mask2d]=2 ### specific for fire_cci!!
if rmsmall: ### specific for fire_cci!! # clean the objects smaller than 2x2 pixels
    struct=numpy.zeros((3,3),"int")==1
    struct[:2,:2]=True
    baclean=scipy.ndimage.morphology.binary_opening(bamap==1,structure=struct)
    bamap[(baclean==False)&(mask2d==False)]=3
bamap_ds=gdal.GetDriverByName("MEM").Create("",img.shape[1], img.shape[0],1,gdal.GDT_Int16)
bamap_ds.SetGeoTransform(img_ds.GetGeoTransform())
bamap_ds.SetProjection(img_ds.GetProjection())
bamap_ds.GetRasterBand(1).WriteArray(bamap)
ss_1=inputImg.split("/")[-1].split("_")[0]
ss_2=inputImg.split("/")[-1].split("_")[2]
predate=inputImg.split("/")[-1].split("_")[1]
postdate=inputImg.split("/")[-1].split("_")[3]
tile=inputImg.split("/")[-1].split("_")[4]
ronum=inputImg.split("/")[-1].split("_")[-1].split(".")[0]
flnmshp=indir+"/"+ss_1+"_"+predate+"_"+ss_2+"_"+postdate+"_"+tile+"_"+ronum+".shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
if os.path.exists(flnmshp):
    driver.DeleteDataSource(flnmshp)

proba=rf.predict_proba(Xtp)
#proba[:,0] for burned, and proba[:,1] for unburned
probal = proba[:,0]
probamap=probal.reshape(img.shape[:2]).astype("float")
probamap[maa!=0]=maa[maa!=0]
mask2d[maa==2]=maa[maa==2]
probamap[mask2d]=2 
probras=indir+"/"+ss_1+"_"+predate+"_"+ss_2+"_"+postdate+"_"+tile+"_"+ronum+"_ba_proba.tif"
array=probamap
path=probras
geotransform=img_ds.GetGeoTransform()
projection=img_ds.GetProjection()
save_array_as_image(array, path, geotransform, projection, format = "GTiff")

array=array_spec
path=indir+"/"+ss_1+"_"+predate+"_"+ss_2+"_"+postdate+"_"+tile+"_"+ronum+"_dnbr.tif"
geotransform=img_ds.GetGeoTransform()
projection=img_ds.GetProjection()
save_array_as_image(array, path, geotransform, projection, format = "GTiff")

shp_ds = driver.CreateDataSource(flnmshp)
srs = osr.SpatialReference()
srs.ImportFromWkt(img_ds.GetProjection())
shp_layer = shp_ds.CreateLayer(flnmshp.split("/")[-1].split(".")[0], srs = srs )
newField = ogr.FieldDefn('id', ogr.OFTInteger)
shp_layer.CreateField(newField)
gdal.Polygonize( bamap_ds.GetRasterBand(1), None, shp_layer, 0, [], callback=None )
shp_ds.Destroy()

array=bamap
path=indir+"/"+ss_1+"_"+predate+"_"+ss_2+"_"+postdate+"_"+tile+"_"+ronum+"_ba.tif"
geotransform=img_ds.GetGeoTransform()
projection=img_ds.GetProjection()
save_array_as_image(array, path, geotransform, projection, format = "GTiff")


# Create the attribute table
orgsource = ogr.Open(flnmshp, update=True)
layer = orgsource.GetLayer()
layer_defn = layer.GetLayerDefn()
Category = ogr.FieldDefn('Category', ogr.OFTInteger)
layer.AlterFieldDefn(0, Category, ogr.ALTER_NAME_FLAG )
layer.CreateField(ogr.FieldDefn('PreDate', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('PostDate', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('PreImg', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('PostImg', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('Area', ogr.OFTReal))
layer.CreateField(ogr.FieldDefn('Maybes', ogr.OFTString))


for i in range(layer.GetFeatureCount()):
    feature = layer.GetFeature(i)
    Category_value=feature.GetField(feature.GetFieldIndex("Category"))
    if (Category_value==1)|(Category_value<0):
      feature.SetField(feature.GetFieldIndex("PreDate"), predate)
      feature.SetField(feature.GetFieldIndex("PostDate"), postdate)
      feature.SetField(feature.GetFieldIndex("PreImg"), ss_1+"_"+tile)
      feature.SetField(feature.GetFieldIndex("PostImg"), ss_2+"_"+tile)
      feature.SetField(feature.GetFieldIndex("Area"), feature.GetGeometryRef().Area())

    if Category_value<0:
      feature.SetField(feature.GetFieldIndex("Category"), 2)

    layer.SetFeature(feature)

write_metadata(authorstr=yourname,datestr=time.strftime("%d/%m/%Y" ,time.gmtime()),datasourcestr='; '.join(indir.split("/")[-1].split("_")),outmetadafile=flnmshp+".xml")
orgsource=None