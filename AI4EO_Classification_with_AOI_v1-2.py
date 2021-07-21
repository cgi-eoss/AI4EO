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
from scipy import ndimage
import matplotlib.pyplot as plt
from pyeo import raster_manipulation
from zipfile import ZipFile
from pyeo.filesystem_utilities import init_log

init_log("test.log")

from sklearn.ensemble import RandomForestClassifier
from osgeo import gdal_array
import joblib
import scipy
from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as etree
from skimage.morphology import binary_opening
import geopy.distance
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


def apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=False):
    gipp_path = os.path.join(os.path.dirname(__file__), "L2A_GIPP.xml")
    out_dir = os.path.dirname(os.path.dirname(image_path)) + "/" + "L2"

    log.info("calling subprocess: {}".format([sen2cor_path, image_path, '--output_dir', out_dir]))
    now_time = datetime.datetime.now()  # I can't think of a better way of geting the new outpath from sen2cor
    timestamp = now_time.strftime(r"%Y%m%dT%H%M%S")
    sen2cor_proc = subprocess.Popen([sen2cor_path, image_path, '--output_dir', out_dir,
                                     '--GIP_L2A', gipp_path],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
    while True:
        nextline = sen2cor_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and sen2cor_proc.poll() is not None:
            break
        if "CRITICAL" in nextline:
            # log.error(nextline)
            raise subprocess.CalledProcessError(-1, "L2A_Process")

    log.info("sen2cor processing finished for {}".format(image_path))
    log.info("Validating:")
    version = get_sen2cor_version(sen2cor_path)
    out_path = build_sen2cor_output_path(image_path, timestamp, version)
    if not check_for_invalid_l2_data(out_path):
        log.error("10m imagery not present in {}".format(out_path))
        raise BadS2Exception
    if delete_unprocessed_image:
        log.info("removing {}".format(image_path))
        shutil.rmtree(image_path)
    return out_path


def build_sen2cor_output_path(image_path, timestamp, version):
    # Accounting for sen2cors ever-shifting filename format
    if version >= "2.08.00":
        out_path = image_path.replace("MSIL1C", "MSIL2A")
        baseline = get_sen_2_baseline(image_path)
        out_path = out_path.replace(baseline, "N9999")
        out_path = out_path.rpartition("_")[0] + "_" + timestamp + ".SAFE"
    else:
        out_path = image_path.replace("MSIL1C", "MSIL2A")
    return out_path


def get_sen2cor_version(sen2cor_path):
    proc = subprocess.run([sen2cor_path, "--help"], stdout=subprocess.PIPE)
    help_string = proc.stdout.decode("utf-8")

    # Looks for the string "Version: " followed by three sets of digits separated by period characters.
    # Returns the three character string as group 1.
    version_regex = r"Version: (\d+.\d+.\d+)"
    match = re.search(version_regex, help_string)
    if match:
        return match.group(1)
    else:
        version_regex = r"Sen2Cor (\d+.\d+.\d+)"
        match = re.search(version_regex, help_string)
        if match:
            return match.group(1)
        else:
            raise FileNotFoundError("Version information not found; please check your sen2cor path.")

def atmospheric_correction(in_directory, out_directory, sen2cor_path, delete_unprocessed_image=False):
    log = logging.getLogger(__name__)
    image = tempvar
    # Opportunity for multithreading here

    log.info("Atmospheric correction of {}".format(image))
    image_path = os.path.join(in_directory, image)
    image_timestamp = datetime.datetime.now().strftime(r"%Y%m%dT%H%M%S")
    out_name = build_sen2cor_output_path(image, image_timestamp, get_sen2cor_version(sen2cor_path))
    out_path = os.path.join(out_directory, out_name)
    out_glob = out_path.rpartition("_")[0] + "*"
    if glob.glob(out_glob):
        log.warning("{} exists. Skipping.".format(out_path))
    try:
        l2_path = apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=delete_unprocessed_image)
    except (subprocess.CalledProcessError, BadS2Exception):
        log.error("Atmospheric correction failed for {}. Moving on to next image.".format(image))
        pass
    else:
        l2_name = os.path.basename(l2_path)
        log.info("L2  path: {}".format(l2_path))
        log.info("New path: {}".format(os.path.join(out_directory, l2_name)))
        os.rename(l2_path, os.path.join(out_directory, l2_name))


# Input from the terminal pre post fire image and sec2cor paths
fpre = sys.argv[1]
fpos = sys.argv[2]
sen2cor_path = sys.argv[3]
emin = sys.argv[4]
nmin = sys.argv[5]
emax = sys.argv[6]
nmax = sys.argv[7]


# Ensures the directory exists, or else create one
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


tmpdir = os.path.dirname(fpre)
ensure_dir(tmpdir + "/" + "L2")
li = ["raspre", "raspos"]

# Applying sen2cor to create the L2 data
for x in li:
    if x == "raspre":
        tempvar = fpre
        in_directory = fpre
        out_directory = os.path.dirname(tmpdir) + "/" + "L2"
        ensure_dir(out_directory + "/")
        atmcorr_raspre = atmospheric_correction(in_directory, out_directory, sen2cor_path,
                                                delete_unprocessed_image=False)
    if x == "raspos":
        tempvar = fpos
        in_directory = fpos
        out_directory = os.path.dirname(tmpdir) + "/" + "L2"
        ensure_dir(out_directory + "/")
        atmcorr_raspos = atmospheric_correction(in_directory, out_directory, sen2cor_path,
                                                delete_unprocessed_image=False)

# Creating the multiband tiff images of bands 4,8, and 12
l2_dir = out_directory
l1_dir = os.path.dirname(fpre)
out_dir = os.path.dirname(l2_dir) + "/B4_B8_B12"
ensure_dir(out_dir + "/")
multi_cmask = pyeo.raster_manipulation.preprocess_sen2_images(l2_dir, out_dir, l1_dir, cloud_threshold=60,
                                                              buffer_size=0, epsg=None, bands=('B04', 'B08', 'B12'),
                                                              out_resolution=10)

# Generating the appropriate file and folder names
fprestring = fpre.split("/")[-1].split("N")[-1].split("_")[-1].split("T")[0]
fposstring = fpos.split("/")[-1].split("N")[-1].split("_")[-1].split("T")[0]
flag1img = [x for x in os.listdir(out_directory) if fprestring in x][-1].split(".")[0]
flag2img = [x for x in os.listdir(out_directory) if fposstring in x][-1].split(".")[0]
raspre0 = out_dir + "/" + [x for x in os.listdir(out_dir) if x.startswith(flag1img) & x.endswith(".tif")][0]
raspos0 = out_dir + "/" + [x for x in os.listdir(out_dir) if x.startswith(flag2img) & x.endswith(".tif")][0]
s2a = raspre0.split("/")[-1].split("N9999")[0].split("MSIL2A")[0].split("_")[0]
firstdate = raspre0.split("/")[-1].split("N9999")[0].split("MSIL2A")[-1].split("_")[1].split("T")[0]
s2b = raspos0.split("/")[-1].split("N9999")[0].split("MSIL2A")[0].split("_")[0]
secdate = raspos0.split("/")[-1].split("N9999")[0].split("MSIL2A")[-1].split("_")[1].split("T")[0]
t55 = raspre0.split("/")[-1].split("N9999")[-1].split("_")[2]
r030 = raspre0.split("/")[-1].split("N9999")[-1].split("_")[1]

# Stacking the pre and post image
outpath = os.path.dirname(l2_dir) + "/" + "STACKED"
raster_paths = [raspre0, raspos0]

out_dir = outpath + "/" + s2a + "_" + firstdate + "_" + s2b + "_" + secdate + "_" + t55 + "_" + r030
ensure_dir(out_dir + "/")
out_raster_path = out_dir + "/" + s2a + "_" + firstdate + "_" + s2b + "_" + secdate + "_" + t55 + "_" + r030 + ".tif"

pyeo.raster_manipulation.stack_images(raster_paths, out_raster_path, geometry_mode='intersect', format='GTiff',
                                      datatype=5)

# Apply aoi to stacked GTiff if provided
if emin != 'None':
    bounds = [emin, nmin, emax, nmax]
    ds = gdal.Open(out_raster_path)
    out_raster_subset = out_dir + "/subset.tif"
    gdal.Warp(out_raster_subset, ds, outputBounds=bounds)
    print("subset created, replacing...")
    os.remove(out_raster_path)
    os.rename(out_raster_subset, out_raster_path)


def creat_emptyshapefile_fromrasterfile(rfin, sfout):
    pp = gdal.Open(rfin)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(pp.GetProjection())
    ensure_dir(os.path.dirname(sfout) + "/")
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(sfout) == False:
        outDataSource = shpDriver.CreateDataSource(sfout)
        outLayer = outDataSource.CreateLayer(sfout, srs, geom_type=ogr.wkbPolygon)
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        outLayer.CreateField(idField)
        outDataSource.Destroy()


creat_emptyshapefile_fromrasterfile(out_raster_path, out_dir + "/training/training.shp")
creat_emptyshapefile_fromrasterfile(out_raster_path, out_dir + "/manual/manual.shp")


# Selection of AI/ML alogorithm (RF or SVM)
#
#
#

# Calculating  distance between two S2-Tile locations
def dms2dec(dms_str):
    dms_str = re.sub(r'\s', '', dms_str)
    sign = -1 if re.search('[swSW]', dms_str) else 1
    numbers = [*filter(len, re.split('\D+', dms_str, maxsplit=4))]
    degree = numbers[0]
    minute = numbers[1] if len(numbers) >= 2 else '0'
    second = numbers[2] if len(numbers) >= 3 else '0'
    frac_seconds = numbers[3] if len(numbers) >= 4 else '0'
    second += "." + frac_seconds
    return sign * (int(degree) + float(minute) / 60 + float(second) / 3600)


path = out_dir + "/" + [x for x in os.listdir(out_dir) if x.endswith(".tif")][0]
tifname = "T" + path.split("/")[-1].split(".")[0].split("T")[-1].split("_")[0]
pkls = os.path.dirname(outpath) + "/" + "pkls"
result = []
result2 = []
for x in os.listdir(pkls):
    # print("Filename=",os.listdir(pkls))
    numberofpkls = len(os.listdir(pkls))
    if x.endswith(".txt"):
        print(os.path.join(pkls, x))
        pkltilename = x.split(".")[0].split("_")[0]
        with open(os.path.join(pkls, x), 'r') as file:
            pklfile = file.read()
            # print(pklfile)
            p2 = pklfile
        longitude2 = pklfile.split(",")[0]
        # print(longitude2)
        latitude2 = pklfile.split(",")[-1]
        # print(latitude2)
        print("PKL TILE", pkltilename, "LONG/LAT=", longitude2, latitude2)
        p1 = subprocess.Popen(['gdalinfo', path], stdout=subprocess.PIPE)
        output = p1.communicate()[0]
        out1 = output.split(b"Center")[-1].split(b"(")[-1].split(b")")[0]
        lon = out1.split(b",")[0]
        lat = out1.split(b",")[-1]
        lond = lon.split(b"d")[0]
        lond = lond.decode()
        lonm = lon.split(b"d")[-1].split(b"'")[0]
        lonm = lonm.decode()
        lons = lon.split(b"d")[-1].split(b"'")[-1].split(b'"')[0]
        lons = lons.decode()
        londir = lon.split(b"d")[-1].split(b"'")[-1].split(b'"')[-1]
        londir = londir.decode()

        latd = lat.split(b"d")[0]
        latd = latd.decode()
        latm = lat.split(b"d")[-1].split(b"'")[0]
        latm = latm.decode()
        lats = lat.split(b"d")[-1].split(b"'")[-1].split(b'"')[0]
        lats = lats.decode()
        latdir = lat.split(b"d")[-1].split(b"'")[-1].split(b'"')[-1]
        latdir = latdir.decode()
        longitude = lond + "," + lonm + "," + lons + "," + londir
        latitude = latd + "," + latm + "," + lats + "," + latdir
        out1 = out1.decode()
        print("CENTER COORDINATES ", tifname, "=", out1)
        print("CENTER COORDINATES ", tifname, "=", longitude, latitude)

        lg1 = dms2dec(longitude)
        lt1 = dms2dec(latitude)
        coords_1 = (lg1, lt1)
        lg2 = longitude2
        lt2 = latitude2
        print("LONGITUDE, LATITUDE of ", tifname, " =", lg1, lt1)
        print("LONGITUDE, LATITUDE of ", pkltilename, "=", lg2, lt2)
        coords_2 = (lg2, lt2)
        from geopy.distance import lonlat, distance

        # print(lonlat(*coords_1))
        dist = distance(lonlat(*coords_1), lonlat(*coords_2)).km
        print("Distance between two tiles=", dist, "kilometers")
        result.append((pkltilename, dist))
        print(result)
        result2.append(dist)
        minimum = min(result2)
        index = result2.index(minimum)
        print("MIN=", minimum, "INDEX=", index)
        print(result2)
        tilenamef = result[index][0]
        print("PROXIMITY TILE=", tilenamef)

        continue
    else:
        continue

pkldir = pkls

# Calculate days in between
firstdate = path.split("/")[-1].split(".")[0].split("_")[1]
seconddate = path.split("/")[-1].split(".")[0].split("_")[3]
date_format = "%Y%m%d"
a = dt.strptime(firstdate, date_format)
b = dt.strptime(seconddate, date_format)
delta = b - a
noofdays = int(delta.days)

print("NUMBER OF DAYS IN THE INPUT TILE=", noofdays)

for z in os.listdir(pkldir):
    pkltiles = [x for x in os.listdir(pkldir) if x.__contains__(tilenamef) & x.endswith(".pkl")]
daysss = []
tiledays = []
for x in pkltiles:
    spectile = str(x)
    tileday = spectile.split("S")[0]
    tiledays.append(tileday)
    days = spectile.split("_")[1]
    dayss = int(days)
    daysss.append(dayss)

    continue

daysss.sort()
print("AVAILABLE PKLS FOR THE PROXIMITY TILE=", pkltiles)


def getClosestValue(arr, target):
    n = len(arr)
    left = 0
    right = n - 1
    mid = 0

    # edge case
    if (target >= arr[n - 1]):
        return arr[n - 1]
    # BSearch solution: Time & Space: Log(N)

    while (left < right):
        mid = (left + right) // 2  # find the mid

        if (arr[mid] == target):
            return arr[mid]

        if (target < arr[mid]):
            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1]):
                return findClosest(arr[mid - 1], arr[mid], target)
            # update right
            right = mid
        else:
            if (mid < n - 1 and target < arr[mid + 1]):
                return findClosest(arr[mid], arr[mid + 1], target)
            # update i
            left = mid + 1
    return arr[mid]


def findClosest(val1, val2, target):
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1


# Selection of appropriate RF trained model for the tile

arr = daysss
target = noofdays
selecteddaypkl = getClosestValue(arr, target)
sdp = "_" + str(selecteddaypkl)
selectpklfile = [x for x in tiledays if x.__contains__(sdp)][-1]

finall = [x for x in os.listdir(pkls) if x.__contains__(selectpklfile) & x.endswith(".pkl")][-1]
print("SELECTED PKL FILE=", finall)

pkldir = pkls
original = pkldir + "/" + finall
target = out_dir + "/training/rf.pkl"
shutil.copyfile(original, target)

# (or) Selection of appropriate SVM trained model for the tile
#
#
#

yourname = "JohnSmith"
userefl = True
rmsmall = False
project = "AI4EO"


def save_array_as_image(array, path, geotransform, projection, format="GTiff"):
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


def find_indices(x, y):
    xsorted = numpy.argsort(x)
    ypos = numpy.searchsorted(x[xsorted], y)
    return xsorted[ypos]


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def remove_smallobjects(objective, size=10):
    labels, n_labels = scipy.ndimage.label(objective)
    sizes = scipy.ndimage.sum(objective, labels, range(n_labels + 1))
    mask_size = sizes < size
    remove_pixel = mask_size[labels]
    reclass = remove_pixel & objective
    return reclass


def write_metadata(authorstr="Bob", datestr="10/1/2014", datasourcestr="names of images",
                   institutionstr="University of Leicester", outmetadafile=""):
    # ISO 19139:2007 Geographic information -- Metadata -- XML schema implementation
    # https://www.iso.org/obp/ui/#iso:std:iso:ts:19139:ed-1:v1:en
    # you have to buy the document!!
    root = Element('metadata')
    author = Element('author')
    root.append(author)
    author.text = authorstr
    institution = Element('institution')
    root.append(institution)
    institution.text = institutionstr
    modified = Element('modified')
    root.append(modified)
    modified.text = datestr
    datasource = Element('input_datasource')
    root.append(datasource)
    datasource.text = datasourcestr
    tree = ElementTree(root)
    tree.write(open(outmetadafile, "wb"))


def convert_shapefile2array(trainingfnm, img_ds, nodata_value, options):
    trng_ds = ogr.Open(trainingfnm)
    trng_layer = trng_ds.GetLayer()
    trng_rds = gdal.GetDriverByName("MEM").Create("", img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_Int16)
    trng_rds.SetGeoTransform(img_ds.GetGeoTransform())
    trng_rds.SetProjection(img_ds.GetProjection())
    band = trng_rds.GetRasterBand(1)
    band.SetNoDataValue(-1)
    tra = numpy.zeros((img_ds.RasterYSize, img_ds.RasterXSize), dtype=numpy.int)
    tra[:] = nodata_value
    trng_rds.GetRasterBand(1).WriteArray(tra)
    gdal.RasterizeLayer(trng_rds, [1], trng_layer, burn_values=[nodata_value], options=options)
    tra = trng_rds.GetRasterBand(1).ReadAsArray()
    trng_ds, trng_rds = None, None
    return tra


def trainrf(rffname):
    rf = None
    rf = RandomForestClassifier()
    rf = rf.fit(X, Y)
    joblib.dump(rf, traindir + "/rf.pkl")
    return rf


indir = out_dir
inputImg = indir + "/" + [x for x in os.listdir(indir) if x.endswith(".tif")][0]
traindir = indir + "/training"
manualdir = indir + "/manual"
img_ds = gdal.Open(inputImg)
img = img_ds.ReadAsArray()

img = img.transpose(1, 2, 0)
li = ("pre4", "pre8", "pre12", "pos4", "pos8", "pos12")
mask2d=(img==(-9999)).any(axis=2)|(img==(0)).any(axis=2)
# spectral indices
spec = numpy.zeros((img_ds.RasterYSize, img_ds.RasterXSize, 3))
numpy.seterr(divide='ignore', invalid='ignore')
spec[:, :, 0] = (img[:, :, li.index("pre8")].astype("float") - img[:, :, li.index("pre12")]) / (
            img[:, :, li.index("pre8")] + img[:, :, li.index("pre12")])
spec[:, :, 1] = (img[:, :, li.index("pos8")].astype("float") - img[:, :, li.index("pos12")]) / (
            img[:, :, li.index("pos8")] + img[:, :, li.index("pos12")])
spec[:, :, 2] = spec[:, :, 1] - spec[:, :, 0]
array_spec = spec[:, :, 0] - spec[:, :, 1]

specmask = (numpy.isnan(spec) | numpy.isinf(spec)).copy()
spec[specmask] = 0
ls = ("prenbr", "posnbr", "inbr")
mask2d = mask2d | specmask.any(axis=2)
if mask2d.all():
    bamap = numpy.zeros(mask2d.shape, "int8")
    bamap[:, :] = 2
else:
    tra = convert_shapefile2array(traindir + "/training.shp", img_ds, nodata_value=-1, options=['ATTRIBUTE=id'])
    maa = convert_shapefile2array(manualdir + "/manual.shp", img_ds, nodata_value=0, options=['ATTRIBUTE=id'])
    mask2d[(maa == 2) | (maa < 0) | (mask2d == True)] = True
    if userefl:
        toclass = numpy.concatenate((spec, img), axis=2)
    else:
        toclass = spec

Xtp = toclass.reshape(toclass.shape[0] * toclass.shape[1], toclass.shape[2])
train2d = (tra != (-1)) & (mask2d == False)
X = toclass[train2d]
train2d = (tra != (-1)) & (mask2d == False)
Y = tra[train2d]
if os.path.isfile(traindir + "/rf.pkl"):
    if os.path.getmtime(traindir + "/training.shp") > os.path.getmtime(traindir + "/rf.pkl"):
        rf = trainrf(traindir + "/rf.pkl")
    else:
        rf = joblib.load(traindir + "/rf.pkl")
else:
    rf = trainrf(traindir + "/rf.pkl")
time.sleep(0.1)
clp = rf.predict(Xtp)
bamap = clp.reshape(img.shape[:2]).astype("int16")
bamap[maa != 0] = maa[maa != 0]
mask2d[maa == 2] = maa[maa == 2]
bamap[mask2d] = 2  ### specific for fire_cci!!
if rmsmall:  ### specific for fire_cci!! # clean the objects smaller than 2x2 pixels
    struct = numpy.zeros((3, 3), "int") == 1
    struct[:2, :2] = True
    baclean = scipy.ndimage.morphology.binary_opening(bamap == 1, structure=struct)
    bamap[(baclean == False) & (mask2d == False)] = 3
bamap_ds = gdal.GetDriverByName("MEM").Create("", img.shape[1], img.shape[0], 1, gdal.GDT_Int16)
bamap_ds.SetGeoTransform(img_ds.GetGeoTransform())
bamap_ds.SetProjection(img_ds.GetProjection())
bamap_ds.GetRasterBand(1).WriteArray(bamap)
ss_1 = inputImg.split("/")[-1].split("_")[0]
ss_2 = inputImg.split("/")[-1].split("_")[2]
predate = inputImg.split("/")[-1].split("_")[1]
postdate = inputImg.split("/")[-1].split("_")[3]
tile = inputImg.split("/")[-1].split("_")[4]
ronum = inputImg.split("/")[-1].split("_")[-1].split(".")[0]
flnmshp = indir + "/" + ss_1 + "_" + predate + "_" + ss_2 + "_" + postdate + "_" + tile + "_" + ronum + ".shp"
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

array = array_spec
path = indir + "/" + ss_1 + "_" + predate + "_" + ss_2 + "_" + postdate + "_" + tile + "_" + ronum + "_dnbr.tif"
geotransform = img_ds.GetGeoTransform()
projection = img_ds.GetProjection()
save_array_as_image(array, path, geotransform, projection, format="GTiff")

shp_ds = driver.CreateDataSource(flnmshp)
srs = osr.SpatialReference()
srs.ImportFromWkt(img_ds.GetProjection())
shp_layer = shp_ds.CreateLayer(flnmshp.split("/")[-1].split(".")[0], srs=srs)
newField = ogr.FieldDefn('id', ogr.OFTInteger)
shp_layer.CreateField(newField)
gdal.Polygonize(bamap_ds.GetRasterBand(1), None, shp_layer, 0, [], callback=None)
shp_ds.Destroy()

array = bamap
path = indir + "/" + ss_1 + "_" + predate + "_" + ss_2 + "_" + postdate + "_" + tile + "_" + ronum + "_ba.tif"
geotransform = img_ds.GetGeoTransform()
projection = img_ds.GetProjection()
save_array_as_image(array, path, geotransform, projection, format="GTiff")

# Create the attribute table
orgsource = ogr.Open(flnmshp, update=True)
layer = orgsource.GetLayer()
layer_defn = layer.GetLayerDefn()
Category = ogr.FieldDefn('Category', ogr.OFTInteger)
layer.AlterFieldDefn(0, Category, ogr.ALTER_NAME_FLAG)
layer.CreateField(ogr.FieldDefn('PreDate', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('PostDate', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('PreImg', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('PostImg', ogr.OFTString))
layer.CreateField(ogr.FieldDefn('Area', ogr.OFTReal))
layer.CreateField(ogr.FieldDefn('Maybes', ogr.OFTString))

for i in range(layer.GetFeatureCount()):
    feature = layer.GetFeature(i)
    Category_value = feature.GetField(feature.GetFieldIndex("Category"))
    if (Category_value == 1) | (Category_value < 0):
        feature.SetField(feature.GetFieldIndex("PreDate"), predate)
        feature.SetField(feature.GetFieldIndex("PostDate"), postdate)
        feature.SetField(feature.GetFieldIndex("PreImg"), ss_1 + "_" + tile)
        feature.SetField(feature.GetFieldIndex("PostImg"), ss_2 + "_" + tile)
        feature.SetField(feature.GetFieldIndex("Area"), feature.GetGeometryRef().Area())

    if Category_value < 0:
        feature.SetField(feature.GetFieldIndex("Category"), 2)

    layer.SetFeature(feature)

write_metadata(authorstr=yourname, datestr=time.strftime("%d/%m/%Y", time.gmtime()),
               datasourcestr='; '.join(indir.split("/")[-1].split("_")), outmetadafile=flnmshp + ".xml")
orgsource = None
