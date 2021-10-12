#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set -ex

# Enable zip outputs
ENABLE_ZIP_OUTPUT=${ENABLE_ZIP_OUTPUT:-true}

#Check contents of pkl folder
PKL="/tmp/input/pkls"
find ${PKL} -type f | sed -e 's/.*\.//' | sort | uniq -c | sort -n | grep -Ei '(pkl|txt)$'
PKL_CONTENTS=$(ls -1 ${PKL})

# Input processing
WORKER_DIR="/home/worker"
WPS_PROPS="${WORKER_DIR}/workDir/FSTEP-WPS-INPUT.properties"
PROCESSOR_DIR="${WORKER_DIR}/processor"
SOURCE="${WORKER_DIR}/workDir/inDir"
SRC_FPRE="${SOURCE}/fpre"
SRC_FPOS="${SOURCE}/fpos"
POLYGON2NSEW="python ${PROCESSOR_DIR}/polygon2nsewBounds.py"
EXTRACT_SRS="python ${PROCESSOR_DIR}/extractSRS.py"

# Output directories
TARGET="${WORKER_DIR}/workDir/outDir/zipped"
PREPROCESSED="${WORKER_DIR}/workDir/outDir/preprocessed"
DNBR="${WORKER_DIR}/workDir/outDir/dNBR"
BAR="${WORKER_DIR}/workDir/outDir/burnt_area"
BAV="${WORKER_DIR}/workDir/outDir/fire_perimeter"
PROB="${WORKER_DIR}/workDir/outDir/ba_probability"
mkdir -p ${TARGET}
mkdir -p ${PREPROCESSED}
mkdir -p ${DNBR}
mkdir -p ${BAR}
mkdir -p ${BAV}
mkdir -p ${PROB}

# Load the inputs to bash variables
source ${WPS_PROPS}
AOI="${aoi}"

# Calculate CRS
FPRE_MTD=$(find -L "${SRC_FPRE}/" -maxdepth 3 -type f -name MTD_MSIL1C.xml | head -1)
JP2=$(dirname ${FPRE_MTD})/GRANULE/*/IMG_DATA/*_B02.jp2
CRS="EPSG:"$(${EXTRACT_SRS} ${JP2})
echo "${CRS}"

FPRE_SRC_PRODUCT=$(ls -1 ${SRC_FPRE})
FPOS_SRC_PRODUCT=$(ls -1 ${SRC_FPOS})

FPRE_PRODUCT_NAME=$(echo ${fpre} | cut -c 14-)
FPOS_PRODUCT_NAME=$(echo ${fpos} | cut -c 14-)

# Generate appropriate output folder name
S2A=$(echo ${FPRE_PRODUCT_NAME} | cut -c 1-3)
S2B=$(echo ${FPOS_PRODUCT_NAME} | cut -c 1-3)
FIRSTDATE=$(echo ${FPRE_PRODUCT_NAME} | cut -c 12-19)
SECONDDATE=$(echo ${FPOS_PRODUCT_NAME} | cut -c 12-19)
T55=$(echo ${FPRE_PRODUCT_NAME} | cut -c 39-44)
R030=$(echo ${FPRE_PRODUCT_NAME} | cut -c 34-37)
NAME="${S2A}_${FIRSTDATE}_${S2B}_${SECONDDATE}_${T55}_${R030}"
OUTPUT_NAME="EO4SD-Lab_AIBAM_${FIRSTDATE}_${SECONDDATE}_${T55}"

# Create temporary L1C directory
TEMP_INPUT=/tmp/input/L1C
mkdir -p ${TEMP_INPUT}

# move fpre to L1C folder and create empty AUX_DATA folder in case it has been stripped, to work around sen2cor issue.
mkdir -p ${TEMP_INPUT}/${FPRE_PRODUCT_NAME}
cp -a ${SRC_FPRE}/. ${TEMP_INPUT}/${FPRE_PRODUCT_NAME}/
mkdir -p ${TEMP_INPUT}/${FPRE_PRODUCT_NAME}/AUX_DATA

# move fpos to L1C folder and create empty AUX_DATA folder in case it has been stripped, to work around sen2cor issue.
mkdir -p ${TEMP_INPUT}/${FPOS_PRODUCT_NAME}
cp -a ${SRC_FPOS}/. ${TEMP_INPUT}/${FPOS_PRODUCT_NAME}/
mkdir -p ${TEMP_INPUT}/${FPOS_PRODUCT_NAME}/AUX_DATA

# Activate pyeo_env
conda activate pyeo_env

# If AOI specified calculate extents and run classify script adapted for aoi input
if [ "" != "${AOI}" ]; then
  AOI_EXTENTS=($(${POLYGON2NSEW} "${AOI}"))
  NORTH_BOUND=${AOI_EXTENTS[0]}
  SOUTH_BOUND=${AOI_EXTENTS[1]}
  EAST_BOUND=${AOI_EXTENTS[2]}
  WEST_BOUND=${AOI_EXTENTS[3]}
  UL=($(echo "${WEST_BOUND} ${NORTH_BOUND}" | cs2cs +init=epsg:4326 +to +init=epsg:${CRS#EPSG:}))
  LR=($(echo "${EAST_BOUND} ${SOUTH_BOUND}" | cs2cs +init=epsg:4326 +to +init=epsg:${CRS#EPSG:}))
  EMIN=${UL[0]}
  NMAX=${UL[1]}
  EMAX=${LR[0]}
  NMIN=${LR[1]}
else
  EMIN=None
  NMAX=None
  EMAX=None
  NMIN=None
fi

python /home/worker/processor/classify.py ${TEMP_INPUT}/${FPRE_PRODUCT_NAME} ${TEMP_INPUT}/${FPOS_PRODUCT_NAME} $HOME/bin/L2A_Process ${EMIN} ${NMIN} ${EMAX} ${NMAX}

# take outputs from:
PRODUCTS="/tmp/input/STACKED/${NAME}"

# enable zip output
if [ "$ENABLE_ZIP_OUTPUT" != "false" ]; then

  mkdir -p "${TARGET}/${OUTPUT_NAME}"
  cp -a "${PRODUCTS}/${NAME}_ba.tif"  "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_BA.tif"
  cp -a "${PRODUCTS}/${NAME}_dnbr.tif" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_dNBR.tif"
  cp -a "${PRODUCTS}/${NAME}_ba_proba.tif" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_BA_prob.tif"
  cp -a "${PRODUCTS}/${NAME}.dbf" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_FP.dbf"
  cp -a "${PRODUCTS}/${NAME}.prj" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_FP.prj"
  cp -a "${PRODUCTS}/${NAME}.shp" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_FP.shp"
  cp -a "${PRODUCTS}/${NAME}.shp.xml" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_FP.shp.xml"
  cp -a "${PRODUCTS}/${NAME}.shx" "${TARGET}/${OUTPUT_NAME}/${OUTPUT_NAME}_FP.shx"

  DST_PRODUCT=$(ls -1 ${TARGET})
  cd ${TARGET}/${DST_PRODUCT}
  zip -r ${TARGET}/${DST_PRODUCT}.zip .
  cd ${TARGET}
  # have two goes at deleting this - the first time occasionally is incomplete
  # though that may be a windows/docker only thing
  set +e
  rm -rf ${DST_PRODUCT}
  rm -rf ${DST_PRODUCT}
fi

# Also move products to individual outputs
# move tiff products
cp -a "${PRODUCTS}/${NAME}_ba.tif"  "${BAR}/${OUTPUT_NAME}_BA.tif"
cp -a "${PRODUCTS}/${NAME}_dnbr.tif" "${DNBR}/${OUTPUT_NAME}_dNBR.tif"
cp -a "${PRODUCTS}/${NAME}_ba_proba.tif" "${PROB}/${OUTPUT_NAME}_BA_prob.tif"

# Create directory for zipped shapefile
ZIP_SF="${BAV}/${OUTPUT_NAME}_FP"
mkdir -p ${ZIP_SF}

# copy the remaining shapefiles to ba_vector output
cp -a "${PRODUCTS}/${NAME}.dbf" "${ZIP_SF}/${OUTPUT_NAME}_FP.dbf"
cp -a "${PRODUCTS}/${NAME}.prj" "${ZIP_SF}/${OUTPUT_NAME}_FP.prj"
cp -a "${PRODUCTS}/${NAME}.shp" "${ZIP_SF}/${OUTPUT_NAME}_FP.shp"
cp -a "${PRODUCTS}/${NAME}.shp.xml" "${ZIP_SF}/${OUTPUT_NAME}_FP.shp.xml"
cp -a "${PRODUCTS}/${NAME}.shx" "${ZIP_SF}/${OUTPUT_NAME}_FP.shx"

# zip shapefile
DST_PRODUCT=$(ls -1 ${BAV})
cd ${BAV}/${DST_PRODUCT}
zip -r ${BAV}/${DST_PRODUCT}.zip .
cd ${BAV}
# have two goes at deleting this - the first time occasionally is incomplete
# though that may be a windows/docker only thing
set +e
rm -rf ${DST_PRODUCT}
rm -rf ${DST_PRODUCT}