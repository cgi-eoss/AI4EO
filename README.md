# AI4EO
The AI4EO project is an ESA funded project led by CGI, partnering with the
University of Leicester. The project is focussing on developing and demonstrating
a burned area (BA) mapping service that combines EO data, specifically Sentinel-2
optical data, with an AI-enabled algorithm.

The output products are:

- Raster burned area layer (GeoTIFF)
- Vector burned area perimeter layer (SHP)
- Raster dNBR layer (GeoTIFF)
- Raster burned probability layer (GeoTIFF)

This service is offered as a on-demand service on the [EO4SD Lab
](https://eo4sd-lab.net/).

**Requirements:**
- Sen2Cor version 2.9.0 - http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/
- Python For Earth Observation library (pyeo) - https://github.com/clcr/pyeo

**Usage:**

Any downloaded Sentinel-2 images (e.g. from SciHub) will be in .zip
format which should be extracted manually to get the. SAFE format.

The pre and post Level 1 images in. SAFE format must be stored in a single directory
which should be named as ‘L1C’.

The ‘pkls’ folder containing the pickle files for the trained models must be
alongside to the L1C folder.

Example:

`/directory_1/L1C`

`/directory_1/pkls`

The python script requires arguments of the pre-image L1C path, post-image L1C
path, and the path to the sen2cor processor.

Example:

`$ fpre=/directory_1/L1C
/S2B_MSIL1C_20200612T020449_N0209_R017_T51KVB_20200612T042940.SAFE`

`$ fpos=/directory_1/L1C
/S2A_MSIL1C_20200707T020451_N0209_R017_T51KVB_20200707T051834.SAFE`

`$ sen2cor_path=/scratch/satburn/uz6/Sen2Cor-02.08.00-Linux64/bin/L2A_Process`

Run the algorithm:

`python AI4EO_Prototype_Classification_v1-2.py $fpre $fpos $sen2cor_path`

To run the algorithm on a subset of the sentinel-2 image use run the
`AI4EO_Classification_with_AOI_v1-2.py` with further arguments of the North
, East, South and West extents of the subset in the CRS of the input scenes:

`python AI4EO_Classification_with_AOI_v1-2.py $fpre $fpos $sen2cor_path
<west_extent> <south_extent> <east_extent> <north_extent>` 