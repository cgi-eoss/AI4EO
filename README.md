# AI4EO
The code for the AI-enabled Burned Area Mapping Service.
This will be offered as a on-demand service on the EO4SD Lab.

To run locally:

-> Pre and Post .SAFE files should in the L1C folder.

-> The 'pkls' folder needs to be placed alongside the L1C directory.

Example:

/directory_1/L1C
/directory_1/pkls

-> The nearest/proximity available tile's .pkl will be selected from the 'pkls' folder for the RF classification.

-> Inputs needed in the terminal,

Example:

$ fpre=/scratch/satburn/uz6/IMAGE/L1C/S2A_MSIL1C_20191216T004701_N0208_R102_T53HPA_20191216T020938.SAFE

$ fpos=/scratch/satburn/uz6/IMAGE/L1C/S2A_MSIL1C_20191226T004701_N0208_R102_T53HPA_20191226T021317.SAFE

$ sen2cor_path=/scratch/satburn/uz6/Sen2Cor-02.08.00-Linux64/bin/L2A_Process

$ python classify_v1-1.py $fpre $fpos $sen2cor_path

-> Output includes the burned area map raster and shapefile, dNBR raster,
and other relevant files.