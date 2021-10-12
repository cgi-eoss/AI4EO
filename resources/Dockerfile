FROM continuumio/miniconda3

MAINTAINER EO4SD-LAB

# installing Sen2Cor
ARG SEN2COR_VERSION=2.9.0
ARG PADDED_SEN2COR_VERSION=02.09.00

RUN apt-get update && apt-get install -y\
    bash \
    file \
    curl \
    unzip \
    zip

ENV INSTALLER_URL http://step.esa.int/thirdparties/sen2cor/${SEN2COR_VERSION}/Sen2Cor-${PADDED_SEN2COR_VERSION}-Linux64.run

RUN curl --location --output $HOME/installer.run ${INSTALLER_URL} \
    && chmod 755 $HOME/installer.run \
    && $HOME/installer.run --target $HOME \
    && rm $HOME/installer.run

# Clone and create the environments:
RUN git clone https://github.com/clcr/pyeo.git
WORKDIR "pyeo"
RUN conda env create --file environment.yml --name pyeo_env

# Make RUN commands to use the new environment:
RUN echo "conda activate pyeo_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN python -m pip install -e . -vv
RUN python -m pip install geopy

# Demonstrate the environment is activated:
RUN echo "Make sure pyeo is installed:"
RUN python -c "import pyeo; print('working')"

# For storing executable files inside container
RUN mkdir -p /home/worker/processor/
RUN mkdir -p /tmp/input/pkls/

# clone processing script and pkl files from AI4EO project repository
ARG CACHE_DATE=2021-09-28
RUN git clone https://github.com/cgi-eoss/AI4EO.git
RUN cp -a AI4EO/AI4EO_Classification_with_AOI_v1-2.py /home/worker/processor/classify.py
RUN unzip AI4EO/pkls.zip -d /tmp/input

#COPY AI4EO_Classification_with_AOI_v1-2.py /home/worker/processor/classify.py
COPY extractSRS.py polygon2nsewBounds.py L2A_GIPP.xml workflow.sh /home/worker/processor/

RUN chmod +x "/home/worker/processor/workflow.sh"
ENTRYPOINT ["/home/worker/processor/workflow.sh"]