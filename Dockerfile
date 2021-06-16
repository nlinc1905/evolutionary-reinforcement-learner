FROM ubuntu:18.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"

# Install Linux dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-setuptools python3-pip git wget unar unzip

COPY . /app
WORKDIR /app

# Install Python libraries
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt --user

# Install Pygame Learning Environment (PLE)
RUN git clone https://github.com/ntasfi/PyGame-Learning-Environment && \
    cd PyGame-Learning-Environment && \
    python3 -m pip install --user -e .

# Download Atari 2600 ROMs from http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
# Import them using the atari-py library
RUN wget http://www.atarimania.com/roms/Roms.rar && \
    unar Roms.rar && \
    cd Roms && \
    unzip ROMS.zip && \
    python3 -m atari_py.import_roms ROMS

