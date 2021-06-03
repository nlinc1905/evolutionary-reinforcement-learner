FROM ubuntu:18.04

# Install Linux dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-setuptools python3-pip git

COPY . /app
WORKDIR /app

# Install Python libraries
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt --user

# Install Pygame Learning Environment (PLE)
RUN git clone https://github.com/ntasfi/PyGame-Learning-Environment && \
    cd PyGame-Learning-Environment && \
    python3 -m pip install --user -e .
