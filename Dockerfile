FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
      libglib2.0-0 libxext6 libsm6 libxrender1 \
      git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O ~/anaconda.sh && \
      /bin/bash ~/anaconda.sh -b -p /opt/conda && \
      rm ~/anaconda.sh && \
      ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
      echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
      echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
      TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
      curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
      dpkg -i tini.deb && \
      rm tini.deb && \
      apt-get clean

RUN apt update
RUN apt install -y vim build-essential

RUN spt install cuda-tool-kit

COPY environment.yaml /
RUN conda env create -f environment.yaml || conda env update -f environment.yaml
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN pip install pip==21.0.1
RUN pip install awscli
RUN pip install packaging==21.1
RUN pip install numpy
RUN pip install --ignore-installed git+https://github.com/huggingface/transformers 
RUN pip install sentencepiece
RUN pip install protobuf

RUN mkdir /src
RUN mkdir /src/data
WORKDIR /src

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
