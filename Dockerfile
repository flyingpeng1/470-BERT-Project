FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
      libglib2.0-0 libxext6 libsm6 libxrender1 \
      git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
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

COPY environment.yaml /
RUN conda env create -f environment.yaml || conda env update -f environment.yaml

RUN pip install pip==21.0.1
RUN pip install awscli
RUN pip install packaging==21.1
RUN pip install numpy
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --ignore-installed git+https://github.com/huggingface/transformers 
RUN pip install sentencepiece
RUN pip install protobuf

RUN mkdir /src
RUN mkdir /src/data
WORKDIR /src

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
