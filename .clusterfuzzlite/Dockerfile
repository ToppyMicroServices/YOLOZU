FROM gcr.io/oss-fuzz-base/base-builder-python

WORKDIR /src
COPY . /src

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install atheris pyinstaller
