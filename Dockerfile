FROM quay.io/jupyter/minimal-notebook:2023-11-19

USER root

RUN apt-get update && apt-get install -y make

USER jovyan

RUN conda install -y pandas=2.1.2 \
    scikit-learn=1.3.2 \ 
    numpy=1.26 \
    matplotlib-base=3.8.1 \
    seaborn=0.13.0 \
    pytest=7.4.3 \
    jupyter-book=0.15.1 \
    click=8.1.7 