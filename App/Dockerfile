FROM jupyter/datascience-notebook:r-4.0.3
USER $NB_UID
COPY docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt