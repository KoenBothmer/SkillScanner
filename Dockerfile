FROM jupyter/datascience-notebook:r-4.0.3
USER $NB_UID
RUN pip install nltk
RUN pip install gensim
RUN pip install dtale
RUN pip install langdetect
RUN pip install matplotlib-venn
RUN pip install matplotlib_venn_wordcloud
RUN pip install translate
RUN pip install geopy
RUN pip3 install sent2vec --no-cache-dir
RUN pip install sentence-transformers
RUN pip install torch
RUN pip install pdfplumber