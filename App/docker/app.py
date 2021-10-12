from flask import Flask
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import pickle

from fpdf import FPDF
from flask import send_file
import fpdf
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly as plt
import numpy as np
import plotly.express as px

k31_full = pickle.load(open('k_31_full', 'rb'))
cluster_label_bigrams = pickle.load(open('cluster_label_bigrams','rb')) 
cluster_importance = pickle.load(open('cluster_importance', 'rb'))

app = Flask(__name__)

@app.route('/')
def return_pdf():
    #model = 'all-distilroberta-v1'
    #model = SentenceTransformer(model)
    #embeddings_cv = model.encode(['one string','and another'])
    #k31_full = pickle.load(open('k_31_full', 'rb'))
    #centers = k31_full.cluster_centers_
    plot = get_plot(['i do stuff in sql','work hard play hard'])
    plot.write_image("fig1.png")
    pdf=FPDF()
    pdf.add_page()
    #pltx="fig1.png"
    pdf.image('fig1.png',x = 10, y = 10, w = 100, h = 100, type = '', link = '')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 0, 'Hello World! this string asdfasdf')
    pdf.set_font('Arial', '', 12)
    pdf.cell(20,250, 'And here some explanation but what happens if we add more text', align = "L", fill = 'T', border='1')
    pdf.output('report.pdf', 'F')
    return send_file('report.pdf', as_attachment=True)
    #return get_plot(['i do stuff in sql','work hard play hard'])
    
def get_plot(cv): #takes in list of skills and returns a plot with score for each cluster
    df_sim = pd.DataFrame()
    df_sim['cluster'] = range(len(k31_full.cluster_centers_))
    
    labels = cluster_label_bigrams
    importance = cluster_importance
    
    model = 'all-distilroberta-v1'
    model = SentenceTransformer(model)
    embeddings_cv = model.encode(cv)
    embeddings_f = embeddings_cv.astype(float)
    clusters_cv = k31_full.predict(embeddings_f)
    clusters_cv_l  = clusters_cv.tolist()
    
    cv_scores = []
    for i, cluster in enumerate(clusters_cv):
        cv_scores.append(util.pytorch_cos_sim(k31_full.cluster_centers_[cluster], embeddings_f[i]).item())
    
    scores = []
    for cluster in range(len(k31_full.cluster_centers_)):
        if cluster not in clusters_cv_l:
            scores.append(0)
        else:
            score = 0
            indexes = np.where(clusters_cv==cluster)[0]
            for i in indexes:
                if cv_scores[i] > score:
                    score = cv_scores[i]
            scores.append(score)   
    
    df_sim['score'] = scores
    df_sim['importance'] = importance
    df_sim['labels'] = labels
    
    df_sim['CV_similarity'] = df_sim['importance']*df_sim['score']
    df_sim['Importance_Cluster_in_Job_Postings'] = df_sim['importance']-df_sim['CV_similarity']
    df_sim = df_sim.sort_values('importance')
    
    fig = px.bar(df_sim, y='labels', x=["CV_similarity","Importance_Cluster_in_Job_Postings"], hover_data = ['importance'])
    fig.update_layout(height=800, \
                          title = 'Author\'s CV similarity to requirement clusters in context of relative cluster presence',\
                          barmode='stack', \
                          yaxis_title="3 most common Bigrams in Cluster",\
                          xaxis_title="CV Similarity to Cluster")
    fig.show()
    return fig