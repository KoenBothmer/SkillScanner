from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

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
import plotly.graph_objects as go

k31_full = pickle.load(open('k_31_full', 'rb'))
cluster_label_bigrams = pickle.load(open('cluster_label_bigrams','rb')) 
cluster_importance = pickle.load(open('cluster_importance', 'rb'))

df_cv_summary = pd.read_csv('df_cv_summary.csv',index_col=0)

file = open("text/intro.txt")
intro = file.read()#.replace("\n", " ")

intro = intro.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/plot_explanation.txt")
plot_explanation = file.read()
plot_explanation = plot_explanation.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/per_skill_description.txt")
per_skill_description = file.read().replace("/n", " ").replace("\br", "\n")
per_skill_description = per_skill_description.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/table_description.txt")
table_description = file.read().replace("/n", " ").replace("\br", "\n")
table_description = table_description.encode('latin-1', 'replace').decode('latin-1')
file.close()

def return_pdf(skills):
    analysis = get_plot(skills)
    plot = analysis[0]
    df = analysis[1]
    df_sim = analysis[2]
    cv_scores = analysis[3]
    
    plot.write_image("fig1.png", scale=1)
    
    table = get_table(cv_scores)
    table.write_image("table1.png")
    
    total_score = df['score'].sum()/len(k31_full.cluster_centers_)*100
    overview = get_overview(total_score)
    overview.write_image("overview1.png")
    
    pdf=FPDF('P', 'mm', 'A4')
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 14) #setting font for title
    pdf.cell(40, 0, 'Skill Scanner CV Report - Data Scientist', ln=2) #Write Title
    pdf.set_font('Arial', '', 9) #setting font for text cells
    pdf.set_xy(10,15) #place cursor
    pdf.multi_cell(w=190, h=5, txt=intro, align='J')
    
    #Total Score Output
    total_score_s = str(round(total_score,2))
    competition_mean = 49
    competition_mean_s = "49"
    top10 = 72
    top10_s = "72"
    top25 = 67
    top25_s = "67"
    top50 = 61
    top50_s = "61"
    text = "Your total score is "+total_score_s+" This is "
    if total_score<competition_mean:
        text = text+" below average for a Data Scientist CV:"
    elif total_score<top25:
        text = text+"an average score for a Data Scientist CV:"
    else:
        text = text+"a high score for a Data Scientist CV:"
    text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Your Score: '+total_score_s+"%", ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=text)
    pdf.image('overview1.png', w=200)
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Comparison Plot', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=plot_explanation)
    
    pdf.image('fig1.png', w=200)#x = pdf.get_x, y = 15, w = 200)#, h = 200, type = '', link = '')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Analysis per input skill', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=per_skill_description)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    
    for index, row in df.iterrows():
        mean_score = df_cv_summary[df_cv_summary['cluster']==row['cluster']]['mean'].mean()
        mean_score_s = str(round(mean_score,2))
        score = round(row['score'],2)
        text = "Input Skill "+str(index+1)+":\nYour input skill \""+row['skill']+"\" was clustered in cluster "+str(row['cluster'])+\
        " which contains skills regarding "+cluster_label_bigrams[row['cluster']]+\
        ". Your score for this skill is "+str(score)+"."
        
        if score<mean_score:
            text = text + "This score is quite low, the average score among Data Scientist CV's is "+mean_score_s+" this may be due to a misclassification of our model but this could also indicate an opportunity to further clarify your CV."
        else:
            text = text + " this is above the average score among Data Scientist CV's which is "+mean_score_s
            
        text = text+"\n------------------------------------------------------------------\n"
        
        text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(w=190, h=5, txt = text)
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Your score compared to Data Scientist CV\'s', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=table_description)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    pdf.image('table1.png',w=190)
    return pdf
   
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
    
    df_report = pd.DataFrame()
    df_report['skill']=cv
    df_report['cluster']=clusters_cv_l
    df_report['score']=cv_scores
    
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
    
    df_sim['Your Coverage'] = df_sim['importance']*df_sim['score']
    df_sim['Skill Group Importance'] = df_sim['importance']-df_sim['Your Coverage']
    df_sim = df_sim.sort_values('importance')
    
    fig = px.bar(df_sim, y='labels', x=["Your Coverage","Skill Group Importance"], hover_data = ['importance'])
    fig.update_layout(height=2.3*300, width=3*300, \
                          #font=dict(size=10),\
                          title = 'Input CV Similarity to Requirements in Job Postings',\
                          barmode='stack', \
                          legend_title_text = '', \
                          yaxis_title="Importance",\
                          xaxis_title="Coverage",\
                          legend=dict(yanchor="bottom",y=0,xanchor="right",x=1
                        
                ))
    fig.update_xaxes(showticklabels=False)
    return fig, df_report, df_sim, scores
    
def get_table(scores):
    scores = [round(num, 2) for num in scores]
    
    labels = cluster_label_bigrams
    
    mean = df_cv_summary['mean'].tolist()
    mean = [round(num, 2) for num in mean]
    
    top10 = df_cv_summary['top10'].tolist()
    top10 = [round(num, 2) for num in top10]
    
    top25 = df_cv_summary['top25'].tolist()
    top25 = [round(num, 2) for num in top25]
    
    top50 = df_cv_summary['top50'].tolist()
    top50 = [round(num, 2) for num in top50]
    
    fig = go.Figure(data=[go.Table(
        columnwidth = [1100,100],
        header=dict(values=['Skill Cluster','Your Score','Average','Top 10%','Top 25%', 'Top 50%']),
        cells=dict(values=[labels, scores,mean,top10,top25,top50]))])
    
    fig.update_layout(height = 4*300, width = 3*300)
    
    return(fig)
    
def get_overview(score):
    
    fill_color = []
    n = 5

    vals = [round(score, 2),72,67,61,49]
    
    fill_color = []
    for v in vals:
        if v<=vals[4]:#less then average
            fill_color.append('#ffcccc')
        elif v<vals[3]:#less than top50
            fill_color.append('#fff5e6')
        elif v<vals[2]:#less than top25
            fill_color.append('#e6ffb3')
        else:
            fill_color.append('#ccffcc')
        
    vals = ['<b>'+str(round(score, 2))+'</b>',49,72,67,61]    
    
    
    fig = go.Figure(data=[go.Table(
        columnwidth = [100,100],
        header=dict(values=['<b>Your CV Coverage [%]</b>','Data Scientist<br>Top 10% [%]','Data Scientist <br>Top 25% [%]', 'Data Scientist<br>Top 50% [%]','Data Scientist<br>Average [%]']),
        cells=dict(values=vals, fill_color = fill_color)
    )])
    
    
    fig.update_layout(height = 0.9*300, width = 3*300)
    
    return(fig)
