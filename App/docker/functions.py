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
cluster_labels_unformated = []
for label in cluster_label_bigrams:
    cluster_labels_unformated.append(label.replace("</b>","").replace("<b>",""))

df_cv_summary = pd.read_csv('df_cv_summary.csv',index_col=0)
df_handbook = pd.read_csv('handbook.csv')

def read_text(file):
    file = open(file)
    text = file.read()
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return(text)

def return_pdf(skills):
    analysis = get_plot(skills)
    plot = analysis[0]
    df = analysis[1]
    df_sim = analysis[2]
    cv_scores = analysis[3]
    
    recommendation_plot=get_recommendation(df_sim)
    recommendation_plot.write_image("recommendation1.png",scale=1)
    
    plot.write_image("fig1.png", scale=1)
    
    
    total_score = df['score'].sum()/len(k31_full.cluster_centers_)*100
    overview = get_overview(total_score)
    overview.write_image("overview1.png")
    
    score_visual = get_score_visual(total_score)
    score_visual.write_image("score1.png")
    
    pdf=FPDF('P', 'mm', 'A4')
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 20) #setting font for title
    pdf.cell(40, 0, 'Skill Scanner: CV Review for JOB SEEKERS', ln=2) #Write Title
    pdf.set_font('Arial', '', 9) #setting font for text cells
    pdf.set_xy(10,25) #place cursor
    
    intro = read_text('text/intro.txt')
    pdf.multi_cell(w=190, h=5, txt = intro, align='J')
    
    #Total Score Output
    total_score_s = str(round(total_score))

    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 18) #setting font for title
    pdf.cell(40, 0, '1.  Comparison to Competition') #Write Title
    
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'See how you rank up against other JOB SEEKERS\' CV\'s.')
    
    pdf.set_xy(10, pdf.get_y()+15)
    pdf.set_font('Arial','B',11)
    pdf.cell(40, 0, "Your total CV Coverage Score is:")
    pdf.set_xy(pdf.get_x() +30, pdf.get_y()-7.5)
    pdf.image('score1.png', w=50)
    
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'What is a good Coverage Score?')
    
    pdf.set_xy(10, pdf.get_y()+5)
    pdf.image('overview1.png', w=200)
    
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'What do the scores mean?')

    pdf.set_xy(10, pdf.get_y()+5)
    pdf.image('img/bins_legend.png',w=190)
    
    pdf.set_xy(10,pdf.get_y()+5)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(w=190, h=5, txt='** Note: In this example you can only input 5 skills, with more input skills the coverage score will increase.')
    pdf.set_xy(10, pdf.get_y()+5)
    
    pdf.add_page()
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 18) #setting font for title
    pdf.cell(40, 0, '2. Fit to Demand', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(10, pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=read_text('text/plot_explanation.txt'))
    
    pdf.image('fig1.png', w=200)#x = pdf.get_x, y = 15, w = 200)#, h = 200, type = '', link = '')
    
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(w=190, h=5, txt = '** Note: A coverage of 100% is impossible to attain, a coverage of over 70% can be considered excellent.')
    
    
    pdf.add_page()
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 18) #setting font for title
    pdf.cell(40, 0, '3.  Find and Select') #Write Title
    
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Choose the right study program for YOU')
    
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(10, pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=read_text('text/recommendation_description.txt'))
    
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    pdf.image('recommendation1.png',w=200)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(w=190, h=5, txt='** Note: In the future more learning content will be analyzed.')
    
    pdf.set_xy(10, pdf.get_y()+10)
    pdf.set_font('Arial', '', 9)
    pdf.cell(40, 0, 'Thank you for reading, please remember to fill out our questionnaire at https://forms.gle/ct4DSno6UxN4qofu8')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Appendix A: Methodology Explanation', ln=2) #Write Title
    
    #sketch workflow
    pdf.set_font('Arial', 'I', 9) #setting font for text cells
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.image('img/workflow.png',w=190)
    pdf.multi_cell(w=190, h=5, txt = 'Functionality sketch of Skill Scanner backend')
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=read_text('text/methodology_description.txt'))
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
        
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
    df_sim['Skill Set Importance'] = df_sim['importance']-df_sim['Your Coverage']
    df_sim = df_sim.sort_values('importance')
    
    fig = px.bar(df_sim, y='labels', x=["Your Coverage","Skill Set Importance"], hover_data = ['importance'])
    fig.update_layout(height=2.3*300, width=3*300, \
                          #font=dict(size=10),\
                          #title = 'Input CV Similarity to Requirements in Job Postings',\
                          barmode='stack', \
                          legend_title_text = '', \
                          yaxis_title="Skill Set Importance ──────»",\
                          xaxis_title="Coverage ──────»",\
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
    
    fill_color = []
    for score in scores:
        if score<=top50[scores.index(score)]:#less then average
            fill_color.append('#ffcccc')
        elif score<=top25[scores.index(score)]:
            fill_color.append('#fff5e6')
        elif score<=top10[scores.index(score)]:
            fill_color.append('#e6ffb3')                 
        else:
            fill_color.append('#ccffcc')
    
    print(fill_color)
    fig = go.Figure(data=[go.Table(
        columnwidth = [1100,100],
        header=dict(values=['Skill Cluster','Your Score','Average','Top 10%','Top 25%', 'Top 50%']),
        
        cells=dict(values=[labels, scores,mean,top10,top25,top50],
              fill = dict(color=['rgb(245, 245, 255)',fill_color])
                  )
    )])
    fig.update_layout(height = 3.4*300, width = 3*300,margin=go.layout.Margin(l=0, r=0,b=0, t=0 ))
    
    return(fig)
    
def get_overview(score):
    vals = [49,61,67]
    relative_vals = [49,12,6,33]
    y=[1,1,1,1]
    color=['#ffcccc','#fff5e6','#e6ffb3','#ccffcc']
    
    fig = px.bar(x=relative_vals,y=y, orientation = 'h', color = color, color_discrete_sequence=['#ffcccc','#fff5e6','#e6ffb3','#ccffcc'])
    
    fig.update_layout(barmode="stack", \
                      plot_bgcolor = 'rgba(0,0,0,0)', \
                      paper_bgcolor = 'rgba(0,0,0,0)', \
                      showlegend=False,\
                      yaxis_title="",\
                      xaxis_title="",         
                      margin=dict(l=0,r=0,b=0,t=0),
                      height=150
                     )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    fig.add_annotation(x=score,
    text="Your Score: <b>"+str(round(score)) + '</b>%', y=1.6,
    font=dict(size=25),
    showarrow=False)
    
    fig.add_annotation(x=score,y=1,
    text="────────────",
    showarrow=False,
    textangle=-90)
    
    #for val in vals:
    #    fig.add_annotation(x=vals[vals.index(val)],#y=0,
    #    font=dict(size=25),
    #    text = str(val),
    #    yshift = 0,                   
    #    showarrow=True,
    #    arrowhead=1)
    
    fig.add_annotation(x=vals[0], y=1,
            text="Needs Improvement",
            showarrow=False,
            xshift=-100)
    fig.add_annotation(x=vals[1], y=1,
            text="Fair",
            showarrow=False,
            xshift=-60)
    fig.add_annotation(x=vals[2], y=1,
            text="Good",
            showarrow=False,
            xshift=-20)
    fig.add_annotation(x=vals[2], y=1,
            text="Excellent",
            showarrow=False,
            xshift=50)
    return(fig)

def get_recommendation(df_cv):
    recommendations = []
    recommendation_scores=[]
    recommendation_labels = []

    for index, row in df_cv.iterrows():
        if row['score']==0:
            df_handbook_filtered = df_handbook[df_handbook['cluster']==row['cluster']]
            if len(df_handbook_filtered)>0 and df_handbook_filtered['score'].max()>0.5:
                recommendation = df_handbook_filtered['score'].argmax()
                recommendation = df_handbook_filtered['module'].iloc[recommendation]
                recommendation_label = "Recommended Module: <b>"+recommendation+'</b><br>Skill Set: '+row['labels']
                
                recommendations.append(recommendation)
                recommendation_scores.append(df_handbook_filtered['score'].max())
                recommendation_labels.append(recommendation_label)
            else:
                recommendations.append(None)
                recommendation_scores.append(0)
                recommendation_labels.append(None)
        else:
            recommendations.append(None)
            recommendation_scores.append(0)
            recommendation_labels.append(None)
    df_cv['recommendation']=recommendations
    df_cv['recommendation_score']=recommendation_scores
    df_cv['recommendation_label']=recommendation_labels
    df_cv['coverage of recommendation'] = (df_cv['recommendation_score']*df_cv['Skill Set Importance'])
    df_cv['Recommendation Importance']=(df_cv['Skill Set Importance']-df_cv['coverage of recommendation'])
    df_cv=df_cv[df_cv['recommendation_score']>0]
    df_cv=df_cv.sort_values('Skill Set Importance',ascending=True)
            
    fig = px.bar(df_cv.head(5), y='recommendation_label', x=["coverage of recommendation","Recommendation Importance"])
    fig.update_layout(height=2*300, width=3*300, \
                         #font=dict(size=10),\
                         title = 'Top 5 Study Program Module Recommendations <br>Program: IU International University - Data Scientist 60ECT',\
                         barmode='stack', \
                         legend_title_text = '', \
                         yaxis_title="Skill Set Importance ──────»",\
                         xaxis_title="Recommendation Coverage of Skill Set ──────»",\
                         legend=dict(yanchor="bottom",y=0,xanchor="right",x=1))
    fig.update_xaxes(showticklabels=False)                     
    return fig

def get_score_visual(score):
    
    vals = [0,49,61,67]
    
    x=[score]
    y=[1]

    color=['#ffcccc','#fff5e6','#e6ffb3','#ccffcc']
    if score>vals[3]:
        color = ['#ccffcc']
    elif score>vals[2]:
        color = ['#e6ffb3']
    elif score>vals[1]:
        color = ['#fff5e6']
    else:
        color = ['#ffcccc']
    
    fig = px.bar(x=x,y=y, orientation = 'h', color_discrete_sequence = color)
    
    fig.update_layout(barmode="stack", \
                      plot_bgcolor = 'rgba(0,0,0,0)', \
                      paper_bgcolor = 'rgba(0,0,0,0)', \
                      showlegend=False,\
                      yaxis_title="",\
                      xaxis_title="",         
                      margin=dict(l=0,r=0,b=0,t=0),
                      height=100,
                      width =300
                     )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    
    fig.add_annotation(y=1,
            font=dict(size=30),
            text=str(round(score))+"% Coverage",
            showarrow=False)

    return(fig)