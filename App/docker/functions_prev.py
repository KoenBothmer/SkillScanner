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

file = open("text/intro.txt")
intro = file.read()#.replace("\n", " ")
#ntro = unicode(intro, 'utf-8')
#intro = intro.encode('iso-8859-1')
intro = intro.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/plot_explanation.txt")
plot_explanation = file.read()
#plot_explanation = unicode(plot_explanation, 'utf-8')
#plot_explanation = plot_explanation.encode('iso-8859-1')
plot_explanation = plot_explanation.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/per_skill_description.txt")
per_skill_description = file.read().replace("/n", " ").replace("\br", "\n")
per_skill_description = per_skill_description.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/table_description.txt")
table_description = file.read().replace("/n", " ").replace("\br", "\n")
#table_description = table_description.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/recommendation_description.txt")
recommendation_description = file.read().replace("/n", " ").replace("\br", "\n")
#recommendation_description = recommendation_description.encode('latin-1', 'replace').decode('latin-1')
file.close()

file = open("text/methodology_description.txt")
methodology_description = file.read().replace("/n", " ").replace("\br", "\n")
#methodology_description = methodology_description.encode('latin-1', 'replace').decode('latin-1')
file.close()

def return_pdf(skills):
    analysis = get_plot(skills)
    plot = analysis[0]
    df = analysis[1]
    df_sim = analysis[2]
    cv_scores = analysis[3]
    
    recommendation_plot=get_recommendation(df_sim)
    recommendation_plot.write_image("recommendation1.png",scale=1)
    
    plot.write_image("fig1.png", scale=1)
    
    table = get_table(cv_scores)
    table.write_image("table1.png")
    
    total_score = df['score'].sum()/len(k31_full.cluster_centers_)*100
    overview = get_overview(total_score)
    overview.write_image("overview1.png")
    
    pdf=FPDF('P', 'mm', 'A4')
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 14) #setting font for title
    pdf.cell(40, 0, 'Skill Scanner: CV Report', ln=2) #Write Title
    pdf.set_font('Arial', '', 9) #setting font for text cells
    pdf.set_xy(10,15) #place cursor
    pdf.multi_cell(w=190, h=5, txt=intro, align='J')
    
    #sketch visual
    pdf.set_font('Arial', 'I', 9) #setting font for text cells
    pdf.set_xy(pdf.get_x()+50, pdf.get_y()+5)
    pdf.image('img/sketch.png',w=100)
    pdf.multi_cell(w=190, h=5, txt = 'Skill Scanner uses AI to extract and compare skills from three sources')
    
    #Total Score Output
    total_score_s = str(round(total_score))
    competition_mean = 49
    competition_mean_s = "49"
    top10 = 72
    top10_s = "72"
    top25 = 67
    top25_s = "67"
    top50 = 61
    top50_s = "61"
    text = "Your total score is "+total_score_s+" This is"
    if total_score<competition_mean:
        text = text+" below average for a Data Scientist CV:"
    elif total_score<top25:
        text = text+"an average score for a Data Scientist CV:"
    else:
        text = text+"a high score for a Data Scientist CV:"
    text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_xy(10, pdf.get_y()+5)
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Your CV Coverage Score: '+total_score_s+"%", ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=text)
    pdf.image('overview1.png', w=200)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(w=190, h=5, txt='** Please note: In this example you can only input 5 skills, with more input skills the coverage will increase.')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Comparison to Demand', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=plot_explanation)
    
    pdf.image('fig1.png', w=200)#x = pdf.get_x, y = 15, w = 200)#, h = 200, type = '', link = '')
    
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(w=190, h=5, txt = '** Please Note: A coverage of 100% is impossible to attain, a coverage of over 70% can be considered excellent.')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Comparison to Competition', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=table_description)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    pdf.image('table1.png',w=190)
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Recommendation for Education', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=recommendation_description)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    pdf.image('recommendation1.png',w=200)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(w=190, h=5, txt='** In the future more learning content will be analyzed.')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Appendix A: Analysis per input skill', ln=2) #Write Title
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=per_skill_description)
    pdf.set_xy(pdf.get_x(),pdf.get_y()+5)
    
    for index, row in df.iterrows(): #Per Skill Analysis
    
        mean_score = df_cv_summary[df_cv_summary['cluster']==row['cluster']]['mean'].mean()
        mean_score_s = str(round(mean_score,2))
        score = round(row['score'],2)
        text = "Input Skill "+str(index+1)+": \""+row['skill']+"\"\nWas clustered in skill set \""+cluster_labels_unformated[row['cluster']]+\
        "\". Your similarity score for this skill is "+str(score)+"."
        
        if score<mean_score:
            text = text + "This score is quite low, the average score among Data Scientist CV's is "+mean_score_s+" this may be due to a misclassification of our model but this could also indicate an opportunity to further clarify your CV."
        else:
            text = text + " this is above the average score among Data Scientist CV's which is "+mean_score_s +"."
            
        text = text+"\n------------------------------------------------------------------\n"
        
        text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(w=190, h=5, txt = text)
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11) #setting font for title
    pdf.cell(40, 0, 'Appendix B: Methodology Explanation', ln=2) #Write Title
    
    #sketch workflow
    pdf.set_font('Arial', 'I', 9) #setting font for text cells
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.image('img/workflow.png',w=190)
    pdf.multi_cell(w=190, h=5, txt = 'Functionality sketch of Skill Scanner backend')
    pdf.set_font('Arial', '', 9)
    pdf.set_xy(pdf.get_x(), pdf.get_y()+5)
    pdf.multi_cell(w=190, h=5, txt=methodology_description)
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
                          title = 'Input CV Similarity to Requirements in Job Postings',\
                          barmode='stack', \
                          legend_title_text = '', \
                          yaxis_title="Importance ->",\
                          xaxis_title="Coverage ->",\
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
    
    
    fig.update_layout(height = 1*300, width = 3*300,margin=go.layout.Margin(l=10, r=10,b=0, t=0 ))
    
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
    #df_cv['coverage of recommendation'] = df_cv['coverage of recommendation']*100
    df_cv['Recommendation Importance']=(df_cv['Skill Set Importance']-df_cv['coverage of recommendation'])
    #df_cv['Recommendation Importance'] = df_cv['Recommendation Importance']*100
    df_cv=df_cv[df_cv['recommendation_score']>0]
    df_cv=df_cv.sort_values('Skill Set Importance',ascending=False)
            
    fig = px.bar(df_cv.head(5), y='recommendation_label', x=["coverage of recommendation","Recommendation Importance"])
    fig.update_layout(height=2*300, width=3*300, \
                         #font=dict(size=10),\
                         title = 'Top 5 Study Program Module Recommendations <br>Program: IU International University - Data Scientist 60ECT',\
                         barmode='stack', \
                         legend_title_text = '', \
                         yaxis_title="Importance ->",\
                         xaxis_title="Coverage ->",\
                         legend=dict(yanchor="bottom",y=0,xanchor="right",x=1))
    fig.update_xaxes(showticklabels=False)                     
    return fig