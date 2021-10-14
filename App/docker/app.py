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

k31_full = pickle.load(open('k_31_full', 'rb'))
cluster_label_bigrams = pickle.load(open('cluster_label_bigrams','rb')) 
cluster_importance = pickle.load(open('cluster_importance', 'rb'))

app = Flask(__name__)

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'curriculum'

# Flask-Bootstrap requires this line
Bootstrap(app)

# with Flask-WTF, each web form is represented by a class
# "NameForm" can change; "(FlaskForm)" cannot
# see the route for "/" and "index.html" to see how this is used
class NameForm(FlaskForm):
    skill_1 = StringField('Just pass the first skill', validators=[DataRequired()])
    skill_2 = StringField('Just pass the second skill', validators=[DataRequired()])
    skill_3 = StringField('Optional additional skill')
    skill_4 = StringField('Optional additional skill')
    skill_5 = StringField('Optional additional skill')
    skill_6 = StringField('Optional additional skill')
    skill_7 = StringField('Optional additional skill')
    skill_8 = StringField('Optional additional skill')
    skill_9 = StringField('Optional additional skill')
    skill_10 = StringField('Optional additional skill')
    skill_11 = StringField('Optional additional skill')
    skill_12 = StringField('Optional additional skill')
    skill_13 = StringField('Optional additional skill')
    skill_14 = StringField('Optional additional skill')
    skill_15 = StringField('Optional additional skill')
    submit = SubmitField('Download Report')

# all Flask routes below

@app.route('/', methods=['GET', 'POST'])
def index():

    # you must tell the variable 'form' what you named the class, above
    # 'form' is the variable name used in this template: index.html
    form = NameForm()
    message = ""
    skills = []
    if form.validate_on_submit():
        skills.append(form.skill_1.data)
        skills.append(form.skill_2.data)
        skills.append(form.skill_3.data)
        skills.append(form.skill_4.data)
        skills.append(form.skill_5.data)
        skills.append(form.skill_6.data)
        skills.append(form.skill_7.data)
        skills.append(form.skill_8.data)
        skills.append(form.skill_9.data)
        skills.append(form.skill_10.data)
        skills.append(form.skill_11.data)
        skills.append(form.skill_12.data)
        skills.append(form.skill_13.data)
        skills.append(form.skill_14.data)
        skills.append(form.skill_15.data)
  
        pdf = return_pdf(skills)
        pdf.output('report.pdf', 'F')
        return send_file('report.pdf', as_attachment=True)
    return render_template('index.html', form=form, message=message)

# 2 routes to handle errors - they have templates too

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

def return_pdf(skills):
    plot = get_plot(skills)
    plot.write_image("fig1.png")
    pdf=FPDF()
    pdf.add_page()
    pdf.image('fig1.png',x = 10, y = 10, w = 100, h = 100, type = '', link = '')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 0, 'Data Scientist CV Review Report')
    pdf.set_font('Arial', '', 12)
    pdf.cell(20,250, 'This is a v0 test document to see if we can pass visuals and text to pdf through flask based on form input', align = "L", fill = 'T', border='1')
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

# keep this as is
if __name__ == '__main__':
    app.run(debug=True)
