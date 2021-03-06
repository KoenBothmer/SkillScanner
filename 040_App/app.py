from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, FileField
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

from functions import return_pdf, get_plot, get_table

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
    Job_Title = SelectField(u'Job Title', choices=['Data Scientist','more to come'])
    skill_1 = StringField('Please enter at least 2 skills, skill 1', validators=[DataRequired()])
    skill_2 = StringField(validators=[DataRequired()])
    skill_3 = StringField()
    skill_4 = StringField()
    skill_5 = StringField()
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
        
        skills_ = skills
        skills=[]
        for skill in skills_: # remove empty skills
            if len(skill)>0:
                skills.append(skill)
  
        pdf = return_pdf(skills)
        pdf.output('report.pdf', 'F')
        
        return send_file('report.pdf', as_attachment=True)
        #return render_template('thanks.html', filename = 'report.pdf')#, as_attachment=True)
    return render_template('index.html', form=form, message=message)

@app.route('/sample_report')
def go_to_end():
    return send_file('sample_report.pdf', as_attachment=True)
    
# 2 routes to handle errors - they have templates too

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# keep this as is
if __name__ == '__main__':
    app.run(debug=True)
