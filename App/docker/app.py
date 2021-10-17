from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
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
    Job_Title = SelectField(u'Select job title to analyze your skillset on', choices=['Data Scientist','more to come'])
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
    skill_16 = StringField('Optional additional skill')
    skill_17 = StringField('Optional additional skill')
    skill_18 = StringField('Optional additional skill')
    skill_19 = StringField('Optional additional skill')
    skill_20 = StringField('Optional additional skill')
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
        skills.append(form.skill_16.data)
        skills.append(form.skill_17.data)
        skills.append(form.skill_18.data)
        skills.append(form.skill_19.data)
        skills.append(form.skill_20.data)
  
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


# keep this as is
if __name__ == '__main__':
    app.run(debug=True)
