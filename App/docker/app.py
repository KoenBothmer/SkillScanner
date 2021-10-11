from flask import Flask
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pickle
app = Flask(__name__)

@app.route('/')
def hello_world():
    model = 'all-distilroberta-v1'
    #model = SentenceTransformer(model)
    #embeddings_cv = model.encode(['one string','and another'])
    k31_full = pickle.load(open('k_31_full', 'rb'))
    centers = k31_full.cluster_centers_
    return str(centers)