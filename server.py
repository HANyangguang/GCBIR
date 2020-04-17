import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template

from annoy import AnnoyIndex

app = Flask(__name__)

b = AnnoyIndex(4096, 'angular')
b.load('static/feature/dataset.tree')
img_paths = pickle.load(open('static/feature/img_paths.pkl', 'rb'))

# Read image features
fe = FeatureExtractor()
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)
        query = fe.extract(img)
        ids, distances = b.get_nns_by_vector(query, 30, search_k=-1, include_distances=True)
        scores = [(distance, img_paths[id]) for (distance, id) in zip(distances, ids)]
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0", port=5000, debug=True)
