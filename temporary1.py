from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# Importing our resource database.
df = pd.read_csv('resources.csv')
df.head()

count = CountVectorizer()
count_matrix = count.fit_transform(df['tags'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df['name'])

def MLrecommend(inputR, cosine_sim=cosine_sim):
    idx = indices[indices == inputR].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top = list(score_series.iloc[1:4].index)

    for i in top:
        recommendations.append(list(df['name'])[i])

    dfR = df[df['name'].isin([recommendations[0], recommendations[1], recommendations[2]])]
    dfR = dfR[['name', 'url']]
    return dfR

@app.route('/recommend')
def recommend():
    inputR = request.args.get('inputR', 0, type=float)
    dfR = MLrecommend(inputR)
    dfR
    NEWrec = dfR.iloc[0]['name']
    return jsonify(recommendation='2')

if __name__ == "__main__":
    app.run()
