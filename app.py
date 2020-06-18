import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import newspaper
from newspaper import Article
import urllib



app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))


with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)