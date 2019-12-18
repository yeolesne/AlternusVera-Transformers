import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pickle
import nltk
nltk.download('wordnet')

class CredibilityFactChecks():
  def lemmatize_stemming(self, text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

  def preprocess(self, text):
    result = []
    cf = CredibilityFactChecks()
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(cf.lemmatize_stemming(token))
    return result
  
  def predict(text):
    model = pickle.load(open('/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Snehal/Credibilty/Model/Transformers_Factor_Credibility-Fact-Checks.pkl','rb'))
    cf = CredibilityFactChecks()
    cf.preprocess(text)
    prediction = model.predict([text])
    print(prediction)
    prob = model.predict_proba([text])[:,1]
    print(prob)
    return bool(prediction), float(prob)
