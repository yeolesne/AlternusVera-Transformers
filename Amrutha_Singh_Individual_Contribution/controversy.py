from google.colab import drive
from sklearn.externals import joblib
import pickle
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import spacy
nlp = spacy.load('en')

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

class ControversyFactor():
  def __init__(self):
    drive.mount('/content/drive')
    self.model_path = '/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Amrutha/Controversy_Speaker_Factors/Models/Transformers_Factor_Controversy.pkl'
    self.data_path = '/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Amrutha/Controversy_Speaker_Factors/Datasets/test.tsv'

  def load_data(self):
    test_data = pd.read_csv(self.data_path, sep='\t', header = None)
    clean_texts = [clean_corpus(str(doc)) for doc in test_data['statement']]
    nlp_docs = [nlp(clean_text) for clean_text in clean_texts]
    sentence = list(nlp_docs[23].sents)[0]


    for doc in nlp_docs:
      filtered_text = ''
    for sentence in doc.sents:
      sent_filt_text = ' '.join([token.lemma_
                                   for token in sentence if (token.tag_ in tags_to_keep
                                                             and not token.is_stop
                                                             and not token.ent_type_ in entities_to_remove
                                                            )])
    filtered_text = filtered_text + ' ' + sent_filt_text
    test_data['stemmed'] = test_data["filtered_text"].apply(preprocess)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(test_data['stemmed'])
    X_test = bow_transformer.transform(test_data['stemmed'])
    return X_test

  def load_model(self):
    file = open(self.model_path, 'rb')
    self.model = pickle.load(file)

  def predict(self, data):
    return self.model.predict(data)

"""**Step 2:** Installing necessary packages"""

#!pip install textacy

import nltk
nltk.download('punkt')
nltk.download('stopwords')

"""**Step 3:** Data Pre-processing"""

file_name = '/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Amrutha/Controversy_Speaker_Factors/Models/Transformers_Factor_Controversy.pkl'

from sklearn.model_selection import train_test_split
import re
import textacy
from textacy import preprocessing

def preprocess(raw_news):
    import nltk
    news = re.sub("[^a-zA-Z]", " ", raw_news)
    news = re.sub(r"\"", "", news)
    news = re.sub(r"\'", "", news)
    news = re.sub(r"[0-9]", "digit", news)
    news =  news.lower()
    news_words = nltk.word_tokenize( news)
    stops = set(nltk.corpus.stopwords.words("english"))
    words = [w for w in  news_words  if not w in stops]
    stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in words]
    return " ".join(stems)

def clean_corpus(sentence):
    sentence = re.sub(r"\\\'", r' ', sentence)
    sentence = re.sub(r' s ', r" ", sentence)
    ## removing character sequences of Word files
    sentence = re.sub(r'(\\n|\\xe2|\\xa2|\\x80|\\x9c|\\x9c|\\x9d|\\t|\\nr|\\x93)', r' ', sentence)
    sentence = sentence.replace(r"b'", ' ') ## removing the beginning of some documents
    sentence = re.sub(r'\{[^\}]+\}', r' ', sentence) ## removing word macros

    sentence = textacy.preprocessing.replace_emails(sentence, replace_with=r' ') ## removing emails
    sentence = textacy.preprocessing.replace_urls(sentence, replace_with=r' ') ## removing urls

    sentence = re.sub(r'(#|$| [b-z] |\s[B-Z]\s|\sxxx\s|\sXXX\s|XXX\w+)', r' ', sentence)

    ## removing character sequences of pdf files
    sentence = re.sub(r'(\\x01|\\x0c|\\x98|\\x99|\\xa6|\\xc2|\\xa0|\\xa9|\\x82)', r' ', sentence)

    sentence = re.sub(r'(c/-d*)', r' ', sentence) #remove address

    sentence = re.sub(r'(\\x01|\\x0c|\\x98|\\x99|\\xa6|\\xc2|\\xa0|\\xa9|\\x82|\\xb7)', r' ', sentence)

    # removing trade mark of specific pdf file
    sentence = sentence.replace(r'LE G A SA L E M D PL O C E S', ' ')

    # striping consecutive white spaces
    sentence = re.sub(r'\s\s+', ' ', sentence).strip()
    sentence = sentence.strip()
    return sentence

from google.colab import drive
drive.mount('/content/drive')
path_train = '/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Amrutha/Controversy_Speaker_Factors/Datasets/train.tsv'
train_data = pd.read_csv(path_train, sep='\t', header = None)

columns = ['id', 'label', 'statement', 'subjects', 'speaker',
         'speaker_job', 'state', 'party', 'barely_true_counts',
         'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts',
         'context']
train_data.columns = columns

clean_texts = [clean_corpus(str(doc)) for doc in train_data['statement']]
nlp_docs = [nlp(clean_text) for clean_text in clean_texts]

filtered_texts = []
tags_to_keep = ['JJ', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
entities_to_remove = ['PERSON', 'NAME','GPE', 'NORP', 'FACILITY', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                      'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
for doc in nlp_docs:
    filtered_text = ''
    for sentence in doc.sents:
        sent_filt_text = ' '.join([token.lemma_
                                   for token in sentence if (token.tag_ in tags_to_keep
                                                             and not token.is_stop
                                                             and not token.ent_type_ in entities_to_remove
                                                            )])
        filtered_text = filtered_text + ' ' + sent_filt_text
    filtered_texts.append(filtered_text)

"""**Step 4:** Stemming and labelling"""

from gensim import corpora
import gensim
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer


stemmer = PorterStemmer()
vocab_doc = ["abuse administration afghanistan aid america american army attack attacks authorities authority ban banks benefits bill border budget campaign candidate candidates catholic china chinese church concerns congress conservative control country court crime criminal crisis cuts debate debt defense deficit democrats disease dollar drug drugs economy education egypt election elections enforcement fighting finance fiscal force funding gas government gun health immigration inaccuracies india insurance investigation investigators iran israel job jobs judge justice killing korea labor land law lawmakers laws lawsuit leadership legislation marriage media mexico military money murder nation nations news obama offensive officials oil parties peace police policies policy politics poll power president prices primary prison progress race reform republican republicans restrictions rule rules ruling russia russian school security senate sex shooting society spending strategy strike support syria syrian tax taxes threat trial unemployment union usa victim violence vote voters war washington weapons world account advantage amount attorney chairman charge charges cities class comment companies cost credit delays effect expectations families family february germany goal housing information investment markets numbers oklahoma parents patients population price projects raise rate reason sales schools sector shot source sources status stock store whether worth"]
vocab_doc_stem = "abus account administr advantag afghanistan aid america american amount armi attack attorney author ban bank benefit bill border budget campaign candid cathol chairman charg china chines church citi class comment compani concern congress conserv control cost countri court credit crime crimin crisi cut debat debt defens deficit delay democrat diseas dollar drug economi educ effect egypt elect enforc expect famili februari fight financ fiscal forc fund ga germani goal govern gun health hous immigr inaccuraci india inform insur invest investig iran israel job judg justic kill korea labor land law lawmak lawsuit leadership legisl market marriag media mexico militari money murder nation news number obama offens offici oil oklahoma parent parti patient peac polic polici polit poll popul power presid price primari prison progress project race rais rate reason reform republican restrict rule russia russian sale school sector secur senat sex shoot shot societi sourc spend statu stock store strategi strike support syria syrian tax threat trial unemploy union usa victim violenc vote voter war washington weapon whether world worth"
texts = ([list(set([stemmer.stem(word) for word in vocab_doc[0].split()]))])
dict_tokenized = gensim.corpora.Dictionary(texts)
dict_bow_corpus = [dict_tokenized.doc2bow(doc) for doc in texts]
lda_model = gensim.models.LdaMulticore(dict_bow_corpus, num_topics=10, id2word=dict_tokenized, passes=2, workers=2)
train_data['filtered_text']= filtered_texts
train_data['stemmed'] = train_data["filtered_text"].apply(preprocess)

LDA_score=[]
count=0
dictionary = corpora.Dictionary.load('/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Amrutha/Controversy_Speaker_Factors/Datasets/vocab_list.dict')
voc_bow = dictionary.doc2bow(vocab_doc_stem.lower().split())
vec_lda_voc = lda_model[voc_bow]
for doc in train_data['stemmed']:
    bow_vector_st = dictionary.doc2bow([preprocess(doc)])
    vec_lda_ex = lda_model[bow_vector_st]
    sim = gensim.matutils.cossim(vec_lda_voc, vec_lda_ex)
    LDA_score.append(sim)
    if (count > 3 and count < 10):
        #print(doc,'=>',sim)
        print ('\n')
    count+=1
train_data['LDA_controversy_score']= LDA_score

controversy_label=[]
for score in train_data['LDA_controversy_score']:
    if 0.2<score<0.4:
        controversy_label.append(1)   # non-controversial
    elif 0.4<score<0.7:
        controversy_label.append(2)
    elif score>0.7:
        controversy_label.append(3)    # strongly-controversial
    else:
        controversy_label.append(0)
train_data['controversy_label'] = controversy_label
X_train, X_test, y_train, y_test = train_test_split(train_data['stemmed'], train_data['controversy_label'], random_state = 0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""**Step 5:** Random Forest Classifer"""

from sklearn.ensemble import RandomForestClassifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train_tfidf,y_train)

y_pred=clf.predict(count_vect.transform(X_test))

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
pickle.dump(clf, open(file_name, 'wb'))

"""**Step 6:** Predict using the model. The output can be interpreted as follows -

If the output is 1 - The news is non-controversial

If the output is 3 - The news is controversial (Fake)
"""

import pickle

class ControversyFactor():
  def __init__(self):
    file_name = '/content/drive/My Drive/AlternusVeraDataSets2019/Transformers/Amrutha/Controversy_Speaker_Factors/Models/Transformers_Factor_Controversy.pkl'
    self.pickle_file = open(file_name, 'rb')

  def load_model(self):
    self.model = pickle.load(self.pickle_file)

  def predict(self, text):
    return self.model.predict(count_vect.transform(text))

cf = ControversyFactor()
cf.load_model()

data = ['Says the Annies List political group supports third-trimester abortions on demand.']
cf.predict(data)
