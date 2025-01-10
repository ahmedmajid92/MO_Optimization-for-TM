import numpy as np
import pandas as pd
import re
import random
from sklearn.feature_extraction.text import CountVectorizer
import operator
import math
import pprint
import nltk
#nltk.download('omw-1.4')

from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from stemming.porter2 import stem
import warnings
warnings.filterwarnings('ignore')

##################################################
# Read Dataset
##################################################

df = pd.read_csv('Datasets/papers.csv')
df= df.iloc[:50]

##################################################
# Make Stop Words
##################################################

stop_words = stopwords.words('english')
##Creating a list of custom stopwords
new_words = ["fig ","figure","image","sample","using","therefore","equation ","In fact","otherwise","summary","Theoretically","for example","In brief","Furthermore","general","practice","such ","that "," Later","experiment","Experiment","by","Overview "," diagram","describes"
             "show", "result", "large","Fig. ","e.g.","over","left","Next","After","about","That","illustrates","Let","Obviously","Diagram","showing ","denotes","Section","both","Formally","result","Additionally","Moreover","arbitrarily","consequently","Following","discussed"
             "also", "one", "two", "three","shown","right","Further","assume","where","e.p.s.p.","Such","small","Specifically","i.e.","Since","follows","Intuitively","Finally","especially","extensively","intuitively ","Of course","Alternatively","Theoretically","Definition"
             "four", "five", "seven","eight","nine", "(Fig. )","However", "usually", "Hence","Thus"," Here","elsewhere","Then", "arc", "new", "Empirically","Similarly","Equation","respectively","Qualitatively","Clearly","Finally","Theorem "," Typically","Lemma","results","demonstrate","false","true"]
stop_words = stop_words + new_words
stop_words = [x.lower() for x in stop_words]

##################################################
# Dataset Preprocessing
##################################################

corpus = []
list_of_paper_text = []
list_of_titles = []

correct_word_list = words.words()
word_set = set(correct_word_list)

# for each paper
for i in range(0,len(df.index)):

    text = df['paper_text'][i]
    title = df['title'][i]
    #if i == 0:
      #print(title)

    # remove Reference Section
    text = text.split('REFERENCES')[0]

    #if i == 0:
     #print("After remove references")
      #print(text)

    # remove acknowleooements Section
    text = text.split('ACKNOWLEOOEMENTS')[0]

    #if i == 0:
      #print("After remove Acknowledgements")
      #print(text)

    # convert to lower case
    text = text.lower()
    title = title.lower()

    #if i == 0:
      #print("After convert to lower")
      #print(text)

    #remove new lines
    text = text.replace('\r', ' ').replace('\n', ' ')

    # remove special characters and digits (execpt dot) (^.)
    text = re.sub(r"[^\w'+.+']", " ", text)
    text = re.sub(r'[0-9]', " ", text)
    title = re.sub("[^a-zA-Z]+", " ", title)

    if i == 0:
      print("After remove digis and special characters")
      print(text)
      print(title)

    #Remove fig. inc.
    text = text.replace('fig.', '.')
    text = text.replace('inc.', '.')

    #Replace multiple dots with one dot
    text = re.sub(r'\.+', ".", text)

    #Remove spaces before dot at the end of sentences
    text = re.sub(r'\s+(["."])', r'\1', text)

    #if i == 0:
      #print("After remove spaces before dot")
      #print(text)
      #print(title)

    #replace multiple dots with one dot
    text = re.sub(r'\.+', ".", text)

    ##Convert to list from string
    text = text.split()
    title = title.split()
    text = [ch for ch in text if ch.isalnum() or (ch[:-1].isalnum() and ch[-1] == ".")]
    title = [ch for ch in title if ch.isalnum()]
    title[-1] += '.'

    #if i == 0:
      #print("After tekenization")
      #print(text)
      #print(title)

    #Remove all incorrect words
    textb = []
    for word in text:
      if word[-1] == '.':
        if stem(word[:-1]) in word_set or word[:-1] in word_set:
          textb.append(word)
      else:
        if stem(word) in word_set or word in word_set:
          textb.append(word)
    text = textb

    if i == 0:
      print("After remove incorrect words")
      print(text)


    #remove the text between Title and Abstract
    try:
      text = text[text.index('abstract'):]
    except:
      try:
        text = text[text.index('introduction'):]
      except:
        text = text

    if i == 0:
      print('After delete before Abstract')
      print(text)
      print(title)


    # remove stopwords
    text = [word for word in text if word not in stop_words]
    title = [word for word in title if word not in stop_words]

    if i == 0:
      print("After remove stopwords")
      print(text)
      print(title)

    # remove words less than three letters
    k = 0
    textb = []
    for word in text:
      if len(word) > 4:
        textb.append(word)
        k+=1
      elif len(word) == 4:
        if word[-1] != '.':
          textb.append(word)
          k+=1
        else:
          if k != 0:
            textb[k-1] = textb[k-1] + '.'
      else:
        if word[-1] == '.' and k != 0:
          textb[k-1] = textb[k-1] + '.'
    text = textb

    title = [word for word in title if len(word)>=4]

    if i == 0:
      print("After remove words less than 4")
      print(text)

    #Add the title to list of titles
    list_of_titles.append(' '.join(title))

    #Add the title to the text
    if title[-1][-1] != '.':
      title[-1]+='.'
    text = title + text

    # lemmatize
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]

    if i == 0:
      print("After lemmetization")
      print(text)

    text = " ".join(text)
    text = re.sub(r'\.+', ".", text)

    corpus.append(text)

    # save copy of paper_text to use in topic creation step
    textt = text.replace('.', '')
    #remove new lines
    textt = textt.replace('\r', ' ').replace('\n', ' ')
    list_of_paper_text.append(textt.split(' '))

df1 = pd.DataFrame(corpus)
df1.to_csv('Datasets/corpus.csv', index=False)

df2 = pd.DataFrame(list_of_paper_text)
df2.to_csv('Datasets/list_of_paper_text.csv', index=False)

df3 = pd.DataFrame(list_of_titles)
df3.to_csv('Datasets/list_of_titles.csv', index=False)

########################################################################
# split text into sentence
########################################################################

Sentence_list = []

for i in range(0,len(corpus)):
    a_list = nltk.tokenize.sent_tokenize(corpus[i])  # corpus[i].split('.')
    for j in range(0, len(a_list)):
        Sentence_list.append(a_list[j])

# print the all sentences
print("######## Sentence List #########")
print(len(Sentence_list))
#print(Sentence_list)


########################################################################
# sSentence_list reduction Functions
########################################################################

# function of Sentence Reduction (length)
def SentenceReduction_lenght(list_of_sentence):
    new_list_of_sentence = []
    for i in range(0,len(list_of_sentence)):
        sen = list_of_sentence[i]

        # check lenght
        if len(sen.split(' '))> 4 or sen in list_of_titles:
          new_list_of_sentence.append(sen)

    return new_list_of_sentence

### function to compute term frequency
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

### function to compute TF-IDF

def computeTF_idf(wordDict, Nd):
    N = len(Nd)
    tfidf_Dict = {}
    for word, tf in wordDict.items():
      dfi = 0
      for j in range(len(Nd)):
        if word in Nd[j]:
          dfi+=1
      tfidf_Dict[word] = tf * math.log(N/dfi)
    return tfidf_Dict

### function of Sentence Reduction (Scores)

def SentenceReduction_TF(list_of_sentence):
    # compute TF for all sentence
    list_of_words = []
    for i in range(0, len(list_of_sentence)):
        words_in_sentence = list_of_sentence[i].split(' ')
        for j in range(0,len(words_in_sentence)):
          if words_in_sentence[j][-1] != '.':
            list_of_words.append(words_in_sentence[j])
          else:
            list_of_words.append(words_in_sentence[j][:-1])

    # apply TF for the corpus
    #numOfWordsA = dict.fromkeys(list_of_words, 0)

    numOfWordsA = {}
    for word in list_of_words:
      if word in numOfWordsA:
        numOfWordsA[word] += 1
      else:
        numOfWordsA[word] = 1
    #print("numOfWordsA:")
    #print(numOfWordsA)

    tfA = computeTF(numOfWordsA, list_of_words)
    #print(tfA)
    tfd = computeTF_idf(tfA, list_of_paper_text)
    #print(tfd)

    # compute score for each sentence
    #new_dic_of_sentence = dict.fromkeys(list_of_sentence, 0)
    new_dic_of_sentence = {}

    for i in range(0, len(list_of_sentence)):
      sen = list_of_sentence[i]
      sum = 0
      words = sen.split(' ')
      sn = len(words)   #The length of each sentence
      for k in range(0, sn):
        if words[k][-1] == '.':
          words[k] = words[k][:-1]

      for k in range(0, sn):
        w = words[k]
        sum = sum + tfd.get(w,0)  # return 0 if not exist

        if sen in list_of_titles:
          new_dic_of_sentence[sen] = (sum + 0.1) / sn    ##add importance to titles
        else:
          new_dic_of_sentence[sen] = sum / sn
    #print(new_dic_of_sentence)

    # sort the dic by score (asc..)
    res = sorted(new_dic_of_sentence.items(), key=lambda x:x[1])
    # convert list of tuple to dic.. becouse the above sort convert dict.. to list
    res = dict(res)
    #print("after sorting")
    #print(res)

    # convert dic (sentence, score) to list[sentence] only
    f_res = list(res.keys())
    #print(len(f_res))
    #print(f_res)

    # select 75% sentences only (delete sentence 25% has less score)
    f_res = f_res[math.ceil(len(f_res)/4):len(f_res)]
    #print(len(f_res))
    #print(f_res)

    return f_res

########################################################################
# Sentence_list reduction step
########################################################################
print("#### number of sentence before applying sentence reduction #######")
print(len(Sentence_list))
#print(Sentence_list)

# for lenght
Sentence_list = SentenceReduction_lenght(Sentence_list)
print("#### number of sentence after apply lenght of sentence reduction #######")
print(len(Sentence_list))
#print(Sentence_list)


# for score
Sentence_list = SentenceReduction_TF(Sentence_list)
print("#### number of sentence after apply score of sentence reduction #######")
print(len(Sentence_list))
#print(Sentence_list)


##### Save Sentence_list for further processing #####

df4 = pd.DataFrame(Sentence_list)
df4.to_csv('Datasets/Sentence_list.csv', index=False)