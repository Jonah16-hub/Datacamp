from nltk.classify.util import nltk
import lib as lb

#loading data
@lb.st.cache
def load_data(data_url):
    data = lb.pd.read_json('data/'+data_url)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data
@lb.st.cache
def readme():
    with open('assets/readme.md', 'r') as f:
        output = f.read()
    return output

def display_countplot(col,title):
    fig ,ax = lb.plt.subplots(figsize=(6, 3))
    col.value_counts().plot.bar()
    lb.plt.title(title)
    lb.st.pyplot(fig)

def get_hour(dt):
    return dt.hour

def weekday(dt):
    return dt.weekday

def display_barplot(X ,Y , title):
    fig , ax = lb.plt.subplots(figsize=(6, 3))
    # lb.plt.bar(X , Y)
    lb.sns.barplot(x=X, y=Y, orient='h', ax=ax)
    lb.plt.title(title)
    # lb.plt.xticks(    );
    lb.st.pyplot(fig)

# a function which display a progress bar while loading the data
def loader(data_url):
    # upgrading the state automaticaly
    data_load_state = lb.st.empty()
    bar = lb.st.progress(0)
    for i in range(100):
        data_load_state.text(f'Loading data... {i+1}➗')
        bar.progress(i + 1)
        lb.time.sleep(0.1)
    df = load_data(data_url)
    data_load_state.text('Loading Over 💯➗!!')
    return df

def custom_split(sep_list ,  to_split):
    #we create dynamical regular expression
    regular_exp = '|'.join(map(lb.re.escape, sep_list))
    return lb.re.split(regular_exp, to_split)


def display_wordcloud(content, sw , title):
    textt = " ".join(review for review in content)
    wordcloud = lb.WordCloud(stopwords=sw).generate(textt)
    fig , ax = lb.plt.subplots()
    lb.plt.imshow(wordcloud, interpolation='bilinear')
    lb.plt.axis("off")
    lb.plt.title(title)
    lb.st.pyplot(fig)

# Removes all special characters and numericals leaving the alphabets
def clean(text):
    text = lb.re.sub('[^A-Za-z]+', ' ', text)
    return text


# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# from nltk import pos_tag
# # nltk.download()

# POS tagger dictionary
pos_dict = {'J':lb.wordnet.ADJ, 'V': lb.wordnet.VERB,
            'N':lb.wordnet.NOUN, 'R':lb.wordnet.ADV}
def token_stop_pos(text):
    tags = lb.nltk.pos_tag(lb.word_tokenize(text))
    newlist = []

    stop_words_en = set(lb.stopwords.words('english'))
    stop_words_fr = set(lb.stopwords.words('french'))
    stop_words_it = set(lb.stopwords.words('italian'))
    stop_words_es = set(lb.stopwords.words('spanish'))
    stop_words_gm = set(lb.stopwords.words('german'))
    stop_words_pt = set(lb.stopwords.words('portuguese'))
    stop_words_ar = set(lb.stopwords.words('arabic'))
    lb.string.punctuation = lb.string.punctuation +'"'+'"'+'-'+'''+'''+'—'
    removal_list = stop_words_ar | stop_words_pt | stop_words_en | stop_words_fr | stop_words_it | stop_words_es | stop_words_gm
            
    for word, tag in tags:
        if word.lower() not in set(removal_list):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


wordnet_lemmatizer = lb.WordNetLemmatizer()
@lb.st.cache
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

# function to calculate subjectivity
def getSubjectivity(review):
    return lb.TextBlob(review).sentiment.subjectivity

# function to calculate polarity
def getPolarity(review):
    return lb.TextBlob(review).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    

# function to analyze the reviews #Fine-grain
def analysis2(score):
    if score < -0.7:
        return 'Very Negative'
    elif score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    elif score > 0.7:
        return 'Very Positive'
    elif score > 0:
        return 'Positive'

analyzer = lb.SentimentIntensityAnalyzer()

# function to calculate vader sentiment
def vadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

# function to analyse
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'

