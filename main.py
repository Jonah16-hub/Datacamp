import lib as lb
import methods as mt


 
def main():
    lb.st.title("PFIZER DIGITAL FOOTPRINT EVALUATION")
    lb.st.info("Welcome")
    lb.st.write("The purpose of the project is to measure the digital impact of PFIZER on internet for that we gather informations from differents sources")
    lb.st.sidebar.title('*SELECT THE SOURCES*')
    lb.st.sidebar.header('Differents places where we evaluate the digital impact of the company Pfizer')
    
    #select box
    page = lb.st.sidebar.selectbox(
                          "Select data to visualize",
                          [
                            "On news paper (from Google News)",
                            "Twitter (User's tweets)",
                            "About",
                          ],)

    #Readme page
    if page == "About":
        with lb.st.container():
            lb.st.balloons()
            lb.st.markdown(mt.readme())

    #twitter data
    elif page == "Twitter (User's tweets)":
        
        data_url = 'tt.json'
        tt = mt.loader(data_url)
        lb.st.write("Let's take a look to our data scrapped data from twitter")
        tt = tt[['date','content','hashtags','lang']]
        tt.date =  lb.pd.to_datetime(tt['date'], format='%d%b%Y:%H:%M:%S.%f')
        with lb.st.container():
            lb.st.subheader("EDA")
            lb.st.write(tt)
            lb.st.write("we took ",tt.shape[0]," random tweets from Sunday ",tt.date[999], " to ",tt.date[0] ,", in differents languages . We kept the users anonym for confidentiality.")
            lb.st.write("the languages are distributed as follows :")
            mt.display_countplot(tt.lang , "Tweet's language repartition")
            hastags = []
            for hash in tt.hashtags:
                for tags in hash :
                    hastags.append(tags)

            lb.st.write("50 Most Common Hastags in our tweets")

            fig , ax = lb.plt.subplots(figsize = (8, 8), facecolor = None)
            wordcloud = lb.WordCloud(width = 800, height = 800,
            background_color ='white',
            min_font_size = 10).generate(str(hastags))

            # plot the WordCloud image                      
            lb.plt.imshow(wordcloud)
            lb.plt.axis("off")
            lb.plt.tight_layout(pad = 0)
            lb.st.pyplot(fig)


        lb.st.write("Now we will start the text processing phase")
        
        with lb.st.container():
            lb.st.subheader("Tokenization(How can transform tweets into words or text format?)")
            lb.st.write("Now we gonna start Transfering strings into a single textual token And then we gonna apply textual filter depending on the most frequent languages to remove stopwords and punctuation to start our evaluation")
            
            words_dico = dict()
            expr = lb.re.compile("\W+",lb.re.U)
            for text in tt.content: 
                text = str(text)
                text = expr.split(text)
                for word in set(text): 
                    if word not in words_dico:
                        words_dico[word]=1
                    else: 
                        words_dico[word]=words_dico[word]+1
            
            freq_words = lb.FreqDist(words_dico).most_common(50)
            lb.st.write("The 50 most common words in our raw tweets dataset")
            freq_words = lb.pd.Series(dict(freq_words))
            mt.display_barplot(freq_words.values , freq_words.index ,"50 Most common words")

            words_freq = list()
            for key, val in words_dico.items():
                words_freq.append((key, val))
            words_freq.sort(key=lambda tup: tup[1] ,reverse=True)

            lb.st.write("Let's scrub our tweets. We have to remove punctuation and stopwords from the most common languages above")
            filtre2 = 50
            words_to_delete2=[t[0] for t in words_freq[:filtre2]]
            
            filtre2 = 25
            words_to_delete2=[t[0] for t in words_freq[:filtre2]]

            stop_words_en = set(lb.stopwords.words('english'))
            stop_words_fr = set(lb.stopwords.words('french'))
            stop_words_it = set(lb.stopwords.words('italian'))
            stop_words_es = set(lb.stopwords.words('spanish'))
            stop_words_gm = set(lb.stopwords.words('german'))
            stop_words_pt = set(lb.stopwords.words('portuguese'))
            stop_words_ar = set(lb.stopwords.words('arabic'))
            lb.string.punctuation = lb.string.punctuation +'"'+'"'+'-'+'''+'''+'—'
            removal_list = stop_words_ar | stop_words_pt | stop_words_en | stop_words_fr | stop_words_it | stop_words_es | stop_words_gm
            
            stopwords = set(removal_list) | set(words_to_delete2)
            with lb.st.expander("See he list of all removed caracters"):
                lb.st.write(removal_list)
                lb.st.write("The other words to delete : ",words_to_delete2)

            lb.st.write("We can check the most common reccurrents words in the tweets related to PFIZER ")
            mt.display_wordcloud(tt.content , stopwords , "Most Recurrent words on tweets related to PFIZER")

            
        with lb.st.container():
            lb.st.subheader("Normalization and noise Reduction (How can I cope with too many variables in my Document‐Term‐Matrix?)")
            lb.st.write("We are going to reduce the dimensionality of document term matrix. We have already deleted stopwords and infrequent words.")
            lb.st.write("Now we will Lemmatize our tweets and another option was to stemmitize.")
            lb.st.info("Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.")
            lb.st.info("Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language. In Lemmatization root word is called Lemma. A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.")
            
            lb.st.write("Our first step is to remove numerical caracter on tweets dataset with that simple python function :")
            code1 = """
            def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text
            """
            lb.st.code(code1 , language="python")
            l1 = ["content"]
            l2 = []
            for i in tt.columns:
                if i != l1[0]:
                    l2.append(i)
                    
            myData = tt.drop(l2, axis=1)
            myData['Cleaned_comment'] = myData['content'].apply(mt.clean)
            lb.st.write("Let's compare the old dataset and the new one")
            lb.st.write(myData.head(3))

        with lb.st.container():
            lb.st.subheader("Linguistic analysis (How can I identify words with a special meaning or grammatical function?)")
            lb.st.write("Now we are going to do Part of Speech Tagging wich is an important step in our text mining process. As recall , ")
            lb.st.text("""
            POS Tagging (Parts of Speech Tagging) is a process to mark up the words in text format for a particular part of a speech based on its definition and context.
            \nIt is responsible for text reading in a language and assigning some specific token (Parts of Speech) to each word.
            \nIt is also called grammatical tagging.
            \nLet’s learn with a NLTK Part of Speech example:
            \nInput: Everything to permit us.
            \nOutput: [(‘Everything’, NN),(‘to’, TO), (‘permit’, VB), (‘us’, PRP)] \n\n\n \t """)

            lb.st.write("We are also going to use WordNetLemmatizer librarie from NLTK.")
            myData['POS tagged'] = myData['Cleaned_comment'].apply(mt.token_stop_pos)
            myData['Lemma'] = myData['POS tagged'].apply(mt.lemmatize)

            lb.st.write("That's our look our tweets data now (Lemmatize tweets , POS tagged tweets and raw tweets")
            lb.st.write(myData[['content','Lemma' , 'POS tagged']].head())

        with lb.st.container():
            lb.st.subheader("Sentiment Analysis")
            lb.st.image("assets/sa.png",use_column_width="auto")
            lb.st.write("Sentiment analysis is the process of classifying whether a block of text is positive, negative, or, neutral. Sentiment analysis is contextual mining of words which indicates the social sentiment of a brand and also helps the business to determine whether the product which they are manufacturing is going to make a demand in the market or not. The goal which Sentiment analysis tries to gain is to analyze people’s opinion in a way that it can help the businesses expand. It focuses not only on polarity (positive, negative & neutral) but also on emotions (happy, sad, angry, etc.).")
            lb.st.write("We will use Textblob and calculate the subjectivity , polarity and the fine grain in the analysis of tweets")
            lb.st.info("TextBlob returns polarity and subjectivity of a sentence. Polarity lies between [-1,1], -1 defines a negative sentiment and 1 defines a positive sentiment. Negation words reverse the polarity. TextBlob has semantic labels that help with fine-grained analysis. For example — emoticons, exclamation mark, emojis, etc. Subjectivity lies between [0,1]. Subjectivity quantifies the amount of personal opinion and factual information contained in the text. The higher subjectivity means that the text contains personal opinion rather than factual information. TextBlob has one more parameter — intensity. TextBlob calculates subjectivity by looking at the ‘intensity’. Intensity determines if a word modifies the next word. For English, adverbs are used as modifiers (‘very good’). For example: We calculated polarity and subjectivity for “I do not like this example at all, it is too boring”. For this particular example, polarity = -1 and subjectivity is 1, which is fair.However, for the sentence “This was a helpful example but I would prefer another one”. It returns 0.0 for both subjectivity and polarity which is not the finest answer we’d expect.")
            pData = lb.pd.DataFrame(myData[['content', 'Lemma']])
            pData['Subjectivity'] = pData['Lemma'].apply(mt.getSubjectivity) 
            pData['Polarity'] = pData['Lemma'].apply(mt.getPolarity) 
            pData['Analysis'] = pData['Polarity'].apply(mt.analysis)
            #InDepth or fine-grained
            pData['InD_Analysis'] = pData['Polarity'].apply(mt.analysis2)

            lb.st.write("Check the result")
            lb.st.write(pData[['content','Subjectivity','Polarity','Analysis','InD_Analysis']].head(5))

            lb.st.write("From this we can deduce the different feelings in the tweets")
            
            tab1,tab2,tab3 = lb.st.tabs(["Tweet's reviews analysis in 3 kind","Tweet's reviews polarity analysis", "hide"])
            with tab1:
                mt.display_countplot(pData.Analysis , "Sentiment density")
            with tab2:
                mt.display_countplot(pData.InD_Analysis, "More detailed sentiment density")
            with tab3:
                pass
            
            lb.st.write("Another way to evaluate the sentiments is to use VADER")
            lb.st.write("VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative. VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.")
            

            pData['Vader Sentiment'] = pData['Lemma'].apply(mt.vadersentimentanalysis)
            pData['Vader Analysis'] = pData['Vader Sentiment'].apply(mt.vader_analysis)

            lb.st.write('The vader analysis give us the following repartition :')
            mt.display_countplot(pData["Vader Analysis"] ,"Tweet's reviews vader analysis")

            lb.st.write("A comparision of sentiment analysis values that we get with the 2 libraries")
            tab1,tab2,tab3 = lb.st.tabs(["TextBlob","Vader", "hide"])
            with tab1:
                c_counts = pData.Analysis.value_counts()
                fig ,ax = lb.plt.subplots(figsize=(6, 3))
                lb.plt.pie(c_counts.values, labels = c_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
                lb.plt.title("Tweet's tewtblob results")
                lb.st.pyplot(fig) 
            with tab2:
                vader_counts = pData["Vader Analysis"].value_counts()
                fig ,ax = lb.plt.subplots(figsize=(6, 3))
                lb.plt.pie(vader_counts.values, labels = vader_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
                lb.plt.title("Tweet's tewtblob results")
                lb.st.pyplot(fig)

            lPos = []
            lNeg = []
            lNeu = []
            for i in range(len(tt.content)):
                if i != 98 and i != 247:
                    if pData["Vader Analysis"][i] == "Positive":
                        lPos.append(tt.content[i])
                    elif pData["Vader Analysis"][i] == "Negative":
                        lNeg.append(tt.content[i])
                    else:
                        lNeu.append(tt.content[i])
            lPos = lb.pd.DataFrame({'Pos_Comments':lPos})
            lNeg = lb.pd.DataFrame({'Neg_Comments':lNeg})
            lNeu = lb.pd.DataFrame({'Neu_Comments':lNeu})
            lb.st.write("We finally sorted the tweets in 3 sorts")
            
            tab1 , tab2 , tab3 , tab4 = lb.st.tabs(["Positive","Neutral","Negative","Hide"])
            with tab1:
                lb.st.write("5 Positive tweets")
                lb.st.write(lPos.head(5))
                mt.display_wordcloud(lPos.Pos_Comments , stopwords , "Positive words related to PFIZER")
            with tab2:
                lb.st.write("5 Neutral tweets")
                lb.st.write(lNeu.head(5))
                mt.display_wordcloud(lNeu.Neu_Comments , stopwords , "Neutral words related to PFIZER")
            with tab3:
                lb.st.write("5 Negative tweets")
                lb.st.write(lNeg.head(5))
                mt.display_wordcloud(lNeg.Neg_Comments , stopwords , "Negative words related to PFIZER")
           
            
        with lb.st.container():
            lb.st.subheader("KMEANS CLUSTERING")
            vectorizer = lb.TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(pData.content)

            true_k = 3
            model = lb.KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1)
            model.fit(X)

            cluster = {}
            top = []
            lb.st.write("Top words per cluster:")
            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()
            for i in range(true_k):
                lb.st.write("Cluster %d:" % i)
                for ind in order_centroids[i, :10]:
                    # lb.st.write(' %s ' % terms[ind])
                    top.insert(ind,terms[ind])
                cluster[i] = top[-10:]

        

    #Google news data
    elif page == "On news paper (from Google News)":
        data_url = "data/gn.json"
        lb.st.write("")



main()