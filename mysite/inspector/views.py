#libraries for machine learning aspect
import requests
from bs4 import BeautifulSoup

import pandas as pd
import math
import nltk
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64


from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from collections import Iterable
from IPython.display import HTML

wordnet_lemmatizer = WordNetLemmatizer()

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt


from django.http import request,HttpResponse
from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout

from django.contrib.auth.decorators import login_required 

from .forms import CreateUserForm
from .models import Info


def home(request):
    return render(request,'basic/home.html')

@login_required(login_url='login')
def index(request):
    return render(request,'basic/index.html')

@login_required(login_url='login')
def results(request):
    return render(request,'basic/results.html')

# Create your views here.
def registrationPage(request):
    if request.user.is_authenticated:
        return redirect("index")
    else:
        form = CreateUserForm()

        if request.method == "POST":
            form = UserCreationForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get("username")
                messages.success(request, "Account was created for " + user)
                return redirect('login')


        context = {'form':form}
        return render(request,'accounts/register.html',context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect("index")
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request,username = username,password=password)
            
            if user is not None:
                login(request,user)
                return redirect('index')
            else:
                messages.info(request,'Username OR Password is incorrect ')

        context = {}
        return render(request,'accounts/login.html',context)
    
def logoutUser(request):
    logout(request) 
    return redirect ('login')


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item



def topic_modeler(reviews, Ratios):
  #This function creates analyzes each sentence in the reviews, and assigns a topic distribution to them.
  #The sentiment of the sentence will also be analyzed using Flair package
  from flair.models import TextClassifier
  from flair.data import Sentence
  import re
  
  #For checking Flair datatype
  def isfloat(value):
    try:
      float(value)
      return True
    except ValueError:
      return False
  sia = TextClassifier.load('en-sentiment')
  
  #Topic Categorizer to be applied on each review in the dataframe.
  #Return dataframe with each row containing a sentence, sentence topic distribution, and sentence sentiment score
  def topic_categorizer(paragraph):
      #Split paragraph into sentences
      sentence_list = nltk.tokenize.sent_tokenize(paragraph)
      
      #Create Weight list for each topic - we have ten topics
      Battery_List = []
      Sound_List = []
      Camera_List =[]
      Storage_List = []
      Price_List = []
      Feel_List = []
      Screen_List = []
      Software_List = []
      Service_List = []
      Internet_List = []

      #Go through each sentence
      sentence_analyzed = []
      sentence_topics = []
      sentence_sentiment = []
      for sentence in sentence_list:
          
          #Create dictionary of all counts of keywords in this sentence
          count_dict = dict()
          for keyword in list(Ratios.index):
              counts = sentence.count(keyword)
              if counts > 0:
                  count_dict[keyword] = counts
          
          #Append this count information to your Term-Ratios Dataframe
          #Calculate the Weighted Ratio of each term
            
          topic_list = []
          if count_dict:
              #Pull the TF-IDF Ratios from the Ratios table. This will weigh each matching word accordingly
              sent_df = Ratios.join(pd.DataFrame.from_dict(count_dict, orient = 'index'), how = 'inner')
              sent_df = sent_df.rename(columns = {0: 'Counts'})
              Ratio_sum = ((1/sent_df['Ratios'])*sent_df['Counts']).sum()
              sent_df['Weighted_Ratio'] = (1/(sent_df['Ratios'])*sent_df['Counts'])/Ratio_sum
              
              #Aggregate the terms into the topics specified
              sent_gb = sent_df.groupby(['Bat','Sou','Cam','Sto','Pr','Fel','Scr','Sof','Serv','Int'],as_index = False)['Weighted_Ratio'].sum()
              
              #Append these to the original lists
              if sent_gb['Bat'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Bat'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Sou'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Sou'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Cam'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Cam'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Sto'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Sto'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Pr'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Pr'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Fel'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Fel'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Scr'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Scr'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Sof'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Sof'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Serv'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Serv'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)
              if sent_gb['Int'].any() == True:
                  topic_list.append(sent_gb[sent_gb['Int'] == True].reset_index().iloc[0]['Weighted_Ratio'])
              else:
                  topic_list.append(0)

              #Calculate the Sentiment of the Sentence
              sent = Sentence(sentence)
              sia.predict(sent)
              score = str(sent.labels[0])
              score = score.replace('(',' ').replace(')',' ')
              number = [float(s) for s in score.split() if isfloat(s) is True]
              
              if "POSITIVE" in score:
                  flair_score =  number[0]
              elif "NEGATIVE" in score:
                  flair_score = -number[0]

          #Append all processed data to the lists.
          if topic_list:
              sentence_topics.append(topic_list)
              sentence_analyzed.append(sentence)
              sentence_sentiment.append(flair_score)
              
      return sentence_analyzed, sentence_topics, sentence_sentiment
  
  review_topic = pd.DataFrame()
  reviews['Sentence'] , reviews['Sentence Topics'], reviews['Sentence Sentiment'] = zip(*reviews.new_reviewText.apply(topic_categorizer))
  
  return reviews

def Convert(string):
    li = list(string.split(" "))
    return li

def search(request):
    if request.GET:
        jumia_product_url = request.GET['search_url']

    # Define te search url
    response = requests.get(jumia_product_url)
    # Check successful response
    if response.status_code != 200:
        raise Exception('Failed to load page {}'.format(jumia_product_url))
    # Parse using BeautifulSoup
    product_doc = BeautifulSoup(response.text, 'html.parser')

    tags = product_doc.find_all('a',{'class':'cbs'})
    t = []
    for tag in tags:
        t.append(tag.text)
    oops = 'Jumia Smartphone URL Error'
    exist = 'Smartphones' not in t

    if exist:
        context = {
            'exist' : exist,
            'oops' : oops,
        }
        return render(request,'basic/results.html',context)

    else:
        # Phone Name, Brand, Image and Price
        title = product_doc.find('title')
        phone_list = Convert(title.text)
        phone_name = phone_list[0] + " " + phone_list[1] + " " + phone_list[2]
        brand = phone_list[0]

        price_tag = product_doc.find('span',{'class':'-b'})
        p = Convert(price_tag.text)
        val_price= int(p[1].replace(',', ''))

        image_tag = product_doc.find('img',{'class':'-fsh0'})
        image =image_tag.get('data-src','')
        

        review_page = product_doc.find('a',{'class':'-plxs'})
        base_url = 'https://www.jumia.co.ke'
        review_url = base_url + review_page.get('href','')

        response = requests.get(review_url)
        # Check successful response
        if response.status_code != 200:
            raise Exception('Failed to load page {}'.format(review_url))
        # Parse using BeautifulSoup
        review_doc = BeautifulSoup(response.text, 'html.parser')


        # Find the count of reviews of the given product

        total_counts = review_doc.find_all('h2',{'class':'-fs14'})
        review_count = total_counts[1].text
        lst = review_count.split()
        refined = lst[2].replace("(","")
        f_count = refined.replace(")","")
        final_count = int(f_count)
        total_review_pages = math.ceil(final_count / 10)

        # Find URL of each of the pages
        current_page = 1
        review_url =  []
        while current_page <= total_review_pages :
            review_url.append(base_url + review_page.get('href','') + '?page=' + str(current_page))
            current_page += 1
            

        descs = []
        for url in review_url:
            response = requests.get(url)
            review_doc = BeautifulSoup(response.text, 'html.parser')
            desc_tags = review_doc.find_all('p',{'class':'-pvs'})
            for tag in desc_tags:
                descs.append(tag.text)

        num_of_reviews = len(descs)
        if num_of_reviews == 0:
            context = {
                'num_of_reviews':num_of_reviews,
            }
            return render(request,'basic/results.html',context)

        else:
            
            jumia_df = pd.DataFrame(descs,columns =['reviewText'])

            # Preprocessing

            # case text as lowercase, remove punctuation, remove extra whitespace in string and on both sides of string
            jumia_df['remove_lower_punct'] = jumia_df['reviewText'].str.lower().str.replace("'", '').str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

            # apply sentiment analysis
            analyser = SentimentIntensityAnalyzer()

            sentiment_score_list = []
            sentiment_label_list = []

            for i in jumia_df['remove_lower_punct'].values.tolist():
                sentiment_score = analyser.polarity_scores(i)

                if sentiment_score['compound'] >= 0.05:
                    sentiment_score_list.append(sentiment_score['compound'])
                    sentiment_label_list.append('Positive')
                elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
                    sentiment_score_list.append(sentiment_score['compound'])
                    sentiment_label_list.append('Neutral')
                elif sentiment_score['compound'] <= -0.05:
                    sentiment_score_list.append(sentiment_score['compound'])
                    sentiment_label_list.append('Negative')
                
            jumia_df['sentiment'] = sentiment_label_list
            jumia_df['sentiment score'] = sentiment_score_list

            # tokenise string
            jumia_df['tokenise'] = jumia_df.apply(lambda row: nltk.word_tokenize(row[1]), axis=1)

            # initiate stopwords from nltk
            stop_words = stopwords.words('english')

            # remove stopwords
            jumia_df['remove_stopwords'] = jumia_df['tokenise'].apply(lambda x: [item for item in x if item not in stop_words])

            # lemmatise words
            jumia_df['lemmatise'] = jumia_df['remove_stopwords'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x]) 

            jumia_df['new_reviewText'] = jumia_df['lemmatise'].map(lambda x:' '.join(x))

            #Define bag of words
            battery_keyword = ['battery','life','charge','lose','long','full','run','drain']

            sound_keyword = ['speaker','volume','quality','music','headphone','sound','talk','hear']

            camera_keyword = ['camera','video','call','photo','display','visual','lens','pixels']

            storage_keyword = ['store','ram','space','memory', 'gb', 'mb','storage','files']

            price_keyword = ['price','money','worth','cheap', 'expensive','value','affordable','cost']

            feel_keyword = ['fit','large','pocket','sleek','small','comfortable','light','case']

            screen_keyword = ['screen','protector','break','glass','black','drop','scratch','clear']

            software_keyword = ['update','install','performance','software','security','warranty','app','feature']

            service_keyword = ['receive','recommend','excellent','deliver','review','replacement','refund','forever']

            internet_keyword = ['light','network','slow','fast','browser','speed','browsing','internet']

            term_dictionary = dict()
            for keyword in (battery_keyword + sound_keyword + camera_keyword +storage_keyword+ price_keyword + feel_keyword + screen_keyword + software_keyword + service_keyword ):
                term_dictionary[keyword] = sum(jumia_df.reviewText.str.count(keyword))

            total_term_count = sum(term_dictionary.values())

            ratio_dictionary = dict()
            for keyword in (battery_keyword + sound_keyword + camera_keyword +storage_keyword+ price_keyword + feel_keyword + screen_keyword + software_keyword + service_keyword):
                ratio_dictionary[keyword] = term_dictionary[keyword]/total_term_count

            # jumia_sample = jumia_df[1:100]

            sentence = 'The atmosphere and service are great, but the waiter is a bit slow.'
            Ratios = pd.DataFrame.from_dict(ratio_dictionary, orient = 'index')
            Ratios = Ratios.rename(columns = {0: 'Ratios'})
            Bat = Ratios.index.isin(battery_keyword)
            Ratios['Bat'] = Bat
            Sou = Ratios.index.isin(sound_keyword)
            Ratios['Sou'] = Sou
            Cam = Ratios.index.isin(camera_keyword)
            Ratios['Cam'] = Cam
            Sto = Ratios.index.isin(storage_keyword)
            Ratios['Sto'] = Sto
            Pr = Ratios.index.isin(price_keyword)
            Ratios['Pr'] = Pr
            Fel = Ratios.index.isin(feel_keyword)
            Ratios['Fel'] = Fel
            Scr = Ratios.index.isin(screen_keyword)
            Ratios['Scr'] = Scr
            Sof = Ratios.index.isin(software_keyword)
            Ratios['Sof'] = Sof
            Serv = Ratios.index.isin(service_keyword)
            Ratios['Serv'] = Serv
            Int = Ratios.index.isin(internet_keyword)
            Ratios['Int'] = Int

            jumia_sample = jumia_df[0:math.floor(len(jumia_df))]
            reviews_list = np.array_split(jumia_sample,math.ceil(len(jumia_df)/25))
            flair_list = []

            for i in range (0, len(reviews_list)):
                flair_review = topic_modeler(reviews_list[i], Ratios)
                flair_list.append(flair_review)
                

            reviews_filtered = pd.concat(flair_list)
            reviews_filtered = reviews_filtered.drop("Sentence Sentiment", axis=1)

            df_trial = pd.DataFrame()
            df_trial['Topics'] = reviews_filtered['Sentence Topics']
            # df_trial.head(10)


                        
            flattened = []
            count = 0
            for lists in df_trial.Topics:
                lists = list(flatten(lists))
                flattened.append(lists)

            for i in df_trial.index:
                df_trial.at[i, "Topics"] = flattened[count]
                count = count + 1

            maxi = []
            count = 0
            for topic in df_trial.Topics:
                if len(topic) > 0:
                    i = topic.index(max(topic))
                else:
                    i = -1
                maxi.append(i)
            
            topics = ['Battery','Sound','Camera','Storage','Price','Feel','Screen','Software','Service','Internet']

            for i in df_trial.index:
                df_trial.at[i, "Topics"] = maxi[count]
                count = count + 1
            df_trial.Topics = df_trial.Topics.apply(lambda y: 'Other' if y==-1 else topics[y])
            # df_trial.Topics = df_trial.Topics.apply(lambda y: np.nan if len(y)==0 else y)
            # df_trial

            reviews_filtered['DominantTopic'] = df_trial['Topics'] 

            analysis_df = pd.DataFrame()
            analysis_df = reviews_filtered.drop(["remove_lower_punct", "tokenise", "remove_stopwords", "lemmatise","new_reviewText","Sentence Topics","Sentence"], axis=1)
            analysis_df = analysis_df.set_index("DominantTopic")
            analysis_df_1 = analysis_df.drop("sentiment score",axis = 1)

            # topics = analysis_df["DominantTopic"].tolist()

            results = analysis_df.groupby(['DominantTopic', 'sentiment']).count().reset_index()

            graph_results = results[['DominantTopic', 'sentiment', 'sentiment score']]
            graph_results = graph_results.pivot(index='DominantTopic', columns='sentiment', values='sentiment score').reset_index()

            graph_results.set_index('DominantTopic', inplace=True)

            fig, ax = plt.subplots()

            fig = plt.figure(figsize=(8,4))
            ax  = graph_results.plot.bar(rot=90)
            
            ax.set_ylabel('Number of Reviews')
            ax.set_title('Number of sentiment per topic')

            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')

            # canvas = FigureCanvas(fig)
            # response = HttpResponse( content_type = 'image/png')
            # canvas.print_png(response)
            info = Info(brand_name = brand, price = val_price)
            info.save()
        
            json_records =analysis_df_1.reset_index().to_json(orient ='records')
            data = []
            data = json.loads(json_records)

            context ={
                'num_of_reviews' : num_of_reviews,
                'phone_name' : phone_name,
                'brand' : brand,
                'price' : val_price,
                'image' : graphic,
                'picture' : image,
                'd' : data,
            }

            return render(request,'basic/results.html',context)

