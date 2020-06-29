# # <u>TWITTER SENTIMENT ANALYSIS FOR TWEETS ON COVID-19</u>

# ## Authentication with twitter API

# In[52]:


import tweepy
import sys

non_bmp_map=dict.fromkeys(range(0x10000,sys.maxunicode+1),0xfffd)

consumer_key = 'hT3VsfQ1eWptnVGdtCSEDRr9A'
consumer_secret = 'gYDdL8CcAEHLTgmPAcHuweH8nftVBGtzeIHSD2zenskM7Jgg0r'
access_token = '1272911068705275907-Pw35srDftuKptNSmXQIEXXp5hAOS9u'
access_token_secret = '9Agn6fXwqyoAMTwkwKEDzqiXaI7pG8buMRew8407fN3QC' 
try:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
except:
    print('Authentication Failed')
    exit(0)


# ## Fetching Tweets For Corona Virus

# In[53]:


if api.verify_credentials():
    print("Authentication OK")
    t=tweepy.Cursor(api.search,  
                    q="Corona OR COVID OR pandemic OR coronavirusindia OR lockdown -filter:retweets",
                    lang='en',
                    tweet_type='recent',
                    ).items(2500)
    tmpd=[]
    for i in t:
        tmpd.append([i.geo,i.text.translate(non_bmp_map),i.user.screen_name,i.user.location])
else:
    print('Authentication Failed')
    exit(0)
import pandas as pd
tdet = pd.DataFrame(data=tmpd,columns=['geo','txt','user','location'])


# ## Clean Text Function

# In[3]:


import re


def clean_txt(text):
    text = re.sub("RT @[A-za-z0-9]*:","",text)
    text = re.sub("@[A-za-z0-9]*","",text)
    text = re.sub("\n","",text)
    text = re.sub("#","",text)
    text = re.sub("https?://[a-zA-Z0-9.?=%/]*","",text)
    text = re.sub("/","",text)
    
    return text


# ## Cleaning the text

# In[4]:


tdet['txt']=tdet['txt'].apply(lambda x: clean_txt(x))


# ## Calculating Sentiment 

# In[5]:


import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

dataset = tdet

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
filt_text=[]
for i in range(0,2500):
    review = re.sub('[^a-zA-Z]', ' ', dataset['txt'][i])
    review = review.lower()
    review = review.split()
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in review if not w in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    filt_text.append(filtered_sentence)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =  ' '.join(review)  
    corpus.append(review)


# In[6]:


pol = []
sub = []

for i in range(0,2500):
    obj = corpus[i]
    blob = TextBlob(obj)
    pole = blob.sentiment.polarity
    pol.append(pole)
    subj = blob.sentiment.subjectivity
    sub.append(subj)


# ## Classifying Sentiments 

# In[7]:


pos=0
neg=0
v_neg=0
v_pos=0
nu=0
for i in pol:
    if(i<=-0.5):
        v_neg+=1
    elif(i<0 and i>-0.5):
        neg+=1
    elif(i==0):
        nu+=1
    elif(i>0 and i<0.5):
        pos+=1
    else:
        v_pos+=1
senti=[v_pos,pos,nu,neg,v_neg]


# ## Generating Word Cloud 

# In[8]:


import numpy as np    
from PIL import Image
back = np.array(Image.open('cloud.png'))
text = ' '.join(filt_text)
wordC = WordCloud(background_color='white',max_words=200,mask=back).generate(text)


# # <u>Sentiment Analysis For Lockdown in India</u>

# ## Collenting Tweets 

# In[9]:


try:    
    t=tweepy.Cursor(api.search,  
                        q="lockdown AND extension AND india -filter:retweets",
                        lang='en',
                        tweet_type='recent',
                        ).items(500)
    tmpd=[]
    for i in t:
        tmpd.append([i.geo,i.text.translate(non_bmp_map),i.user.screen_name,i.user.location])
except:
    print('Authentication Failed')
    exit(0)
tdet = pd.DataFrame(data=tmpd,columns=['geo','txt','user','location'])


# ## Cleaning Text 

# In[10]:


tdet['txt']=tdet['txt'].apply(lambda x: clean_txt(x))
dataset = tdet


# ## Calculating Sentiments 

# In[11]:


corpus = []
filt_text=[]
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['txt'][i])
    review = review.lower()
    review = review.split()
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in review if not w in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    filt_text.append(filtered_sentence)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =  ' '.join(review)  
    corpus.append(review)


# In[12]:


pol = []
sub = []
polfile=[]

for i in range(0,len(dataset)):
    obj = corpus[i]
    blob = TextBlob(obj)
    pole = blob.sentiment.polarity
    pol.append(pole)
    polfile.append(pole)
    subj = blob.sentiment.subjectivity
    sub.append(subj)
dataset['senti']=polfile


# ## Classifying Sentiments

# In[13]:


pos=0
neg=0
nu=0
for i in pol:
    if(i<0):
        neg+=1
    elif(i==0):
        nu+=1
    else:
        pos+=1


# In[14]:


locksenti=[pos,nu,neg]


# # <u>Creating the Dash App</u>

# ## Preparing data for Scatter Plot

# In[15]:


scatx=[]
scaty=[]
for i in range(0,len(dataset)):
    scatx.append(dataset['senti'][i])
    scaty.append(dataset['user'][i])


# ## Importing required modules

# In[16]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go


# ## Ploting the pie chart for Corona Virus

# In[17]:


fig=px.pie(values=senti,
                           names=['Very Positive Tweets','Positive Tweets',
                                  'Neutral Tweets','Negative Tweets','Very Negative Tweets'],
                           hole=0.6,
                           color=['Very Positive Tweets','Positive Tweets',
                                  'Neutral Tweets','Negative Tweets','Very Negative Tweets'],
                           color_discrete_map={'Very Positive Tweets':'lightcyan',
                                               'Positive Tweets':'lightblue',
                                               'Neutral Tweets':'darkcyan',
                                               'Negative Tweets':'royalblue',
                                               'Very Negative Tweets':'darkblue'},
                           hover_name=['Very Positive Tweets','Positive Tweets',
                                  'Neutral Tweets','Negative Tweets','Very Negative Tweets'],
            )
fig.update_layout(title={
    'text':"People Reaction to COVID-19 Pandemic",
    'x':0.5,
    'y':0.1,
    'xanchor':'center',
    'yanchor':'bottom',
    'font':dict(size=17,
                color='darkblue')})


# ## Ploting the Word Cloud for Corona Virus

# In[18]:


cloud=px.imshow(wordC)
cloud.update_xaxes(showticklabels=False)
cloud.update_yaxes(showticklabels=False)
cloud.update_traces(hoverinfo='skip',hovertemplate=None)
cloud.update_layout(title={
    'text':"Word Cloud of the Tweets",
    'x':0.5,
    'xanchor':'center',
    'font':dict(size=17,
                color='darkblue')})


# ## Plotting the Scatter Plot for Lockdown in India

# In[19]:


lockscat=go.Figure(data=go.Scatter(x=scatx,y=scaty,mode='markers+lines',
                                  marker=dict(size=10,
                                              color='#ff7777',
                                             colorscale='rainbow'),
                                  line=dict(
                                              color='#aa0000',
                                             )))
lockscat.update_yaxes(showticklabels=False,title='')
lockscat.update_xaxes(title='Polarity of Data')
lockscat.update_layout(title={
    'text':"Frequency Distribution of Tweets",
    'x':0.5,
    'xanchor':'center',
    'font':dict(size=17,
                color='darkred')})


# ## Ploting the pie chart for Lockdown in India

# In[20]:


fig1=px.pie(values=locksenti,
                           names=['In Favour of Lockdown',
                                  'Neutral Tweets','Not in favour of Lockdown'],
                           hole=0.6,
                           color=['In Favour of Lockdown',
                                  'Neutral Tweets','Not in favour of Lockdown'],
                           color_discrete_map={
                                               'In Favour of Lockdown':'red',
                                               'Neutral Tweets':'lightpink',
                                               'Not in favour of Lockdown':'darkred',
                                               },
                           hover_name=['In Favour of Lockdown',
                                  'Neutral Tweets','Not in favour of Lockdown'],
            )
fig1.update_layout(title={
    'text':"People Reaction to Lockdown Extension",
    'x':0.5,
    'y':0.1,
    'xanchor':'center',
    'yanchor':'bottom',
    'font':dict(size=17,
                color='brown')})


# ## Creating the App Layout

# In[49]:


app = dash.Dash(__name__)
app.layout= html.Div(children=[html.Div(children=[
   html.H1(children=html.U('Twitter Sentiment Analysis of Corona Virus Pandemic'),
           style={'color':'#0099aa','background-color':'#ffffff',
                  'text-align':'center', 'position':'relative',
                  'margin-top':'0'}),
   html.Div(style={'height':'15px'}),
   #px.pie(
   html.Div(children=[html.Div(html.H2(html.U('Tweets Analysis of People Reacting to Corona Virus Worldwide')),
                               style={'background-color':'#ffffff','display':'block','text-align':'center'}),
   html.Div(children=[
   dcc.Graph(figure=fig.update_traces(hoverinfo='skip',hovertemplate=['Very Positive Tweets','Positive Tweets',
                                  'Neutral Tweets','Negative Tweets','Very Negative Tweets']),
                                   style={'text-align':'center'})],
                                   style={'display':'inline-block','width':'49.5%'}
),
   html.Div(style={'width':'1%','display':'inline-block'}),
   html.Div(children=[
   dcc.Graph(figure=cloud)],style={'display':'inline-block','width':'49.5%'}),
   
   html.Div(style={'height':'50px'})]),

   
   #html.Div(children=[html.Img(src='1.png'.format(encoded_image),style={'width':'200px','height':'100px'})]),

   html.Div(children=[html.Div(html.H2(html.U('Tweets Analysis of People Reacting to Lockdown Extension in India')),
                               style={'background-color':'#ffffff','display':'block','text-align':'center'}),
   html.Div(children=[
   dcc.Graph(figure=fig1.update_traces(hoverinfo='skip',hovertemplate=['In Favour of Lockdown',
                                  'Neutral Tweets','Not in favour of Lockdown']),
                                   style={'text-align':'center'})],
                                   style={'display':'inline-block','width':'49.5%'}
),
   html.Div(style={'width':'1%','display':'inline-block'}),
   html.Div(children=[
   dcc.Graph(figure=lockscat,style=dict(background='#ffffff'))],style={'display':'inline-block','width':'49.5%'}),
   html.Div(style={'height':'40px'}),
   html.Div(children=[html.H4('Developed By HackUs Team'),
                      html.Br(),
                      'Members- Sayak Ghoshal, Saumya Roy, Sausamya Mitra, Sangjukta Ganguly'
                     ],style={'background-color':'#ffffff','color':'#770000'
                                ,'text-align':'right','font-size':'20px', 'font-weight':'bold'})])

   
],style={'background-color':'#eeeeff','width':'84%','position':'absolute',
         'top':'0px','left':'8%','right':'8%'})],
         style={'width':'100%','position':'absolute',
                 'top':'0px','left':'0px'})


# ## Running thr app on Server

# In[ ]:


app.run_server()


# In[ ]:




