#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral
from bokeh.palettes import Spectral6, Magma, Inferno
from bokeh.themes import built_in_themes
from bokeh.io import curdoc

from datetime import date, timedelta
from IPython import get_ipython
from PIL import Image
from streamlit import caching
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components


# In[12]:


st.title('Elissa\'s Angels Inc.')


# In[13]:


image = Image.open('img/eskwelabs_logo.jpg')
st.sidebar.image(image, caption='', use_column_width=True)
st.sidebar.markdown("<h1 style='text-align: center;margin-bottom:50px'>Eskwelabs Data Science Fellowship Cohort V</h1>", unsafe_allow_html=True)


add_selectbox = st.sidebar.radio(
    "",
    ("Introduction and Problem Statement", "List of Tools", "Data Sourcing", "Data Set", "Data Cleaning", 
     "Exploratory Data Analysis", "Pipeline", "Recommender Engine", 
     "Results", "Contributors")
)


# In[14]:


if add_selectbox == 'Introduction and Problem Statement':
    st.write('')
    
    st.subheader('Introduction')
    st.write('-----------------------------------------------------------------------') 
    st.write('<b>BUSINESS OBJECTIVE:</b>', unsafe_allow_html=True)
    st.write('Provide unique and actionable insights and strategies on how to boost streams of the artists they handle in the market')
    
    st.write('<b>CLIENT SCENARIO:</b>', unsafe_allow_html=True)
    st.write("""
    Elissa's Angels is a record label based in Manila, Philippines. Their artists were previously performing well but are recently dropping from
    the charts. They reached out to the Eskwelabs Data Science team to get help on how to get their artists back up in the charts. 
    """, unsafe_allow_html=True)
    st.write("ARTIST ROSTER:")
    st.markdown("<ul>"                "<li>Darren Espanto</li>"                "<li>Clean Bandit</li>"                "<li>Troye Sivan</li>"                "<li>Juan Karlos</li>"                "<li>IV Of Spades</li>"                "</ul>", unsafe_allow_html=True)
    
    st.write("""
    The team looked into collaborations as a possible way to improve the performance of their artists in terms of streams.
    The chart below shows the comparison of collaboration vs non-collaboration tracks performance over the period of ~3 years:
    """, unsafe_allow_html=True)
    
    image = Image.open('img/collab_noncollab.png').convert('RGB')
    st.image(image, caption='')


# In[15]:


elif add_selectbox == 'Outline':
    st.subheader('Outline')
    st.write('-----------------------------')


# In[16]:


elif add_selectbox == 'List of Tools':
    st.subheader('List of Tools')
    st.write('-----------------------------')
    st.write('-----------------------------------------------------------------------') 
    image = Image.open('img/spotify.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('img/jupyter.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('img/pandas.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    #image = Image.open('img/heroku.jpg').convert('RGB')
    #st.image(image, caption='', width=150, height=50)
    image = Image.open('img/streamlit.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    #image = Image.open('img/bokeh.png').convert('RGB')
    #st.image(image, caption='', width=300, height=150)
    #image = Image.open('img/github.png').convert('RGB')
    #st.image(image, caption='', width=300, height=150)
    #image = Image.open('img/regex.jpeg').convert('RGB')
    #st.image(image, caption='', width=300, height=150)
    #image = Image.open('img/scipy.png').convert('RGB')
    #st.image(image, caption='', width=300, height=150)
    image = Image.open('img/seaborn.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('img/matplotlib.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('img/numpy.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)


# In[ ]:


elif add_selectbox == 'Data Sourcing':
    st.subheader('Data Sourcing')
    st.write('-----------------------------')
    
    #st.write('<b></b>', unsafe_allow_html=True)
    #st.markdown('<b>Data from January 2018 - September 2020</b>', unsafe_allow_html=True)
    
    st.write("""
    The project used Spotify’s API data on the daily Top 200 charts in the Philippines from January 2018 to September 2020. Data was scraped instead of being manually downloaded to ease the data extraction process. The extracted data includes the tacks’ audio features such as key, mode, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, and tempo.
    For our clients' artist data, we extracted their entire discography from Spotify's API.

    More information on the tracks’ audio features can be found here: https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/
    """)
    
    DATA_URL = ('data/team_elissa_discography_tracks.csv')
    
    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data)
    
    df=data
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 
                'speechiness', 'tempo', 'valence', 'popularity']

    scaler = MinMaxScaler()
    df['loudness'] = scaler.fit_transform(df[['loudness']])
    df['tempo'] =  scaler.fit_transform(df[['tempo']])

    tracks1_df = df[df['artist'] == "Clean Bandit"]

    for col in ['popularity', 'danceability', 'energy',
           'loudness', 'speechiness', 'acousticness', 'instrumentalness',
           'liveness', 'valence', 'tempo']:
        fig = plt.figure()
        ax= fig.add_subplot(111)
    
        sns.distplot(tracks1_df[col], ax=ax, label= "Clean Bandit")
        xc = tracks1_df[col].median()
        plt.axvline(x=xc, color='red')
    #sns.distplot(tracks2_df[col], ax=ax, label= KEYWORD2)
    #plt.title("%s vs %s: %s " % (KEYWORD1,KEYWORD2,col))
        plt.ylabel('Frequency')
        plt.legend(frameon=False)
        st.pyplot(fig)
    


# In[ ]:


elif add_selectbox == 'Data Set':
    st.subheader('Data Set')
    st.write('-----------------------------')
    
    st.write('<b>SPOTIFY DAILY CHARTS TRACKS:</b>', unsafe_allow_html=True)
    st.markdown('<b>Data from January 2018 - September 2020:</b>', unsafe_allow_html=True)
    
    DATA_URL = ('data/spotify_daily_charts.csv')
        
    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data)
    st.markdown('<b>Data Dimensions:</b> Rows: 197800 Columns: 6', unsafe_allow_html=True)
    
    ################################################
    
    st.write('<b>SPOTIFY DAILY CHARTS TRACKS W/ FEATURES</b>', unsafe_allow_html=True)
    st.markdown('<b>Data from January 2018 - September 2020</b>', unsafe_allow_html=True)
    
    DATA_URL2 = ('data/spotify_daily_charts_tracks.csv')
    
    @st.cache
    def load_data(nrows):
        data2 = pd.read_csv(DATA_URL2, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data2.rename(lowercase, axis='columns', inplace=True)
        return data2
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data2 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data2)
    st.markdown('<b>Data Dimensions:</b> Rows: 2292 Columns: 19', unsafe_allow_html=True)
    
    ################################################
    
    st.write('<b>MERGED DATASET</b>', unsafe_allow_html=True)
    st.markdown('<b>Data from January 2018 - September 2020</b>', unsafe_allow_html=True)
    
    DATA_URL3 = ('data/df_merged.csv')
    
    @st.cache
    def load_data(nrows):
        data3 = pd.read_csv(DATA_URL3, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data3.rename(lowercase, axis='columns', inplace=True)
        return data3
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data3 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data3)
    st.markdown('<b>Data Dimensions:</b> Rows: 197800 Columns: 23', unsafe_allow_html=True)
    
    ##################################################
    
    st.write('<b>CLIENT DSCOGRAPHY</b>', unsafe_allow_html=True)
    st.markdown('<b>Data from January 2018 - September 2020</b>', unsafe_allow_html=True)
    
    DATA_URL4 = ('data/raw_client.csv')
    
    @st.cache
    def load_data(nrows):
        data4 = pd.read_csv(DATA_URL4, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data4.rename(lowercase, axis='columns', inplace=True)
        return data4
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data4 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data4)
    st.markdown('<b>Data Dimensions:</b> Rows: 59 Columns: 14', unsafe_allow_html=True)


# In[18]:


elif add_selectbox == 'Data Cleaning':
    st.subheader('Data Cleaning')
    st.write('-----------------------------')
    
    st.write('<b>FINAL TRACKS DATASET:</b>', unsafe_allow_html=True)
    st.markdown('<b>Data from January 2018 - September 2020:</b>', unsafe_allow_html=True)
    st.write('Inspired by the RFM model, we took the quartiles for streams and positions and added them together to derive a performance metric.', unsafe_allow_htm=True)
    
    DATA_URL = ('data/SumVal.csv')
        
    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data)
    st.markdown('<b>Data Dimensions:</b> Rows: 2292 Columns: 33', unsafe_allow_html=True)
    
    ################################################
    
    st.write('<b>FINAL CLIENT DATASET', unsafe_allow_html=True)
    st.markdown('<b>Data from January 2018 - September 2020</b>', unsafe_allow_html=True)
    
    DATA_URL2 = ('data/final_client_data.csv')
    
    @st.cache
    def load_data(nrows):
        data2 = pd.read_csv(DATA_URL2, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data2.rename(lowercase, axis='columns', inplace=True)
        return data2
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data2 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data2)
    st.markdown('<b>Data Dimensions:</b> Rows: 6 Columns: 13', unsafe_allow_html=True)
    
    #################################################
    
    st.write('<b>MODELLING FEATURES:</b>', unsafe_allow_html=True)
    st.markdown("<ul>"                "<li>Duration</li>"                "<li>Acousticness</li>"                "<li>Danceability</li>"                "<li>Energy</li>"                "<li>Instrumentalness</li>"                "<li>Liveness</li>"                "<li>Loudness</li>"                "<li>Speechiness</li>"                "<li>Tempo</li>"                 "<li>Valence</li>"                 "<li>Mode</li>"                 "<li>Key</li>"                "</ul>", unsafe_allow_html=True)


# In[ ]:


elif add_selectbox == 'Exploratory Data Analysis':
    st.subheader('Exploratory Data Analysis')
    st.write('-----------------------------')
    
    DATA_URL = ('data/merged_charts_tracks_data.csv')
        
    @st.cache
    def load_data(nrows):
        df = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df.rename(lowercase, axis='columns', inplace=True)
        return df
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(df)
    
    merged = df
    
    merged = pd.read_csv('data/merged_charts_tracks_data.csv')

    all_other_df = merged[merged['position'] < 51].groupby('date').mean()[['streams']] # Based on top 50 on charts
#all_other_df


    test = merged[merged['artist_name'] == "Clean Bandit"].groupby("track_name").sum()[['streams']]         .sort_values("streams", ascending = False)
    test = test.reset_index()

    clean_df = pd.DataFrame()


    for name in test['track_name']:
        clean_df = clean_df.append(merged[merged['track_name'] == name])
        #print(name)
    
    test = merged[merged['artist_name'] == "Darren Espanto"].groupby("track_id").sum()[['streams']]         .sort_values("streams", ascending = False)
    test = test.reset_index()

    darren_df = pd.DataFrame()

    for name in test['track_id']:
        darren_df = darren_df.append(merged[merged['track_id'] == name])
        #print(name)
    darren_df.groupby("track_name").last()

    test = merged[merged['artist_name'] == "Troye Sivan"].groupby("track_name").sum()[['streams']]         .sort_values("streams", ascending = False)
    test = test.reset_index()

    troye_df = pd.DataFrame()

    for name in test['track_name']:
        troye_df = troye_df.append(merged[merged['track_name'] == name])
        #print(name)
    troye_df.groupby("track_name").last()

    test = merged[merged['artist_name'] == "juan karlos"].groupby("track_name").sum()[['streams']]         .sort_values("streams", ascending = False)
    test = test.reset_index()

    juan_df = pd.DataFrame()

    for name in test['track_name']:
        juan_df = juan_df.append(merged[merged['track_name'] == name])
        #print(name)
    juan_df.groupby("track_name").last()

    test = merged[merged['artist_name'] == "IV Of Spades"].groupby("track_name").sum()[['streams']]         .sort_values("streams", ascending = False)
    test = test.reset_index()

    spades_df = pd.DataFrame()

    for name in test['track_name']:
        spades_df = spades_df.append(merged[merged['track_name'] == name])
        #print(name)
    spades_df.groupby("track_name").last()



    clean_df.groupby("track_name").last()
    all_other_df['clean'] =clean_df.groupby('date')['streams'].mean()
    all_other_df['juan'] =juan_df.groupby('date')['streams'].mean()
    all_other_df['darren'] =darren_df.groupby('date')['streams'].mean()
    all_other_df['troye'] =troye_df.groupby('date')['streams'].mean()
    all_other_df['spades'] =spades_df.groupby('date')['streams'].mean()

    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
#clean_df = clean_df.set_index('date')
#juan_df = juan_df.set_index('date')

#spades_df = spades_df.set_index('date')

#troye_df = troye_df.set_index('date')
#darren_df = darren_df.set_index('date')
#all_other_df = merged.groupby('date').mean()['streams']

    data1 = all_other_df['clean'].rolling(7).mean()
    data2 =all_other_df['juan'].rolling(7).mean()
    data3 = all_other_df['darren'].rolling(7).mean()
    data4 = all_other_df['troye'].rolling(7).mean()
    data5 = all_other_df['spades'].rolling(7).mean()

    data6 = all_other_df['streams'].rolling(7).mean()



    data6.plot(ax=ax, label='Mean of every song in dataset')

    data1.plot(ax=ax, label='Clean Bandit songs')
    data2.plot(ax=ax, label='juan karlos songs')
    data3.plot(ax=ax, label='Darren Espanto songs')
    data4.plot(ax=ax, label='Troye Sivan songs')
    data5.plot(ax=ax, label='IV of Spades songs')

    plt.legend()
    plt.ylabel('streams')
    plt.title('Spotify Daily Streams')

    st.pyplot(fig)
    
    ##########################################################
    ##########################################################
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
#import matplotlib.dates as mdates
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#get top position of all charting songs of the artist per day
    data1 = spades_df.groupby('date')[['position']].min()
#get rolling 7 day minimum of top daily positions
    data1 = data1.rolling(7).min()
    data2 = juan_df.groupby('date')[['position']].min()
    data2 = data2.rolling(7).min()
    data3 = darren_df.groupby('date')[['position']].min()
#get rolling 7 day minimum of top daily positions
    data3 = data3.rolling(7).min()
    data4 = troye_df.groupby('date')[['position']].min()
    data4 = data4.rolling(7).min()
    data5 = clean_df.groupby('date')[['position']].min()
#get rolling 7 day minimum of top daily positions
    data5 = data5.rolling(7).min()




    data1.plot(ax=ax, label='Clean Bandit')
    data2.plot(ax=ax, label='XXXTENTACION')
    data3.plot(ax=ax, label='Clean Bandit')
    data4.plot(ax=ax, label='XXXTENTACION')
    data5.plot(ax=ax, label='Clean Bandit')


#reverse the yaxis to show 1 on top
    plt.ylim([200,0])
    plt.yticks([1]+np.arange(25,201,25).tolist())

    L = plt.legend()
    L.get_texts()[0].set_text('IV Of Spades')
    L.get_texts()[1].set_text('juan karlos')
    L.get_texts()[2].set_text('Darren Espanto')
    L.get_texts()[3].set_text('Troye Sivan')
    L.get_texts()[4].set_text('Clean Bandit')

    plt.axhline(y=100.5, color='y', linestyle='-')
#plt.show()
    plt.ylabel('Chart Position')
    plt.title('Spotify Weekly Top Chart Positions')
#ax.set_xticklabels(data1.index)
    date = [data1.index[0] , data1.index[120], data1.index[240] , data1.index[360], data1.index[480] , data1.index[600],
       data1.index[720] , data1.index[953]]
    ax.set_xticklabels(date)
    
    st.pyplot(fig)


# In[ ]:


elif add_selectbox == 'Client Track Classification':
    st.subheader('Client Track Classification')
    st.write('-----------------------------')


# In[ ]:


elif add_selectbox == 'Recommender Engine':
    st.subheader('Recommender Engine')
    st.write('-----------------------------')
    
    st.write('<b>Clean Bandit Recommended Tracks:</b>', unsafe_allow_html=True)
    st.markdown('<b>___:</b>', unsafe_allow_html=True)
    
    DATA_URL = ('data/clean_bandit_recommend_df.csv')
        
    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data)
    
    ##########################################################
    
    st.write('<b>Darren Espanto Recommended Tracks:</b>', unsafe_allow_html=True)
    st.markdown('<b>____</b>', unsafe_allow_html=True)
    
    DATA_URL1 = ('data/darren_espanto_recommend_df.csv')
        
    @st.cache
    def load_data(nrows):
        data1 = pd.read_csv(DATA_URL1, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data1.rename(lowercase, axis='columns', inplace=True)
        return data1
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data1 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data1)
    
    ##########################################################
    
    st.write('<b>IV Of Spades Recommended Tracks:</b>', unsafe_allow_html=True)
    st.markdown('<b>___</b>', unsafe_allow_html=True)
    
    DATA_URL2 = ('data/iv_spades_recommend_df.csv')
        
    @st.cache
    def load_data(nrows):
        data2 = pd.read_csv(DATA_URL2, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data2.rename(lowercase, axis='columns', inplace=True)
        return data2
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data2 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data2)
    
    ##########################################################
    
    st.write('<b>Juan Karlos Recommended Tracks</b>', unsafe_allow_html=True)
    st.markdown('<b>___</b>', unsafe_allow_html=True)
    
    DATA_URL3 = ('data/juan_carlos_recommend_df.csv')
        
    @st.cache
    def load_data(nrows):
        data3 = pd.read_csv(DATA_URL3, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data3.rename(lowercase, axis='columns', inplace=True)
        return data3
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data3 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data3)
    
    ##########################################################
    
    st.write('<b>Troye Sivan Recommended Tracks</b>', unsafe_allow_html=True)
    st.markdown('<b>___</b>', unsafe_allow_html=True)
    
    DATA_URL4 = ('data/troye_sivan_recommend_df.csv')
        
    @st.cache
    def load_data(nrows):
        data4 = pd.read_csv(DATA_URL4, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data4.rename(lowercase, axis='columns', inplace=True)
        return data4
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data4 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.write(data4)


# In[ ]:


elif add_selectbox == 'Pipeline':
    st.subheader('Model Pipeline')
    st.write('-----------------------------')
    
    image = Image.open('img/pipeline.png').convert('RGB')
    st.image(image, caption='')


# In[ ]:


elif add_selectbox == 'Results':
    st.subheader('Results')
    st.write('-----------------------------')
 
    image = Image.open('img/model.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)

    DATA_URL = ('data/clean_bar1.csv')
        
    @st.cache
    def load_data(nrows):
        df = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df.rename(lowercase, axis='columns', inplace=True)
        return df
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    clean_bar = df
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(clean_bar['artist'], clean_bar['1'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    st.pyplot(fig)
    
    #############################################
    
    DATA_URL2 = ('data/clean_bar2.csv')
        
    @st.cache
    def load_data(nrows):
        df2 = pd.read_csv(DATA_URL2, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df2.rename(lowercase, axis='columns', inplace=True)
        return df2
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df2 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    clean_bar = df2
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(clean_bar['artist'], clean_bar['success_probability'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    plt.title("Highest probability from entire dataset") 
    st.pyplot(fig)
    
    ###
    
    DATA_URL3 = ('data/juan_bar1.csv')
        
    @st.cache
    def load_data(nrows):
        df3 = pd.read_csv(DATA_URL3, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df3.rename(lowercase, axis='columns', inplace=True)
        return df3
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df3 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    juan_bar = df3
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(juan_bar['artist'], juan_bar['1'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    st.pyplot(fig)
    ##
    
    DATA_URL4 = ('data/juan_bar2.csv')
        
    @st.cache
    def load_data(nrows):
        df4 = pd.read_csv(DATA_URL4, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df4.rename(lowercase, axis='columns', inplace=True)
        return df4
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df4 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    juan_bar = df4
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(juan_bar['artist'], juan_bar['success_probability'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    plt.title("Highest probability from entire dataset")
    st.pyplot(fig)
    
    ##
    
    DATA_URL5 = ('data/troye_bar1.csv')
        
    @st.cache
    def load_data(nrows):
        df5 = pd.read_csv(DATA_URL5, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df5.rename(lowercase, axis='columns', inplace=True)
        return df5
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df5 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    troye_bar = df5
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(troye_bar['artist'], troye_bar['1'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    st.pyplot(fig)
    
    ##
    
    DATA_URL6 = ('data/troye_bar2.csv')
        
    @st.cache
    def load_data(nrows):
        df6 = pd.read_csv(DATA_URL6, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df6.rename(lowercase, axis='columns', inplace=True)
        return df6
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df6 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    troye_bar = df6
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(troye_bar['artist'], troye_bar['success_probability'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    plt.title("Highest probability from entire dataset") 
    st.pyplot(fig)
    
    ##
    
    DATA_URL7 = ('data/darren_bar1.csv')
        
    @st.cache
    def load_data(nrows):
        df7 = pd.read_csv(DATA_URL7, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df7.rename(lowercase, axis='columns', inplace=True)
        return df7
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df7 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    darren_bar = df7
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(darren_bar['artist'], darren_bar['1'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    st.pyplot(fig)
    ##
    
    DATA_URL8 = ('data/darren_bar2.csv')
        
    @st.cache
    def load_data(nrows):
        df8 = pd.read_csv(DATA_URL8, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df8.rename(lowercase, axis='columns', inplace=True)
        return df8
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df8 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    darren_bar = df8
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(darren_bar['artist'], darren_bar['success_probability'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    plt.title("Highest probability from entire dataset") 
    st.pyplot(fig)
    
    ##
    
    DATA_URL9 = ('data/ivos_bar1.csv')
        
    @st.cache
    def load_data(nrows):
        df9 = pd.read_csv(DATA_URL9, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df9.rename(lowercase, axis='columns', inplace=True)
        return df9
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df9 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    ivos_bar = df9
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(ivos_bar['artist'], ivos_bar['1'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    st.pyplot(fig)
    ##
    
    DATA_URL10 = ('data/ivos_bar2.csv')
        
    @st.cache
    def load_data(nrows):
        df10 = pd.read_csv(DATA_URL10, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        df10.rename(lowercase, axis='columns', inplace=True)
        return df10
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    df10 = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    
    ivos_bar = df10
    
    fig = plt.figure(figsize=(13,4))
    ax = fig.add_subplot(111)
    plt.bar(ivos_bar['artist'], ivos_bar['success_probability'], width = 0.4)
    plt.xlabel("Artists involved") 
    plt.ylabel("Success Probability") 
    plt.xticks(rotation = 45)
    plt.title("Highest probability from entire dataset") 
    st.pyplot(fig)
    
    ##Embed Spotify Playlist 
#import streamlit.components.v1 as components
   
    st.write('Below is the tracks based on the results of our model.')
    st.write('https://open.spotify.com/playlist/4tZSAyv6KxNDwrL5KSW5DZ')


# In[ ]:


else:
    st.subheader('Contributors')
    st.write('-----------------------------')
    st.markdown("<ul>"                "<li>Alphonso Balagtas</li>"                "<li>Elissa Mae Cabal</li>"
                "<li>Joleil Villena</li>"\
                "<li>Railenne Mae Ferrer </li>"\
                "<li>Roberto Bañadera Jr.</li>"\
                 "</ul>", unsafe_allow_html=True)

