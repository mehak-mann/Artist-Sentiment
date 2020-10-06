#import packages
from gensim.summarization import keywords
from matplotlib import pyplot
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
from spacy.lang.en import English
nlp = English()
nlp.max_length = 10000000
import lyricsgenius
import pandas as pd
from textblob import TextBlob
import requests, json
import numpy as np
import matplotlib.ticker as plticker
import seaborn as sns
import time
from datetime import date
import matplotlib.dates as mdates

#define Genius API authentication
api_key = 'SfbYPF1AJ0-lnm6Km8_sIJoebvrIFfRyAGoZqxnRfkZIvP5ceGwBNZa4g0DHayP-'
genius = lyricsgenius.Genius(api_key)
BASE_URL = "https://api.genius.com"


def main():
    artist_name = ""
    while True:
        artist_name = input("Given the name of an artist (singer, rapper, etc.): ")
        if (getArtistId(artist_name)):
            break
        print("Artist does not exist (or incorrect spelling). Try again.")
    print("Please wait... ")
    dataset = getSongsWithDates(artist_name)
    print("Songs identified.")
    sentiment = calculateSentiment(artist_name, dataset)
    print("Sentiment calculated.")
    #plotSentimentOverTime(artist_name, sentiment)
    plotScatter2(artist_name, sentiment)
    print("Done.")

# ---------------------------------- Methods for fetching songs with LyricsGenius -------------------------------

def getSongsWithDates(artist_name):

    # find artist id from given artist name
    artist_id = getArtistId(artist_name)

    # get all song ids and make a list.
    song_ids = get_artist_songs(artist_id)

    # finally, make a full list of songs with required meta data.
    dataset = get_song_information(song_ids)
    #print(dataset)
    return dataset

def getArtistId(artist_name):
    # find artist id from given artist name
    find_id = _get("search", {'q': artist_name})
    for hit in find_id["response"]["hits"]:
        if hit["result"]["primary_artist"]["name"] == artist_name:
            artist_id = hit["result"]["primary_artist"]["id"]
            return artist_id
    return None

def get_song_information(song_ids):
    information = []
    # main loop
    for i, song_id in enumerate(song_ids):
        path = "songs/{}".format(song_id)
        data = _get(path=path)["response"]["song"]
        current = [data['title'].strip(u'\u200b'), data["release_date"].strip() if data["release_date"] else "unidentified", data["album"]["name"] if data["album"] else data["title"] + " (single)"]
        information.append(current)
    df = pd.DataFrame(information, columns = ['Song', 'WeekID', 'Album'])
    df = df.drop(df[df['WeekID'] == "unidentified"].index)
    return df

# send request and get response in json format.
def _get(path, params=None, headers=None):
    # generate request URL
    requrl = '/'.join([BASE_URL, path])
    token = "Bearer {}".format(api_key)
    if headers:
        headers['Authorization'] = token
    else:
        headers = {"Authorization": token}
    response = requests.get(url=requrl, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

def get_artist_songs(artist_id):
    # initialize variables & a list.
    current_page = 1
    next_page = True
    songs = []
    while next_page:
        path = "artists/{}/songs/".format(artist_id)
        params = {'page': current_page}
        data = _get(path=path, params=params)
        page_songs = data['response']['songs']
        if page_songs:
            # add all the songs of current page,
            # and increment current_page value for next loop.
            songs += page_songs
            current_page += 1
        else:
            # if page_songs is empty, quit.
            next_page = False

    # get all the song ids, excluding not-primary-artist songs.
    songs = [song["id"] for song in songs
             if song["primary_artist"]["id"] == artist_id]

    return songs

# ------------------------------ Methods for calculating sentiment + plotting ----------------------------------

def calculateSentiment(artist_name, dataset):
    # use get_lyrics funcion to get lyrics for every song in dataset
    lyrics = dataset.apply(lambda row: get_lyrics(row['Song'], artist_name), axis =1)
    dataset['Lyrics'] = lyrics
    dataset = dataset.drop(dataset[dataset['Lyrics'] == 'not found'].index) #drop rows where lyrics are not found on Genius

    # use get_lyric_sentiment to get sentiment score for all the song lyrics
    sentiment = dataset.apply(lambda row: get_lyric_sentiment(row['Lyrics']), axis =1)
    dataset['Sentiment'] = sentiment

    # set the index of the dataframe to the WeekID. This sets us up to resample dataframe based on time
    dataset['WeekID'] = pd.to_datetime(dataset['WeekID'],infer_datetime_format=True)
    dataset = dataset.sort_values(by='WeekID')
    dataset = dataset.reset_index(drop=True)

    return dataset

def plotScatter2(artist_name, data):
    data['Date_Ordinal'] = pd.to_datetime(data['WeekID']).apply(lambda date: date.toordinal())
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig, ax = pyplot.subplots()
    ax = sns.regplot(x='Date_Ordinal', y='Sentiment', data=data, marker="o", color="skyblue", scatter_kws={'s':400})
    for line in range(0,data.shape[0]):
        ax.text(data.Date_Ordinal[line]+0.2, data.Sentiment[line], data.Song[line], horizontalalignment='left', size='small', color='black')
    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # round to nearest years.
    datemin = np.datetime64(data['Date_Ordinal'][0], 'Y')
    datemax = np.datetime64(data['Date_Ordinal'][-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    pyplot.show()
    #Calculate the mean percentage change of lyric sentiment
    #print("Mean Percentage Change of Lyric Sentiment: " + data.pct_change().mean())

def plotScatter(artist_name, data):
    data['Date_Ordinal'] = pd.to_datetime(data['WeekID']).apply(lambda date: date.toordinal())
    size = data['Album'].nunique()
    if (size >= 10):
        size /= 2
    ax = sns.regplot(x='Date_Ordinal', y='Sentiment', data=data, marker="o", color="skyblue", scatter_kws={'s':400})
    for line in range(0,data.shape[0]):
        ax.text(data.Date_Ordinal[line]+0.2, data.Sentiment[line], data.Song[line], horizontalalignment='left', size='small', color='black')
    ax.xaxis.set_major_locator(pyplot.MaxNLocator(size))
    ax.yaxis.set_major_locator(pyplot.MaxNLocator(size))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax.set_xlabel('WeekID')
    new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
    ax.set_xticklabels(new_labels)
    ax.set_title(artist_name + ": Song Sentiment over Time")
    ax.text(0.85, 0.85,'KeyWords: jalskfjlaf\nPercent Change of Lyric Sentiment: akjshasf', fontsize=9)
    pyplot.savefig(artist_name + ": Song Sentiment over Time.jpeg")
    pyplot.show()


def plotSentimentOverTime(artist_name, dataset):
    # Averaging Sentiment with the same week id
    dataset['AvgSentiment'] = dataset.groupby('WeekID')['Sentiment'].transform('mean')
    data = dataset.drop(columns=['Lyrics', 'Song', 'Sentiment'])
    data.drop_duplicates(subset='WeekID', inplace = True) #remove duplicate occurrences of songs
    data = data.reset_index(drop=True)
    #data['Album'] = data['Album'] + '\n(' + str(data['WeekID']).split()[1] + ')'
    data['Album'] = data["Album"] + '\n(' + str(data["WeekID"].map(str)).split()[1] + ')'

    # Remove when confident
    print(data)

    # Plotting the average sentiment by week
    fig, ax = pyplot.subplots()
    ax.plot(data['WeekID'],data['AvgSentiment'])
    ax.set_xlabel('Time (by Album Release)')
    ax.set_ylabel('Sentiment')
    ax.set_xticklabels(data['Album'], rotation=20)
    ax.set_title(artist_name + ": Song Sentiment over Time")
    pyplot.savefig(artist_name + ": Song Sentiment over Time.jpeg")
    pyplot.show()

    sns.lmplot(x=str(data["WeekID"].map(str)).split()[1], y=data['AvgSentiment'], data=data)
    pyplot.title("Scatter Plot with Linear fit")

    #ax.set_title(artist_name + ": Song Sentiment over Time")
    pyplot.savefig(artist_name + ": Song Sentiment over Time.jpeg")
    pyplot.show()


# ------------------------------------ Helper Methods for Sentiment Analysis -----------------------------------

def get_lyrics(title, artist):
    '''
    Function to return lyrics of each song using Genius API
    '''
    try:
        return genius.search_song(title, artist).lyrics
    except:
        return 'not found'

def get_lyric_sentiment(lyrics):
	'''
	Function to return sentiment score of each song
	'''
	analysis = TextBlob(lyrics)
	return analysis.sentiment.polarity

# Function to preprocess text
def preprocess(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in stopwords]

    return ' '.join(a_lemmas)

"""Extract Keywords from text"""
def return_keywords(texts):
    xkeywords = []
    values = keywords(text=preprocess(texts),split='\n',scores=True)
    for x in values[:10]:
        xkeywords.append(x[0])
    try:
        return xkeywords
    except:
        return "no content"


if __name__ == '__main__':
    main()

