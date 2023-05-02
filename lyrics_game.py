#  **Building a text classification model to predict the artist based on a piece of lyrics**
import os
import re
import string
import random
import shutil
import requests
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# alternatively the TFidfVectorizer can be used
# from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordDetokenizer
from nltk.corpus import stopwords
from rich.progress import track
from art import tprint

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordDetokenizer()

def scrape_song_lyrics(artists,
                       limit=10):
    '''scraping of song lyrics of specified artist from webpage'''

    lyrics_url = 'https://www.lyrics.com/artist/'
    print('\n')

    if os.path.exists('./song_lyrics'):
        # remove folder to clear existing song files
        shutil.rmtree('./song_lyrics')
    # create new folder
    os.makedirs('./song_lyrics')
    
    for artist in artists:

        artist_url = lyrics_url + artist.replace(' ','-')
        page_content = requests.get(artist_url)

        # scrape song lyrics
        if page_content.status_code == 200:
            song_links_list = re.findall('<a href="/lyric.*?">', page_content.text)
            song_links_list = random.sample(song_links_list, limit)
            prefix = "https://www.lyrics.com"
            
            songs = []
            count = 1
            
            for link in track(song_links_list, description=f'[green]Scraping songs from {artist}'):
                
                if count > limit:
                    break
                    
                start_idx = link.find('/lyric')
                song_url = ''.join([prefix, link[start_idx:]])

                song_page = requests.get(song_url)
                # extract song title
                song_name = link[link.rfind('/')+1:-2]
                song_name = song_name.replace('+','_')

                if song_name not in songs:                        
                    songs.append(song_name)                    
                    song_html = song_page.text
                    song_soup = BeautifulSoup(song_html, 'html.parser')
                    song_text_body = song_soup.body.find("pre", id="lyric-body-text")

                    if song_text_body:
                        count += 1
                        song_text = song_text_body.get_text()
                        song_text = re.sub(pattern=r"(\r\n|\r|\n)", repl=' ', string=song_text)
                        # store lyrics in .txt file
                        with open(f"./song_lyrics/{artist + '_' + song_name}.txt",'w') as file:
                            file.write(song_text)
                else:
                    continue         
        else:
            print('\033[91m' + f'{artist} not found!')



def load_lyrics():
    '''Create lyrics corpus of scraped song texts'''

    corpus = []
    artists = []
    for filename in os.listdir('./song_lyrics/'):
        
        artist = filename.split('_')[0]
        artists.append(artist)
        
        with open('./song_lyrics/' + filename,'r') as song:
            text = song.read()
            corpus.append(text)
    
    return corpus, artists
 


def tokenize_lemmatize(text,
                       stopwords= set(stopwords.words('english')),
                       lemmatizer=lemmatizer,
                       tokenizer=tokenizer):
    '''Processes text corpus by tokenization and lemmatization'''

    text = "".join([i for i in text if i not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords]



def bow(text, 
        artist_labels,
        vectorizer,
        tokenizer=tokenize_lemmatize,
        stops=None):
    '''Create Bag of Words Model from song lyrics'''
    
    VR = vectorizer(tokenizer=tokenizer, stop_words=stops)
    X_vec = VR.fit_transform(text)
    X_vec_df = pd.DataFrame(X_vec.todense(), columns=VR.get_feature_names_out(), index=artist_labels)
    
    return VR, X_vec, X_vec_df



def get_artist():
    '''Query artists'''
    artist = input('\033[96m' + 'Enter artist: ' + '\033[92m')
    return artist
    


def fit_model(vec, artist_labels):
    '''Trains Multinomial Naive Bayes Model on the lyrics of selected artists'''
    model = MultinomialNB()
    model.fit(vec, artist_labels)
    return model



def predict_artist(vectorizer, model):
    '''predicts artists based on the given lyrics sample'''

    print('\n')
    sample = input('\033[97m' + ' Enter your lyrics sample: ' + '\033[91m')
    sample_list = []
    sample_list.append(sample)
    test_vec = vectorizer.transform(sample_list)
    prediction = model.predict(test_vec)[0]
    probas = model.predict_proba(test_vec)

    print('\n')
    print("\033[92m" + "The lyrics '" + "\033[93m" + sample + "\033[92m" + "' were probably sang by:", "\033[91m" + prediction, f'({round(np.max(probas)*100,2)}%)')


def main():
    '''Runs the lyrics classifier game'''

    # start the game
    print('\033[94m')
    tprint("Who's the singer?")

    status = True
    artists = []
    while status:
        artists.append(get_artist())
        if len(artists) >= 2:
            answer = input('\033[96m' + 'Do you want to enter another artist? [y/n]: ' + '\033[92m')

            if answer == 'n':
                status = False

    limit = input('\033[96m' + 'How many songs per artist do you want do load?: ' + '\033[92m')
    try:
        scrape_song_lyrics(artists,int(limit))
    except:
        print('No number specified, default number of 10 songs will be used')
        scrape_song_lyrics(artists)

    corpus, artist_labels = load_lyrics()
    VecRiz, vec, vec_df = bow(corpus, artist_labels, vectorizer=CountVectorizer)
    ## or use the TFidfVectorizer instead for frequency normalization
    # VecRiz, vec, vec_df = BoW(corpus, artist_labels, vectorizer=TfidfVectorizer)

    #train the model
    model = fit_model(vec, artist_labels)

    stop = False
    while not stop:

        predict_artist(VecRiz, model)
        answer = input('\033[96m' + 'Do you wanna go again? [y/n] ' + '\033[92m')
        if answer == 'n':
            stop = True

    print('\033[94m')
    tprint('Goodbye')


if __name__ == '__main__':
    main()
