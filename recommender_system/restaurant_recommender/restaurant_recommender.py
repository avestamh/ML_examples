import numpy as np
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

#____________________________Laoding the Data___________________________________
zomato_real = pd.read_csv('zomato.csv')
print(zomato_real.head())

# ___________________________Data Cleaning_____________________________________
#deleting unnecessary columns
zomato = zomato_real.drop(['url', 'dish_liked', 'phone'], axis=1)

# removing the duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

# remove the Nan value from the dataset
zomato.isnull().sum()
zomato.dropna(how='any', inplace=True)

# changing the column name
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})
## some transformation
zomato['cost'] = zomato['cost'].astype(str)  # changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',', '.')) # using lambda to replace , from cost
zomato['cost'] = zomato['cost'].astype(float)

# removeing /5 from the rates
zomato = zomato.loc[zomato.rate !="NEW"]
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype(float)

# adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title()) ##capitalize first letter of each work of the restaurant's names
zomato.online_order.replace(('Yes', 'No'), (True, False), inplace=True) # standardiz the column as bolean (binary) instead of string + memory efficency
zomato.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

#computing Mean Rating
restaurants = list(zomato.name.unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()

## an efficient way to do above is:
# zomato['Mean Rating'] = zomato.groupby('name')['rate'].transform('mean')
    
scaler = MinMaxScaler(feature_range=(1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

#_________________________Text Processing_____________________________________
# lower casing
zomato['reviews_list'] = zomato['reviews_list'].str.lower()

## Removal of Punctuation
import string
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato['reviews_list'] = zomato['reviews_list'].apply(lambda text:remove_punctuation(text))

## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    '''custom function to remove the stopwords'''
    return " ".join([word for word in str(text).split() if word not in STOPWORDS ])

zomato['reviews_list'] = zomato['reviews_list'].apply(lambda text: remove_stopwords(text))

# Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato['reviews_list'] = zomato['reviews_list'].apply(lambda text: remove_urls(text))
print(zomato[['reviews_list', 'cuisines']].sample(5))


## Restaurant Names:
restaurant_names = list(zomato['name'].unique())
#extracts the top N most frequent words (or n-grams) from a given text column.
def get_top_words(column, top_nu_of_words, nu_of_word):
    # convert text into a bag-of-word representation (unigram, bigram or more)
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')

    # Converts the input text column into a sparse matrix of token counts
    bag_of_words = vec.fit_transform(column)

    #Calculates the total count of each word across all rows
    sum_words = bag_of_words.sum(axis=0)

    # Creating a List of Word Frequencies:
    # vec.vocabulary_: A dictionary mapping each word to its index
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    # sorting by word frequency
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    # Returns the top N most frequent words (or n-grams) based on the specified count.
    return words_freq[:top_nu_of_words]

zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)
# Randomly sample %50 of your dataframe
df_percent = zomato.sample(frac=0.5)



#_____________TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization____________

df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Restaurant Recommendation System
def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new
recommend('Pai Vihar')
