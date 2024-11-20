import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# Load the datasets
books = pd.read_csv('dataset/Books.csv')[['ISBN', 'Book-Title']]
ratings = pd.read_csv('dataset/Ratings.csv')
users = pd.read_csv('dataset/Users.csv')

# Show the first few rows of the books dataset
print(books.head())

# ___________________Popularity based filtering______________________________

# Merge books and ratings on ISBN, merge users and ratings on User-ID
book_rating = books.merge(ratings, on='ISBN')
user_rating = users.merge(ratings, on='User-ID')

print('book_rating: \n', book_rating.head())
print("user_rating: \n", user_rating.head())

# Create two new DataFrames to have the number of ratings and the average rating
book_num_ratings = book_rating.groupby('Book-Title')['Book-Rating'].count().reset_index().rename(columns={'Book-Rating': 'Num-Rating'})
book_avg_ratings = book_rating.groupby('Book-Title')['Book-Rating'].mean().reset_index().rename(columns={'Book-Rating': 'Avg-Rating'})
final_rating = book_num_ratings.merge(book_avg_ratings, on='Book-Title')

print(final_rating.head())

# Filter books with more than 250 ratings (for popularity-based filtering)
popular_books = final_rating[final_rating['Num-Rating'] > 250].sort_values(by='Avg-Rating', ascending=False).reset_index(drop=True).head(50)
print(popular_books.head(5))

# ________________ Collaborative Filtering __________________
x = book_rating.groupby('User-ID').count()['Book-Rating'] > 200
print(x.head(5))

# Users who rated more than 200 books
educated_user = x[x].index

# Filter book ratings to include only users who rated more than 200 books
book_rating = book_rating[book_rating['User-ID'].isin(educated_user)]

# Get books with at least 50 ratings
y = book_rating.groupby('Book-Title')['Book-Rating'].count() >= 50
famous_book = y[y].index

# Filter books with at least 50 ratings and users who rated at least 200 books
final = book_rating[book_rating['Book-Title'].isin(famous_book)]
print(final.head())

# Create a pivot table: books as rows, users as columns, and ratings as values
pt = final.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
print(pt.head())

# Use cosine similarity to calculate the similarity between each book
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(pt)
print('Similarity Scores: \n', similarity_scores[:5], '\n', 50 * '*')


## Save the Piovot Table and Similarity Scores Using Pickle
with open('pivot_table.pkl', 'wb') as f:
    pickle.dump(pt, f)

with open('similarity_scores.pkl', 'wb') as f:
    pickle.dump(similarity_scores, f)

# Function to recommend 5 similar books
def recommend(book_name):
    try:
        # Find the index of the book in the pivot table
        index = np.where(pt.index == book_name)[0][0]
        # Sort the similarity scores for the book and get the top 5 most similar books (excluding itself)
        similar_books = sorted(enumerate(similarity_scores[index]), key=lambda x: x[1], reverse=True)[1:6]
        
        # Print the titles of the most similar books
        for i in similar_books:
            print(pt.index[i[0]])
    
    except IndexError:
        print(f"The book '{book_name}' was not found in the dataset.")

# Recommend similar books to '4 Blondes'
recommend('4 Blondes')
