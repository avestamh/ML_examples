from flask import Flask, jsonify, request, render_template

import pickle
import numpy as np

# inititalize the flask appp
app = Flask(__name__)

## load the saved collaborative filtering models
with open('pivot_table.pkl', 'rb') as f:
    pt = pickle.load(f)

with open('similarity_scores.pkl', 'rb') as f:
    similarity_score = pickle.load(f)

# Recommendation function
def recommend(book_name):
    try:
        ## Find the index of the book in the pivot table
        index = np.where(pt.index == book_name)[0][0]
        # Sort the similarity scores for the book and get the top 5 most similar books (excluding itself)
        similar_book = sorted(enumerate(similarity_score[index]), key=lambda x: x[1], reverse=True )[1:6]

        # Collect the book title
        recommendations = [pt.index[i[0]] for i in similar_book]
        return recommendations
    except IndexError:
        return f'The book {book_name} was not find in the dataset'


# Root route to display something at the base URL
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    book_name = request.args.get('book_name')
    
    if not book_name:
        return jsonify({"error":"Book name is required!"}), 400
    
    recommendations = recommend(book_name)

    # return jsonify({'book_name': book_name, 'recommendations':recommendations}) ## return jsonify format
    return render_template('recommendations.html', book_name=book_name, recommendations=recommendations)
    
if __name__ == '__main__':
    app.run(debug=True)