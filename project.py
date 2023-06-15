import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the USPTO ID manual dataset
with open('idmanual.json', 'r') as file:
    dataset = json.load(file)

# Preprocess the dataset
descriptions = [entry['description'] for entry in dataset]
class_ids = [entry['class_id'] for entry in dataset]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(descriptions)

# Build the recommendation model
classifier = DecisionTreeClassifier()
classifier.fit(X, class_ids)

# Save the model
joblib.dump(classifier, 'model.pkl')

@app.route('/trademarkia', methods=['GET', 'POST'])
def get_recommended_classes():
    if request.method == 'POST':
        data = request.json
        user_input = data['goods_and_services']
        user_vector = vectorizer.transform([user_input])

        # Predict the class
        predicted_class = classifier.predict(user_vector)[0]

        # Get the recommended classes
        recommended_classes = [entry['class_id'] for entry in dataset if entry['class_id'] == predicted_class]

        return jsonify({'recommended_classes': recommended_classes})
    else:
        return jsonify({'message': 'The project appears to be based on building a recommendation system using a dataset called "ID Manual." The goal is to provide recommended classes based on user input of goods and services. The code utilizes techniques such as TF-IDF vectorization and cosine similarity to find the most similar classes to the user input. The Flask framework is used to create a web application that exposes an API endpoint for receiving user input and returning the recommended classes..'})

if __name__ == '__main__':
    app.run()
