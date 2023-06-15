# TRADEMARKIA PROJECT FOR AI INTERN
Artificial Intelligence Engineer Intern Project.
The provided code is a Flask application that builds a recommendation system based on a dataset called "ID Manual." The goal is to provide recommended classes based on user input of goods and services. Here is a breakdown of the process:
The code imports the necessary libraries: json, joblib, TfidfVectorizer from sklearn.feature_extraction.text, DecisionTreeClassifier from sklearn.tree, and Flask from flask.
The Flask application is initialized with app = Flask(__name__).
The code loads the dataset from the 'idmanual.json' file using json.load(file).
Preprocessing steps are performed on the dataset: extracting descriptions and class IDs, and applying TF-IDF vectorization using TfidfVectorizer.
A decision tree classifier is instantiated and trained on the preprocessed dataset using DecisionTreeClassifier.fit.
The trained classifier is saved to a file named 'model.pkl' using joblib.dump.
The Flask route /trademarkia is defined with the methods GET and POST.
If a POST request is received, the function get_recommended_classes is executed.
The input data is extracted from the request's JSON payload.
The user's input is transformed into a vector using the trained vectorizer.
The classifier predicts the class ID based on the user's input vector using DecisionTreeClassifier.predict.
The recommended classes are extracted from the dataset based on the predicted class ID.
The recommended classes are returned as a JSON response using jsonify.
If a GET request is received, a general message about the project is returned as a JSON response using jsonify.
Finally, the Flask application is run with app.run().
To run this code, make sure you have Python and the required libraries installed. Save the code in a file with a .py extension (e.g., recommendation.py). Then, open a terminal or command prompt, navigate to the directory where the file is saved, and execute the following command:
python recommendation.py
The Flask application will start running, and you can access the API endpoint at http://localhost:5000/trademarkia for testing the recommendation system.
