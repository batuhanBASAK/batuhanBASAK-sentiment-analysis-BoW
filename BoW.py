import pickle
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report




class BoW:
    def __init__(self):
        self.is_trained = False


    def save_to_file(self, filename):
        if not self.is_trained:
            print("The model which hasn't trained cannot be saved to a file!!!")
            return

        # Create a dictionary with all the necessary components
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }

        # Save the model data to a file
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)

        print(f"Model saved to {filename}")


    def load_from_file(self, filename):
        try:
            # Load the model data from the file
            with open(filename, 'rb') as file:
                model_data = pickle.load(file)

            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.is_trained = True

            print(f"Model loaded from {filename}")

        except FileNotFoundError:
            print(f"Error: The file {filename} was not found.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")


        
    def train(self, X, y):

        # Split data into training and testing sets
        print('\nCreating training and testing datasets...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert text data to Bag of Words representation
        self.vectorizer = CountVectorizer()
        X_train_bow = self.vectorizer.fit_transform(X_train)
        X_test_bow = self.vectorizer.transform(X_test)


        # Train the model by counting each token for sentiments
        print("\nTraining...")
        
        # Create a dictionary to hold models and their hyperparameters
        models = {
            'RandomForest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': MultinomialNB()
        }

        param_grid = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'NaiveBayes': {
                'alpha': [0.1, 1.0, 10.0]
            }
        }

        best_model = None
        best_score = 0
        best_model_name = ""

        # Iterate over all models and perform grid search for each one
        for model_name, model in models.items():
            print(f"\nPerforming cross-validation for {model_name}...")
            
            grid_search = GridSearchCV(model, param_grid[model_name], cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
            grid_search.fit(X_train_bow, y_train)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score for {model_name}: {grid_search.best_score_:.2f}")
            
            # Select the best model based on cross-validation score
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_model_name = model_name

        # Final model is the best model found from the grid search
        print(f"\nBest model: {best_model_name} with score: {best_score:.2f}")

        # Train the best model on the full training data
        self.model = best_model
        self.is_trained = True
        self.model.fit(X_train_bow, y_train)

        # Predict on the test set
        y_pred = self.model.predict(X_test_bow)

        # Evaluate the best model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))








    def predict(self, X):
        X_bow = self.vectorizer.transform(X)
        return self.model.predict(X_bow)

