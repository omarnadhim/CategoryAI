import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint

class ProductCategoryClassifier:
    def __init__(self):
        # Load and preprocess data
        dtype_dict = {"Product Name": str, "Category": str}
        self.df = pd.read_csv("product_data.csv", dtype=dtype_dict, low_memory=False)
        self.df.dropna(subset=["Product Name", "Category"], inplace=True)
        
        # Load GPT-2 tokenizer and model
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Feature engineering using TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Adjust max_features as needed
        self.X_tfidf = self.tfidf_vectorizer.fit_transform(self.df["Product Name"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_tfidf, self.df["Category"], test_size=0.4, random_state=25
        )
        
        # Model initialization
        self.model_nn = MLPClassifier(
            hidden_layer_sizes=(1024,),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text if text else ""

    def gpt2_vectorize(self, text):
        processed_text = self.preprocess_text(text)
        inputs = self.gpt2_tokenizer(processed_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.gpt2_model(**inputs)
        embeddings = outputs[0].mean(dim=1).squeeze(0).numpy()
        return embeddings

    def train_model(self):
        self.model_nn.fit(self.X_train, self.y_train)

    def hyperparameter_tuning(self):
        nn_params = {
            "hidden_layer_sizes": [(256,), (512,), (1024,)],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "activation": ["relu", "tanh"],
        }
        nn_grid = GridSearchCV(estimator=self.model_nn, param_grid=nn_params, cv=5)
        nn_grid.fit(self.X_train, self.y_train)
        self.model_nn_best = nn_grid.best_estimator_
        print("Best Neural Network Parameters:", nn_grid.best_params_)

    def evaluate_model(self):
        y_pred = self.model_nn_best.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Validation Set Accuracy: {accuracy:.4f}")
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure()
        self.plot_confusion_matrix(
            conf_matrix, classes=np.unique(self.y_train), title="Confusion Matrix"
        )
        # Classification report
        report = classification_report(self.y_test, y_pred)
        print(f"Classification Report:\n{report}")
        print("---------------------")

    def plot_confusion_matrix(self, conf_matrix, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
        plt.imshow(conf_matrix, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

    def save_model(self):
        joblib.dump(self.model_nn_best, "best_neural_network_model.joblib")

    def load_model(self):
        self.model_nn_best = joblib.load("best_neural_network_model.joblib")

    def suggest_category(self, product_name):
        product_tfidf = self.tfidf_vectorizer.transform([product_name])
        predicted_category = self.model_nn_best.predict(product_tfidf)[0]
        return predicted_category

    def suggest_category_using_description(self, long_description):
        processed_description = self.preprocess_text(long_description)
        description_vector = self.gpt2_vectorize(processed_description)
        predicted_category = self.model_nn_best.predict([description_vector])[0]
        return predicted_category

    def suggest_category_using_product_name_and_long_description(self, product_name, long_description=None):
        product_tfidf = self.tfidf_vectorizer.transform([product_name])
        predicted_category = self.model_nn_best.predict(product_tfidf)[0]

        if predicted_category == "Books" and long_description:
            suggested_category = self.suggest_category_using_description(long_description)
        else:
            suggested_category = predicted_category

        return suggested_category
    
    def feedback(self, input_product, suggested_category):
        user_feedback = input("Is the suggested category correct? (yes/no): ").lower()
        if user_feedback == "yes":
            print("Great! The model made the correct suggestion.")
        elif user_feedback == "no":
            correct_category = input("Please provide the correct category: ")
            # Update training data with the corrected category
            corrected_row = {"Product Name": input_product, "Category": correct_category}
            self.df = self.df.append(corrected_row, ignore_index=True)
            # Retrain the model with the updated training data
            self.X_tfidf = self.tfidf_vectorizer.fit_transform(self.df["Product Name"])
            self.X_train, _, self.y_train, _ = train_test_split(
                self.X_tfidf, self.df["Category"], test_size=0.2, random_state=45
            )
            self.model_nn_best.fit(self.X_train, self.y_train)
            # Save the updated model
            self.save_model()
            print("Model updated with your feedback. Thank you!")

# Usage:

classifier = ProductCategoryClassifier()
#classifier.train_model()
#classifier.hyperparameter_tuning()
#classifier.evaluate_model()
#classifier.save_model()
classifier.load_model()

product_name = "NIVEA Antiperspirant Spray for Women, Pearl & Beauty Pearl Extracts, 2x150ml"

suggested_category = classifier.suggest_category(product_name)
print(suggested_category)
classifier.feedback(product_name, suggested_category)



