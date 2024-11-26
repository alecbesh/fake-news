import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load the data
data = pd.read_csv("fake_or_real_news.csv")
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)

x, y = data['text'], data['fake']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Train LinearSVC
clf = LinearSVC(dual=False)
clf.fit(x_train_vectorized, y_train)

# Evaluate model
accuracy = clf.score(x_test_vectorized, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save predictions for the entire dataset
data['prediction'] = clf.predict(vectorizer.transform(data['text']))
data['predicted_label'] = data['prediction'].apply(lambda x: "REAL" if x == 0 else "FAKE")
data.to_csv("predicted_results.csv", index=False)
print("Predictions saved to 'predicted_results.csv'.")

# Predict and output the result for the individual article
with open("article_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

vectorized_text = vectorizer.transform([text])
predicted_label = clf.predict(vectorized_text)[0]

# Save the prediction result for the individual article
with open("individual_prediction.txt", "w", encoding="utf-8") as f:
    f.write(f"The predicted label for the article is: {'REAL' if predicted_label == 0 else 'FAKE'}")

print(f"The predicted label for the article is: {'REAL' if predicted_label == 0 else 'FAKE'}")

