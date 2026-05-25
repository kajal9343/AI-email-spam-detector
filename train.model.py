import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "message":[
        "Win lottery now",
        "Claim free gift",
        "Free coupons available",
        "Meeting tomorrow",
        "Project submission",
        "Team discussion"
    ],

    "label":[
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham"
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])

model = MultinomialNB()

model.fit(X,df["label"])

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("Model Saved")