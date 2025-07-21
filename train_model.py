import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')[['Category', 'Message']]
df.columns = ['label', 'message']

# Encode labels (spam = 1, ham = 0)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Vectorize messages
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model and vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
