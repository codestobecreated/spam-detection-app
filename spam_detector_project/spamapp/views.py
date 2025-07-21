from django.shortcuts import render
import os
import pickle

# Load the model and vectorizer
# BASE_DIR = path to: /home/code/Desktop/spam_project/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_path = os.path.join(BASE_DIR, "spam_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))
def index(request):
    prediction = ""
    if request.method == "POST":
        message = request.POST.get("message")
        if message:
            msg_vector = vectorizer.transform([message])
            result = model.predict(msg_vector)[0]
            prediction = "SPAM ❌" if result == 1 else "HAM ✅"
    return render(request, "index.html", {"prediction": prediction})
