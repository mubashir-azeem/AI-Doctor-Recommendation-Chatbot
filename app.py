from flask import Flask, render_template, request
import pandas as pd
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

#using panadas so can handle my dataset(csv) files.
#Tfidvectorizer to convert text to number and assign the importance to each word.
#sklearn used here for preprocessing like label encoding (text to numbers).
#using flask for my webapp and render to display and connect with my html page and request to send and take input of symtom from the web page form.

#added nltk for basic text cleaning like removing stopwords and symbols so input becomes clean before going into model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#added spacy for advanced nlp like extracting city automatically instead of manual if else
import spacy

#added tensorflow to build neural network model instead of logistic regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nltk.download('punkt')
nltk.download('stopwords')

#loading spacy model for nlp processing
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)


# added dataset for training model (symptoms)
symptoms_data = pd.read_csv('symptoms.csv')


# taking symptom column as input and specialization as output
X = symptoms_data['Symptom']
y = symptoms_data['Specialization']


# encoding labels (text → numbers so model can understand output)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# converted text into numbers using tfidf so model can understand it
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)


# building neural network model using tensorflow instead of logistic regression
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_vectorized.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(set(y_encoded)), activation='softmax'))

# compiling model (how it learns)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# training model
model.fit(X_vectorized.toarray(), y_encoded, epochs=10)


# cleaning user input before giving it to model
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


# using spacy to extract city automatically instead of manual checking
def extract_city(text):
    text_lower = text.lower()
    
    # fallback manual check (important)
    if "karachi" in text_lower:
        return "Karachi"
    elif "lahore" in text_lower:
        return "Lahore"
    elif "islamabad" in text_lower:
        return "Islamabad"
    
    # spacy detection
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.capitalize()
    
    return None


# using database
def get_doctors(specialization, city):
    
    conn = sqlite3.connect('doctors.db')
    cursor = conn.cursor()
    
    if city:
        cursor.execute("SELECT * FROM doctors WHERE specialization=? AND city=?", (specialization, city))
    else:
        cursor.execute("SELECT * FROM doctors WHERE specialization=?", (specialization,))
    
    results = cursor.fetchall()
    conn.close()
    
    return results


# main prediction logic

# to perform prediction and filter doctors and also handle random inputs properly

def recommend_doctor(user_input):
    
    clean_text = preprocess_text(user_input)
    
    input_vector = vectorizer.transform([clean_text])
    
    # if no meaningful words found (random text)
    if input_vector.nnz == 0:
        return "Sorry, I could not understand your symptoms. Please try again."
    
    # predicting using tensorflow model
    prediction = model.predict(input_vector.toarray())
    
    confidence = prediction.max()  # checking how confident model is
    predicted_class = prediction.argmax()
    
    # if confidence is too low means unclear input
    if confidence < 0.25:
        return "Sorry, I could not understand your symptoms. Please try again."
    
    # converting back number → specialization name
    predicted_specialization = encoder.inverse_transform([predicted_class])[0]
    
    # getting city from spacy + fallback manual check
    city = extract_city(user_input)
    
    doctors_list = get_doctors(predicted_specialization, city)
    
    # if no doctor found
    if not doctors_list:
        return f"No doctors found for {predicted_specialization} in {city}"
    
    # preparing final response
    result = f"Specialist: {predicted_specialization}\n\nDoctors:\n"
    
    for row in doctors_list:
        result += f"- {row[0]} ({row[2]}, {row[3]})\n"
    
    return result

# route
@app.route('/', methods=['GET', 'POST'])
def home():
    
    chat = []
    
    if request.method == 'POST':
        user_input = request.form['symptoms']
        
        if user_input.strip():
            bot_response = recommend_doctor(user_input)
            chat.append(("You", user_input))
            chat.append(("Bot", bot_response))
        else:
            chat.append(("Bot", "Please enter symptoms"))
    
    return render_template('index.html', chat=chat)


# run app
if __name__ == '__main__':
    app.run(debug=True)