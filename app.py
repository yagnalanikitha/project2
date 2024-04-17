from flask import Flask, request, render_template
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


def preprocess(text, language):
    if language == "en":
        doc = nlp_en(text)
    elif language == "fr":
        doc = nlp_fr(text)
    else:
        return "Unsupported language"
    
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


documents_en = [
    "Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence.",
    "Machine translation is the task of automatically converting source text in one language to text in another language."
]

documents_fr = [
    "Le traitement automatique du langage naturel (NLP) est un sous-domaine de la linguistique, de l'informatique, de l'ingénierie de l'information et de l'intelligence artificielle.",
    "La traduction automatique est la tâche de convertir automatiquement le texte source d'une langue en texte dans une autre langue."
]


processed_documents_en = [preprocess(doc, "en") for doc in documents_en]
processed_documents_fr = [preprocess(doc, "fr") for doc in documents_fr]


vectorizer_en = TfidfVectorizer()
tfidf_matrix_en = vectorizer_en.fit_transform(processed_documents_en)

vectorizer_fr = TfidfVectorizer()
tfidf_matrix_fr = vectorizer_fr.fit_transform(processed_documents_fr)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
   
    query = request.form.get('query', '')
    language = request.form.get('language', 'en') 
    
    if query.strip() == "":
        return render_template('error.html', message="Query is empty")

  
    processed_query = preprocess(query, language)
    
   
    if language == "en":
        query_vector = vectorizer_en.transform([processed_query])
        similarities = cosine_similarity(tfidf_matrix_en, query_vector)
        top_documents_indices = similarities.argsort(axis=0)[-3:].flatten()
        top_documents = [documents_en[idx] for idx in top_documents_indices]
    elif language == "fr":
        query_vector = vectorizer_fr.transform([processed_query])
        similarities = cosine_similarity(tfidf_matrix_fr, query_vector)
        top_documents_indices = similarities.argsort(axis=0)[-3:].flatten()
        top_documents = [documents_fr[idx] for idx in top_documents_indices]
    else:
        return render_template('error.html', message="Unsupported language")
    
    return render_template('results.html', documents=top_documents)

if __name__ == '__main__':
    app.run(debug=True)






