import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the IMDb dataset
df = pd.read_csv('IMDB_dataset.csv')  # Replace with the path to your dataset

# Filter reviews for 'V for Vendetta'
v_for_vendetta_reviews = df[df['movie_title'].str.contains('V for Vendetta', case=False, na=False)]

# Data Cleaning and Preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

v_for_vendetta_reviews['cleaned_review'] = v_for_vendetta_reviews['review'].apply(clean_text)

# Sentiment Distribution Plot
v_for_vendetta_reviews['sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Sentiment Distribution for V for Vendetta Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
plt.show()

# Word Cloud for Positive and Negative Reviews
positive_reviews = v_for_vendetta_reviews[v_for_vendetta_reviews['sentiment'] == 1]['cleaned_review']
negative_reviews = v_for_vendetta_reviews[v_for_vendetta_reviews['sentiment'] == 0]['cleaned_review']

positive_text = ' '.join(positive_reviews)
negative_text = ' '.join(negative_reviews)

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Word Cloud for Positive Reviews')
plt.axis('off')
plt.show()

wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Word Cloud for Negative Reviews')
plt.axis('off')
plt.show()

# Train a Sentiment Classification Model
X = CountVectorizer().fit_transform(v_for_vendetta_reviews['cleaned_review'])
y = v_for_vendetta_reviews['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
