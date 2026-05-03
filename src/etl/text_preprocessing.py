import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


def clean_email_text(text):
    if not isinstance(text, str):
        return ""
    
    # Chuẩn hóa chữ thường
    text = text.lower()
    
    # Loại bỏ HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Loại bỏ URL và Email
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Loại bỏ ký tự đặc biệt và số (giữ lại chữ cái)
    text = re.sub(r'[^a-z\s]', '', text)

    # Xử lý khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def remove_stopwords_and_lemmatize(text):
    stop_words = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)