# ---------------------------- PREPROCESSING FUNCTION ----------------------------
import os
import logging
import pandas as pd
import ssl
import certifi
import re
from unicodedata import normalize
from bs4 import BeautifulSoup
import nltk
import spacy
from symspellpy import SymSpell, Verbosity

# ---------------- Setup SSL ------------------
ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())

# ---------------- Download Resources ----------------
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------- Logger Setup ----------------
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------- Path Setup ----------------
# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data paths
raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw")
interim_data_path = os.path.join(PROJECT_ROOT, "data", "interim")
dict_path = os.path.join(PROJECT_ROOT, "data", "dictionaries", "frequency_dictionary_en_82_765.txt")

# ---------------- Load SymSpell Dictionary ----------------
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

if not os.path.exists(dict_path):
    logger.error(f"SymSpell dictionary not found at {dict_path}")
    raise FileNotFoundError(f"Dictionary not found at {dict_path}")

sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

# ---------------- Contractions ----------------
# store contractions set as a variable
updated_contraction_words = {
    "I'm": 'I am', "I'm'a": 'I am about to', "I'm'o": 'I am going to', "I've": 'I have', "I'll": 'I will',
    "I'll've": 'I will have', "I'd": 'I would', "I'd've": 'I would have', 'Whatcha': 'What are you', "amn't": 'am not', "ain't": 'are not',
    "aren't": 'are not', "'cause": 'because', "can't": 'cannot', "can't've": 'cannot have', "could've": 'could have', "couldn't": 'could not',
    "couldn't've": 'could not have', "daren't": 'dare not', "daresn't": 'dare not', "dasn't": 'dare not', "didn't": 'did not', 'didn’t': 'did not',
    "don't": 'do not', 'don’t': 'do not', "doesn't": 'does not', "e'er": 'ever', "everyone's": 'everyone is', 'finna': 'fixing to',
    'gimme': 'give me', "gon't": 'go not', 'gonna': 'going to', 'gotta': 'got to',  "hadn't": 'had not', "hadn't've": 'had not have',
    "hasn't": 'has not', "haven't": 'have not', "he've": 'he have', "he's": 'he is', "he'll": 'he will', "he'll've": 'he will have',
    "he'd": 'he would', "he'd've": 'he would have', "here's": 'here is', "how're": 'how are', "how'd": 'how did', "how'd'y": 'how do you',
    "how's": 'how is', "how'll": 'how will', "isn't": 'is not', "it's": 'it is', "'tis": 'it is', "'twas": 'it was', "it'll": 'it will',
    "it'll've": 'it will have', "it'd": 'it would', "it'd've": 'it would have', 'kinda': 'kind of', "let's": 'let us', 'luv': 'love',
    "ma'am": 'madam', "may've": 'may have', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have',
    "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have',
    "ne'er": 'never', "o'": 'of', "o'clock": 'of the clock', "ol'": 'old', "oughtn't": 'ought not', "oughtn't've": 'ought not have',
    "o'er": 'over', "shan't": 'shall not', "sha'n't": 'shall not', "shalln't": 'shall not', "shan't've": 'shall not have', "she's": 'she is',
    "she'll": 'she will', "she'd": 'she would', "she'd've": 'she would have', "should've": 'should have', "shouldn't": 'should not',
    "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so is', "somebody's": 'somebody is', "someone's": 'someone is',
    "something's": 'something is', 'sux': 'sucks', "that're": 'that are', "that's": 'that is', "that'll": 'that will', "that'd": 'that would',
    "that'd've": 'that would have', "'em": 'them', "there're": 'there are', "there's": 'there is', "there'll": 'there will', "there'd": 'there would',
    "there'd've": 'there would have', "these're": 'these are', "they're": 'they are', "they've": 'they have', "they'll": 'they will', "they'll've": 'they will have',
    "they'd": 'they would', "they'd've": 'they would have', "this's": 'this is', "this'll": 'this will', "this'd": 'this would', "those're": 'those are',
    "to've": 'to have', 'wanna': 'want to', "wasn't": 'was not', "we're": 'we are', "we've": 'we have', "we'll": 'we will', "we'll've": 'we will have',
    "we'd": 'we would', "we'd've": 'we would have', "weren't": 'were not', "what're": 'what are', "what'd": 'what did', "what've": 'what have',
    "what's": 'what is', "what'll": 'what will', "what'll've": 'what will have', "when've": 'when have', "when's": 'when is',"where're": 'where are',
    "where'd": 'where did', "where've": 'where have', "where's": 'where is', "which's": 'which is', "who're": 'who are', "who've": 'who have',
    "who's": 'who is', "who'll": 'who will', "who'll've": 'who will have', "who'd": 'who would', "who'd've": 'who would have', "why're": 'why are',
    "why'd": 'why did', "why've": 'why have', "why's": 'why is', "will've": 'will have', "won't": 'will not', "won't've": 'will not have',
    "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all're": 'you all are',
    "y'all've": 'you all have', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "you're": 'you are', "you've": 'you have',
    "you'll've": 'you shall have', "you'll": 'you will', "you'd": 'you would', "you'd've": 'you would have', 'to cause': 'to cause',
    'will cause': 'will cause', 'should cause': 'should cause', 'would cause': 'would cause', 'can cause': 'can cause', 'could cause': 'could cause',
    'must cause': 'must cause', 'might cause': 'might cause', 'shall cause': 'shall cause', 'may cause': 'may cause', 'jan.': 'january',
    'feb.': 'february', 'mar.': 'march', 'apr.': 'april', 'jun.': 'june', 'jul.': 'july', 'aug.': 'august', 'sep.': 'september', 'oct.': 'october',
    'nov.': 'november', 'dec.': 'december', 'I’m': 'I am', "I’m’a": 'I am about to', "I’m’o": 'I am going to', "I’ve": 'I have', 'I’ll': 'I will',
    "I’ll’ve": 'I will have', "I’d": 'I would', "I’d’ve": 'I would have', "amn’t": 'am not', "ain’t": 'are not', "aren’t": 'are not', "’cause": 'because',
    "can’t": 'cannot', "can’t’ve": 'cannot have', "could’ve": 'could have', "couldn’t": 'could not', "couldn’t’ve": 'could not have', "daren’t": 'dare not',
    "daresn’t": 'dare not', "dasn’t": 'dare not', "doesn’t": 'does not', "e’er": 'ever', 'everyone’s': 'everyone is', "gon’t": 'go not',
    "hadn’t": 'had not', "hadn’t’ve": 'had not have', "hasn’t": 'has not', "haven’t": 'have not', "he’ve": 'he have', "he’s": 'he is',
    "he’ll": 'he will', "he’ll’ve": 'he will have', "he’d": 'he would', "he’d’ve": 'he would have', "here’s": 'here is', "how’re": 'how are',
    "how’d": 'how did', "how’d’y": 'how do you', "how’s": 'how is', "how’ll": 'how will', 'isn’t': 'is not', "it’s": 'it is', "’tis": 'it is',
    "’twas": 'it was', "it’ll": 'it will', "it’ll’ve": 'it will have', "it’d": 'it would', "it’d’ve": 'it would have', "let’s": 'let us', "ma’am": 'madam',
    "may’ve": 'may have', "mayn’t": 'may not', "might’ve": 'might have', "mightn’t": 'might not', "mightn’t’ve": 'might not have', "must’ve": 'must have',
    "mustn’t": 'must not', "mustn’t’ve": 'must not have', "needn’t": 'need not', "needn’t’ve": 'need not have', "ne’er": 'never', "o’": 'of',
    "o’clock": 'of the clock', "ol’": 'old', "oughtn’t": 'ought not', "oughtn’t’ve": 'ought not have', "o’er": 'over', "shan’t": 'shall not',
    "sha’n’t": 'shall not', "shalln’t": 'shall not', "shan’t’ve": 'shall not have', "she’s": 'she is', "she’ll": 'she will', "she’d": 'she would',
    "she’d’ve": 'she would have', "should’ve": 'should have', "shouldn’t": 'should not', "shouldn’t’ve": 'should not have', "so’ve": 'so have',
    "so’s": 'so is', "somebody’s": 'somebody is', "someone’s": 'someone is', "something’s": 'something is', "that’re": 'that are', "that’s": 'that is',
    "that’ll": 'that will', "that’d": 'that would', "that’d’ve": 'that would have', "’em": 'them', "there’re": 'there are', "there’s": 'there is',
    "there’ll": 'there will', "there’d": 'there would', "there’d’ve": 'there would have', "these’re": 'these are', "they’re": 'they are', "they’ve": 'they have',
    "they’ll": 'they will', "they’ll’ve": 'they will have', "they’d": 'they would', "they’d’ve": 'they would have', "this’s": 'this is', "this’ll": 'this will',
    "this’d": 'this would', "those’re": 'those are', "to’ve": 'to have', "wasn’t": 'was not', "we’re": 'we are', 'we’ve': 'we have', "we’ll": 'we will',
    "we’ll’ve": 'we will have', "we’d": 'we would', "we’d’ve": 'we would have', "weren’t": 'were not', "what’re": 'what are', "what’d": 'what did',
    "what’ve": 'what have', "what’s": 'what is', "what’ll": 'what will', "what’ll’ve": 'what will have', "when’ve": 'when have', 'when’s': 'when is',
    "where’re": 'where are', "where’d": 'where did', "where’ve": 'where have', "where’s": 'where is', "which’s": 'which is', "who’re": 'who are',
    "who’ve": 'who have', "who’s": 'who is', "who’ll": 'who will', "who’ll’ve": 'who will have', "who’d": 'who would', "who’d’ve": 'who would have',
    "why’re": 'why are', "why’d": 'why did', "why’ve": 'why have', "why’s": 'why is', "will’ve": 'will have', "won’t": 'will not', "won’t’ve": 'will not have',
    "would’ve": 'would have', "wouldn’t": 'would not', "wouldn’t’ve": 'would not have', "y’all": 'you all', "y’all’re": 'you all are',
    "y’all’ve": 'you all have', "y’all’d": 'you all would', "y’all’d’ve": 'you all would have', "you’re": 'you are', "you’ve": 'you have',
    "you’ll’ve": 'you shall have', "you’ll": "you will", "you’d": 'you would', "you’d’ve": 'you would have', "i'm": 'i am', "i've": 'i have', 'u': 'you',
    '2nite': 'tonight', 'wil': 'will', 'bday': 'birthday', 'hai': 'hi', 'bihday': 'birthday', 'ur': 'your'}

# ---------------- Preprocessing Function ----------------
# Create a function to preprocess youTube comments
def preprocess_comment(comment):
    try:
        # Expand contractions
        for key, value in updated_contraction_words.items():
            pattern = r'\b' + re.escape(key) + r'\b'
            comment = re.sub(pattern, value, comment)

        # Remove URLs and emails
        comment = re.sub(r"http's'?://\S+|www\S+|\S+\.com|\S+@\S+", "", comment)

        # Remove mentions and RT
        comment = re.sub(r'\bRT\b|@\S+', '', comment)

        # Remove HTML tags
        comment = BeautifulSoup(comment, "html.parser").get_text()

        # Normalize accented characters
        comment = normalize('NFKD', comment)

        # Remove special characters
        comment = re.sub(r"[^a-zA-Z0-9\s]", '', comment)

        # Remove numeric tokens
        comment = re.sub(r'\b\d+\b', '', comment)

        # Convert to lowercase
        comment = comment.lower()

        # Spelling correction
        words = comment.split()
        corrected_words = []
        for word in words:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            corrected_word = suggestions[0].term if suggestions else word
            corrected_words.append(corrected_word)

        comment = ' '.join(corrected_words)

        # Lemmatization with SpaCy
        doc = nlp(comment)
        lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

        cleaned_comment = ' '.join(lemmatized_words).strip()

        return cleaned_comment

    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


# ---------------- Normalize DataFrame ----------------
def normalize_text(df):
    try:
        df["clean_comment"] = df["clean_comment"].astype(str).apply(preprocess_comment)
        logger.debug("Text normalization completed.")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# ---------------- Save Processed Data ----------------
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        # interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True) # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


# ---------------- Main Function ----------------
def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Load data
        train_data = pd.read_csv(os.path.join(raw_data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(raw_data_path, "test.csv"))
        logger.debug("Data loaded successfully.")

        # Process data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()