import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
stop_words = stopwords.words('english')
from data import getData

def clean_text(x):
  x = str(x)
  x = x.lower()
  x = re.sub(r'#[A-Za-z0-9]*', ' ', x)
  x = re.sub(r'https*://.*', ' ', x)
  x = re.sub(r'@[A-Za-z0-9]+', ' ', x)
  tokens = word_tokenize(x)
  x = ' '.join([w for w in tokens if not w.lower() in stop_words])
  x = re.sub(r'[%s]' % re.escape('!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~“…”’'), ' ', x)
  x = re.sub(r'\d+', ' ', x)
  x = re.sub(r'\n+', ' ', x)
  x = re.sub(r'\s{2,}', ' ', x)
  return x


# Function to remove pronouns and adverbs
def remove_pronouns_adverbs(text):
    # Define pronoun, determiner, and adverb POS tags
    pronoun_tags = ['PRP', 'PRP$', 'WP', 'WP$']
    determiner_tags = ['DT', 'PDT', 'WDT']
    adverb_tags = ['RB', 'RBR', 'RBS']

    for word in text.split():
        # Get the POS tag of the current word
        pos_tag = nltk.pos_tag([word])[0][1]

        # Remove the word if it is a pronoun, determiner, or adverb
        if pos_tag in pronoun_tags or pos_tag in determiner_tags or pos_tag in adverb_tags:
            text = text.replace(word, '')

    return text

# Assuming you have a DataFrame named 'getData' with a 'text' column


dataframe = getData()
dataframe['clean_text'] = dataframe["text"].apply(clean_text)
dataframe['clean_text'] = dataframe["text"].apply(remove_pronouns_adverbs)
print(dataframe['clean_text'].head())