from bertopic import BERTopic
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
stop_words = stopwords.words('english')


from data import getData



# Define the time-based bins (you can adjust this based on your data)
bins = {
    "2023-07-01": ("2023-07-01 00:00:00", "2023-07-01 23:59:59"),
    "2023-07-02": ("2023-07-02 00:00:00", "2023-07-02 23:59:59"),
    "2023-07-03": ("2023-07-03 00:00:00", "2023-07-03 23:59:59"),
    "2023-07-04": ("2023-07-04 00:00:00", "2023-07-04 23:59:59"),
}


model = BERTopic(verbose=True,min_topic_size=3)
documents = getData()['text'].tolist()
print(documents)
topics, _ = model.fit_transform(documents)
print(topics)
df2 = pd.DataFrame({"Document": documents, "Topic": topics})
print(model.get_topic(-1))
print(df2.head(10))


sim_matrix = cosine_similarity(model.c_tf_idf_)
df = pd.DataFrame(sim_matrix, columns=model.topic_labels_.values(), index=model.topic_labels_.values())
print(sim_matrix)