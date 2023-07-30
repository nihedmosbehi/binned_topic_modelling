import pandas as pd
import gensim.downloader as api
from torch import cosine_similarity
from data import getData
from clean_data import remove_pronouns_adverbs ,clean_text
from bertopic import BERTopic
from sklearn.metrics import jaccard_score
from embeding_model import getModel

# Specify the time period for each bin (in this case, 1 week)
time_period = "8D"
df = getData()
# Convert the timestamps to pandas datetime objects
df["timestamps"] = pd.to_datetime(df["timestamps"])

# Group the data into bins based on the specified time period
df["bin"] = pd.cut(df["timestamps"], bins=pd.date_range(start=df["timestamps"].min(), end=df["timestamps"].max(), freq=time_period))


bins = df["bin"].unique().tolist()

# Function to assign each document to a bin based on its timestamp
def assign_to_bin(timestamp):
    for bin in bins:
        if timestamp in bin:
            return bin

# Now, you can use the groupby method to split the data based on the bins
grouped_data = df.groupby("bin")

groups = []


# Iterate over the groups and append them to the list
for bin_name, group in grouped_data:
    groups.append(group)



documents = []
topics = []
topic_groups_per_bin = {}
bins_with_keywords = []
# Iterate over the groups and process documents with BerTopic
for i in range(len(groups)):
    # Get the text of all documents in the group
    group_documents = groups[i]["text"].tolist()
    # Preprocess the documents
    for j in range(len(group_documents)):
        group_documents[j] = clean_text(group_documents[j])
        group_documents[j] = remove_pronouns_adverbs(group_documents[j])
    # Append the documents to the list
    print(group_documents)
    documents.append(group_documents)

    # Create and fit the BERTopic model for each group
    model = BERTopic(verbose=True,min_topic_size=2)
    group_topics, _ = model.fit_transform(group_documents)
    topics.append(group_topics)

    # Group documents by topic within the same bin
    bin_name = groups[i]["bin"].iloc[0]  # Get the bin interval for this group
    if bin_name not in topic_groups_per_bin:
        topic_groups_per_bin[bin_name] = {}

    for doc_idx, topic_id in enumerate(group_topics):
        if topic_id not in topic_groups_per_bin[bin_name]:
            topic_groups_per_bin[bin_name][topic_id] = [group_documents[doc_idx]]
        else:
            topic_groups_per_bin[bin_name][topic_id].append(group_documents[doc_idx])

# Print the results
print("Documents:")
print(documents)
print("\nTopics:")
print(topics)
print("\nTopic Groups per Bin:")
i = 0
topics_with_bins = []
for bin_name, topic_groups in topic_groups_per_bin.items():
    print("Bin number :",i, bin_name)
    i+=1
    topics_keywords = []
    for topic_id, docs in topic_groups.items():
        keywords = model.get_topic(topic_id)[:5]  # Get the top 5 keywords for the topic
        topics_keywords.append(keywords)
        print("Topic ID:", topic_id, "| Keywords:", keywords)
        print("Documents:")
        for doc in docs:
            print("- ", doc)
        print("\n")
    topics_with_bins.append(topics_keywords)

def getTopicWithBins() :
    return topics_with_bins

print(topics_with_bins)


