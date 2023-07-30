from bertopic import BERTopic
from sklearn.metrics import jaccard_score
from binning_with_bert import getTopicWithBins
from sklearn.metrics.pairwise import cosine_similarity

from embeding_model import getModel

similarity_threshold = 0.02
# Sample list of bins topics and keywords
# bins_topics_keywords = getTopicWithBins()
# print(bins_topics_keywords)
bins_topics_keywords = getTopicWithBins()
# bins_topics_keywords = [[[('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]], [[('attracting', 0.18694434296164386), ('careers', 0.18694434296164386), ('traveling', 0.18694434296164386), ('travel', 0.18694434296164386), ('thrill', 0.18694434296164386)], [('gaining', 0.11770569742029428), ('retreats', 0.11770569742029428), ('prioritize', 0.11770569742029428), ('positivity', 0.11770569742029428), ('popularity', 0.11770569742029428)], [('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]], [[('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]], [[('gaining', 0.11770569742029428), ('retreats', 0.11770569742029428), ('prioritize', 0.11770569742029428), ('positivity', 0.11770569742029428), ('popularity', 0.11770569742029428)], [('attracting', 0.18694434296164386), ('careers', 0.18694434296164386), ('traveling', 0.18694434296164386), ('travel', 0.18694434296164386), ('thrill', 0.18694434296164386)], [('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]], [[('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)], [('gaining', 0.11770569742029428), ('retreats', 0.11770569742029428), ('prioritize', 0.11770569742029428), ('positivity', 0.11770569742029428), ('popularity', 0.11770569742029428)], [('attracting', 0.18694434296164386), ('careers', 0.18694434296164386), ('traveling', 0.18694434296164386), ('travel', 0.18694434296164386), ('thrill', 0.18694434296164386)]], [[('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]], [[('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)], [('gaining', 0.11770569742029428), ('retreats', 0.11770569742029428), ('prioritize', 0.11770569742029428), ('positivity', 0.11770569742029428), ('popularity', 0.11770569742029428)], [('attracting', 0.18694434296164386), ('careers', 0.18694434296164386), ('traveling', 0.18694434296164386), ('travel', 0.18694434296164386), ('thrill', 0.18694434296164386)]], [[('gaining', 0.11770569742029428), ('retreats', 0.11770569742029428), ('prioritize', 0.11770569742029428), ('positivity', 0.11770569742029428), ('popularity', 0.11770569742029428)], [('attracting', 0.18694434296164386), ('careers', 0.18694434296164386), ('traveling', 0.18694434296164386), ('travel', 0.18694434296164386), ('thrill', 0.18694434296164386)], [('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]], [[('gaining', 0.11770569742029428), ('retreats', 0.11770569742029428), ('prioritize', 0.11770569742029428), ('positivity', 0.11770569742029428), ('popularity', 0.11770569742029428)], [('attracting', 0.18694434296164386), ('careers', 0.18694434296164386), ('traveling', 0.18694434296164386), ('travel', 0.18694434296164386), ('thrill', 0.18694434296164386)], [('industry', 0.17275873994826982), ('entertainment', 0.12712215321391784), ('educational', 0.12712215321391784), ('online', 0.12712215321391784), ('learning', 0.12712215321391784)]]]
print(len(bins_topics_keywords))





# Step 2: Define a function to calculate the vector representation of a list of keywords (topic)
def get_vector_representation(keyword_list):
    model = getModel()
    vectors = [model[word[0]] for word in keyword_list if word[0] in model]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None



# Step 3: Calculate the cosine similarity between two lists of keywords (two topics)
def calculate_similarity(keyword_list1, keyword_list2):
    vector1 = get_vector_representation(keyword_list1)
    vector2 = get_vector_representation(keyword_list2)

    if vector1 is not None and vector2 is not None:
        similarity_score = cosine_similarity([vector1], [vector2])[0][0]
        return similarity_score
    else:
        # Return a low similarity score if one of the lists is empty or no valid embeddings are found
        return 0.0

def calculate_similarity(bin1, bin2):
    # that counts the number of common keywords in the topics of the two bins
    keywords_bin1 = set(keyword for topic in bin1 for keyword, _ in topic)
    keywords_bin2 = set(keyword for topic in bin2 for keyword, _ in topic)
    common_keywords = len(keywords_bin1.intersection(keywords_bin2))
    total_keywords = len(keywords_bin1.union(keywords_bin2))
    similarity_score = common_keywords / total_keywords
    return similarity_score


def merge_topics(topic_list1, topic_list2):
    # Concatenate the two lists
    concatenated_list = topic_list1 + topic_list2

    # Flatten the list of lists into a single list
    flattened_list = [item for sublist in concatenated_list for item in sublist]

    # Convert the flattened list to a set
    result_set = set(flattened_list)

    # Convert the set back to a list and return it
    result_list = list(result_set)
    return result_list

def merge_bins(bins, similarity_threshold):
    merged_indices = set()
    merged_bins = []

    for i, bin1 in enumerate(bins):
        if i not in merged_indices:
            for j, bin2 in enumerate(bins[i + 1:], start=i + 1):
                similarity_score = calculate_similarity(bin1, bin2)
                print(similarity_score)
                if similarity_score >= similarity_threshold:
                    merged_bin = merge_topics(bin1, bin2)
                    merged_bins.append(merged_bin)
                    merged_indices.add(i)
                    merged_indices.add(j)
                    break
            else:
                merged_bins.append(bin1)
    return merged_bins

# print(len(merge_bins(bins_topics_keywords, 0.8)))
print(len(merge_bins(bins_topics_keywords, 0.9)))

