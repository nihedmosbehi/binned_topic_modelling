from bertopic import BERTopic
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import umap

# Sample DataFrame with a "text" column
data = {
    "text": [
        "This is the first document about sports.",
        "The second document is about politics and elections.",
        "Sports and athletes are the main topic of the third document.",
        "Fourth document discusses economics and finance.",
        "The last document is again about sports and players."
    ]
}

# Convert DataFrame to the 'documents' list
documents = data["text"]

# Tokenization and vectorization of documents
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Reduce the dimensionality of the sparse matrix
svd = TruncatedSVD(n_components=5)
X_svd = svd.fit_transform(X)

# Convert the sparse matrix to a dense matrix
X_dense = X_svd

# Create a UMAP model and fit it on the dense matrix
umap_model = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine')
umap_embeddings = umap_model.fit_transform(X_dense)

# Create a BERTopic model and fit it on the documents
model = BERTopic()
topics, _ = model.fit_transform(documents)

# Get the topic representation for each document
document_topics = model.get_doctopic()

# Create a DataFrame to display the results
df = pd.DataFrame({"Document": documents, "Topic": topics, "Topic Representation": document_topics})

# Display the DataFrame
print(df)
