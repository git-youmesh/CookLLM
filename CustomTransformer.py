from datasets import load_dataset
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
embeddings_model = SentenceTransformer("all-mpnet-base-v2")
batch_size = 64
def embed(batch):
    batch["embedding"] =embeddings_model.encode(batch["text"])
    return batch

N = 10_000
dataset = load_dataset("yelp_review_full", split=f"train[:{N}]")
dataset = dataset.map(embed, batch_size=batch_size,
batched=True)
dataset.set_format(type='numpy', columns=['embedding'],output_all_columns=True)
topic_model = BERTopic(n_gram_range=(1, 3))
topics, probs = topic_model.fit_transform(
dataset["text"],
np.array(dataset["embedding"]))
print(f"Number of topics: {len(topic_model.get_topics())}")
topic_sizes = topic_model.get_topic_freq()

print("Test")
