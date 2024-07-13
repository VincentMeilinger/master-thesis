from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


sentences = [
    "Huazhong Univ Sci & Technol, Dept Elec & Info, Wuhan 430074, Peoples R China",
    "Huazhong Univ Sci & Technol, Elect & Informat Engn Dept, Wuhan Natl Lab Optoelect, Wuhan 430074, Peoples R China",
    "Life Sciences Research Center, School of Life Sciences and Technology, Xidian University, Xi’an 710071, China",
    "Dept. of Comput. Sci., City Univ. of Hong Kong, Hong Kong, China|c|",
    "Hong Kong Polytech Univ, Dept Comp, Hong Kong, Hong Kong, Peoples R China",
    "George Mason University, Fairfax",
    "Medical Image Processing Group, Chinese Academy of Sciences, Beijing, China",
    "Department of Radiology and Biomedical Research Imaging Center, UNC, Chapel Hill, NC, United States",
    "School of the Computer Science, Tokyo University of Technology, Japan"
]

# Load the model
print("Loading model...")
model = SentenceTransformer('jordyvl/scibert_scivocab_uncased_sentence_transformer')

# Encode the sentences
print("Encoding sentences...")
embeddings = model.encode(sentences)

# Calculate cosine similarity
print("Calculating cosine similarity...")
sims = cosine_similarity(embeddings)

print("Done!")
print(sims)
# Calculate the pairs of sentences with similarity > 0.8
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        if sims[i, j] > 0.9:
            pairs.append((i, j, sims[i, j]))

for i, j, sim in pairs:
    print("____________________________________________________")
    print(f"S1: {sentences[i]}")
    print(f"S2: {sentences[j]}")
    print(f"Similarity: {sim}")
