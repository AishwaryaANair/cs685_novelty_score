import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

# Define LLM files and paths
llm_files = {
    "gpt": "gpt_data - Sheet1.csv",
    "mistralai": "mistralai - Sheet1.csv",
    "llama": "Llama-4-Scout-17Bb-16E - Sheet1.csv",
    "allenai": "allenai - Sheet1.csv"
}

# Load human humor embeddings
human_embeddings = np.load("human_embeddings_e5.npy")

# Compute centroid of human embeddings
human_centroid = human_embeddings.mean(axis=0).reshape(1, -1)

# Process each LLM model
for model in llm_files:
    # Load the LLM-generated data and embeddings
    llm_df = pd.read_csv(llm_files[model])
    llm_embeddings = np.load(f"{model}_embeddings_e5.npy")

    # Compute cosine distance from human centroid
    distances = cosine_distances(llm_embeddings, human_centroid).flatten()

    # Normalize to [0, 1] — higher = more novel
    creativity_score = MinMaxScaler().fit_transform(distances.reshape(-1, 1)).flatten()

    # Add scores to DataFrame
    llm_df["creativity_score"] = creativity_score
    llm_df["fluency"] = llm_df["text"].apply(lambda x: len([s for s in x.split('.') if s.strip()]))

    # Save the updated file
    llm_df.to_csv(f"{model}_with_creativity.csv", index=False)
    print(f"{model} creativity scored (centroid-distance) and saved.")
