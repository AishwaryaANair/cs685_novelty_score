from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load model
model = SentenceTransformer('intfloat/e5-large')

# File paths
llm_files = {
    "gpt": "gpt_data - Sheet1.csv",
    "mistralai": "mistralai - Sheet1.csv",
    "llama": "Llama-4-Scout-17Bb-16E - Sheet1.csv",
    "allenai": "allenai - Sheet1.csv"
}
human_path = "final_human_data - Sheet1.csv"

# Load and encode human text
human_df = pd.read_csv(human_path)
human_texts = human_df['text'].astype(str).tolist()
human_emb = model.encode(human_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
np.save("human_embeddings_e5.npy", human_emb)

# Loop through each LLM file
for name, path in llm_files.items():
    df = pd.read_csv(path)
    texts = df['text'].astype(str).tolist()
    emb = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    np.save(f"{name}_embeddings_e5.npy", emb)
    print(f"{name} embeddings saved.")
