import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network

# Configuration
llm_models = {
    "gpt": "gpt_with_creativity.csv",
    "mistralai": "mistralai_with_creativity.csv",
    "llama": "llama_with_creativity.csv",
    "allenai": "allenai_with_creativity.csv"
}

human_file = "final_human_data - Sheet1.csv"
human_texts = pd.read_csv(human_file)['text'].astype(str).tolist()

# Parameters
node_limit = 200  # max nodes per group

for model_name, llm_path in llm_models.items():
    print(f"Processing {model_name}...")

    llm_df = pd.read_csv(llm_path)
    llm_texts = llm_df['text'].astype(str).tolist()
    
    if "creativity_score" not in llm_df.columns:
        raise ValueError(f"{model_name} CSV must include creativity_score column")

    G = nx.DiGraph()

    # Add human nodes (no edges)
    for i in range(min(len(human_texts), node_limit)):
        G.add_node(
            f"HUMAN_{i}",
            label=f"Human {i}",
            group="human",
            title="Human-written text",
            shape="dot",
            color="#1f78b4"
        )

    # Add LLM nodes with creativity score in tooltip
    for i in range(min(len(llm_texts), node_limit)):
        score = llm_df.iloc[i]["creativity_score"]
        color = f"rgba({int((1-score)*255)}, {int(score*255)}, 100, 1)"  # red-to-green scale
        G.add_node(
            f"LLM_{i}",
            label=f"{model_name.upper()} {i}",
            group="llm",
            title=f"Creativity Score: {score:.3f}",
            shape="dot",
            color=color,
            size=15 + 10 * score  # node size based on creativity
        )

    # Optional: connect human nodes to LLMs randomly just to preserve layout force (NOT semantic)
    for i in range(min(len(llm_texts), node_limit)):
        if i < len(human_texts):
            G.add_edge(f"HUMAN_{i}", f"LLM_{i}", value=0.1)

    # Visualize using PyVis
    net = Network(height="800px", width="100%", bgcolor="#111111", font_color="white", directed=True)
    net.from_nx(G)
    net.force_atlas_2based(gravity=-40, spring_length=120)
    net.save_graph(f"creativity_graph_{model_name}.html")

    print(f"✅ Saved: creativity_graph_{model_name}.html")
