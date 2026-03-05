import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import os

def visualize_clasde(memory_file: str):
    if not os.path.exists(memory_file):
        print(f"Error: {memory_file} not found.")
        return

    with open(memory_file, "r") as f:
        data = json.load(f)

    dataset = data.get("dataset", [])
    graph_data = data.get("graph", {})
    
    # 1. Reward Progress Plot
    rewards = [d["target_value"] for d in dataset]
    best_rewards = np.maximum.accumulate(rewards)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'o-', label='Observed Reward', alpha=0.5)
    plt.plot(best_rewards, 's-', label='Best Reward', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("CLASDE Optimization Progress")
    plt.legend()
    plt.grid(True)

    # 2. Graph Visualization
    G = nx.DiGraph()
    for node in graph_data.get("nodes", []):
        G.add_node(node["id"], reward=node.get("reward"))
    
    for edge in graph_data.get("edges", []):
        G.add_edge(edge["source"], edge["target"])

    plt.subplot(1, 2, 2)
    # Highlight nodes with high rewards
    node_rewards = [G.nodes[n].get('reward', -10) for n in G.nodes]
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_rewards, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='gray', alpha=0.3)
    plt.title("Exploration Graph")
    plt.axis('off')

    plt.tight_layout()
    output_png = os.path.join(os.path.dirname(memory_file), "clasde_summary.png")
    plt.savefig(output_png)
    print(f"Optimization summary plot saved to {output_png}")

if __name__ == "__main__":
    file_path = "results/clasde_memory.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    visualize_clasde(file_path)
