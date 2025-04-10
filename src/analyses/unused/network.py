import networkx as nx
import numpy as np
import pandas as pd
from monkey_names import Zombies


def create_digraph_with_edge_weights(edge_weights):
    normalized_edge_weights = normalize_weights_min_max(edge_weights)
    print(normalized_edge_weights)
    G = nx.from_pandas_edgelist(normalized_edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                create_using=nx.DiGraph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_df = pd.DataFrame(adj_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    # print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(weights)
    return G, adj_matrix, weights


def create_digraph_with_top_70_percent_of_edge_weights(edge_weights):
    # Normalize the edge weights
    normalized_edge_weights = normalize_weights_min_max(edge_weights)
    print(normalized_edge_weights)

    # Create the directed graph
    G = nx.from_pandas_edgelist(normalized_edge_weights, source='Focal Name', target='Social Modifier',
                                edge_attr='weight', create_using=nx.DiGraph)

    # Get all edge weights
    all_weights = [d['weight'] for _, _, d in G.edges(data=True)]

    # Calculate the 20th percentile
    percentile = 0
    threshold = np.percentile(all_weights, percentile)
    print(f"Threshold for bottom {percentile}: {threshold}")

    # Remove edges with weights below the threshold
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)

    # Create the adjacency matrix after removing edges
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_df = pd.DataFrame(adj_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))

    # Extract remaining weights
    remaining_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(remaining_weights)

    return G, adj_matrix, remaining_weights


def normalize_weights_min_max(edgelist):
    normalized_edgelist = edgelist[['Focal Name', 'Social Modifier']]
    normalized_edgelist['weight'] = (edgelist['weight'] - edgelist['weight'].min()) / (edgelist['weight'].max() - edgelist['weight'].min())
    normalized_edgelist['weight'] = normalized_edgelist['weight'] * 8
    return normalized_edgelist


def create_graph_with_edge_weights(edge_weights):
    normalized_edge_weights = normalize_weights_min_max(edge_weights)
    G = nx.from_pandas_edgelist(normalized_edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                create_using=nx.Graph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_df = pd.DataFrame(adj_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    # print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(weights)
    return G, adj_matrix, weights

def create_graph_with_top_60_percent_of_edge_weights(edge_weights):
    normalized_edge_weights = normalize_weights_min_max(edge_weights)
    G = nx.from_pandas_edgelist(normalized_edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                create_using=nx.Graph)

    # Extract all edge weights
    weights = [d['weight'] for _, _, d in G.edges(data=True)]

    # Calculate the 40th percentile threshold
    threshold = np.percentile(weights, 35)
    print(f"40th percentile threshold: {threshold}")

    # Remove edges that have a weight below the 40th percentile
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)

    # Create adjacency matrix after removing edges
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_df = pd.DataFrame(adj_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))

    # Extract remaining weights
    remaining_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(remaining_weights)

    return G, adj_matrix, remaining_weights

