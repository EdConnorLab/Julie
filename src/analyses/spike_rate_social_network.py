import os
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

from analyses.spike_rate_analysis import read_sorted_data, compute_average_spike_rates_from_raw_trial_data, \
    set_node_attributes_with_default
from monkey_names import Monkey
from network import create_graph_with_edge_weights, create_digraph_with_edge_weights
from social_data_reader import SocialDataReader
from social_data_processor import extract_specific_social_behavior, generate_edge_list_from_extracted_interactions
from behaviors import AgonisticBehaviors as Agonistic
from behaviors import SubmissiveBehaviors as Submissive
from behaviors import AffiliativeBehaviors as Affiliative
from behaviors import IndividualBehaviors as Individual

def main():
    social_data = SocialDataReader().social_data
    # Agonistic
    agonistic_behaviors = list(Agonistic)
    agon = extract_specific_social_behavior(social_data, agonistic_behaviors)
    edge_list_agon = generate_edge_list_from_extracted_interactions(agon)

    # Submissive
    submissive_behaviors = list(Submissive)
    sub = extract_specific_social_behavior(social_data, submissive_behaviors)
    edge_list_sub = generate_edge_list_from_extracted_interactions(sub)

    # Affiliative
    affiliative_behaviors = list(Affiliative)
    aff = extract_specific_social_behavior(social_data, affiliative_behaviors)
    edge_list_aff = generate_edge_list_from_extracted_interactions(aff)

    # #Get Spike Rate
    # date = "2023-10-30"
    # round = "1698699440778381_231030_165721"
    # cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    # round_path = os.path.join(cortana_path, date, round)
    # raw_trial_data = read_sorted_data(round_path)
    # avg_spike_rate = compute_average_spike_rates_from_raw_trial_data(raw_trial_data)
    # random_row = avg_spike_rate.loc["Channel.C_017_Unit 1"]
    # norm_values = ((random_row - random_row.min()) / (random_row.max() - random_row.min())).to_dict()


    G, adj_matrix, weights = create_digraph_with_edge_weights(edge_list_agon)
    # G, adj_matrix, weights = create_graph_with_edge_weights(edge_list_aff)
    # set_node_attributes_with_default(G, norm_values, 'spike_rate', default_value=0)

    # colormap = plt.cm.get_cmap('YlOrBr')
    # attribute = nx.get_node_attributes(G, 'spike_rate')
    # node_colors = [colormap(attribute[node]) for node in G.nodes()]
    print(f'degree centrality {nx.degree_centrality(G)}')
    print(f'betweenness centrality {nx.betweenness_centrality(G)}')
    print(f'eigenvector centrality {nx.eigenvector_centrality(G)}')

    # Visualize the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Position nodes using the spring layout

    # For re-labeling the nodes
    value_to_name_mapping = {member.value: member.name for member in Monkey}
    G_relabelled = nx.relabel_nodes(G, value_to_name_mapping)
    pos_relabelled = {value_to_name_mapping.get(node, node): position for node, position in pos.items()}

    nx.draw(G, pos=pos, with_labels=True, width=weights, font_size=7, node_size=1500)
    plt.show()

if __name__ == '__main__':
    main()
