import networkx as nx
import matplotlib.pyplot as plt

from monkey_names import Zombies, BestFrans
from unused.network import create_digraph_with_edge_weights, \
    create_digraph_with_top_70_percent_of_edge_weights, create_graph_with_edge_weights, \
    create_graph_with_top_60_percent_of_edge_weights
from data_readers.social_data_reader import SocialDataReader
from social_data_processor import extract_specific_social_behavior, generate_edge_list_from_extracted_interactions
from enums.behaviors import AgonisticBehaviors as Agonistic
from enums.behaviors import SubmissiveBehaviors as Submissive
from enums.behaviors import AffiliativeBehaviors as Affiliative


def main():
    #social_data = SocialDataReader(file_name="ZombiesFinalRawData.xlsx").social_data
    social_data = SocialDataReader(file_name="BestFransFinalRawData.xlsx").social_data
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

    #G, adj_matrix, weights = create_digraph_with_top_70_percent_of_edge_weights(edge_list_agon)
    G, adj_matrix, weights = create_graph_with_top_60_percent_of_edge_weights(edge_list_aff)
    #G, adj_matrix, weights = create_digraph_with_top_70_percent_of_edge_weights(edge_list_sub)


    # Visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Position nodes using the spring layout

    # For re-labeling the nodes
    value_to_name_mapping = {member.name: member.value for member in Zombies}
    # value_to_name_mapping = {member.value: member.name.replace('Z_', 'Z') for member in Zombies}
    custom_labels = {node: value_to_name_mapping.get(node, node) for node in G.nodes()}
    # Draw nodes
    nx.draw_networkx_nodes(G,pos, node_size=1250, node_color='sandybrown')
    # olivedrab and slategrey lines

    # Extract edge weights
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]  # Use the raw weights directly for edge thickness

    # Draw edges with raw thickness based on their weight
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowsize=25, alpha=0.8,
                           edge_color='slategray', width=weights, arrows=True)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels = custom_labels, font_family='monospace', font_size=11, font_color='white')

    # nx.draw(G, pos=pos, with_labels=True, width=weights, font_size=7, node_size=1500)
    plt.savefig("BestFrans_Agonistic.png")
    plt.show()

if __name__ == '__main__':
    main()
