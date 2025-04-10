import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def main():
    file_path = '/social_data/Zombies_Elo_Ratings.csv'
    combined_elo_ratings_sorted = pd.read_csv(file_path)

    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph, with the Elo rating as node attributes
    for index, row in combined_elo_ratings_sorted.iterrows():
        G.add_node(row['Monkey'], rating=row['Elo Rating'])

    # Create linear directed edges between nodes based on Elo rating hierarchy
    sorted_monkeys = combined_elo_ratings_sorted['Monkey'].values
    for i in range(len(sorted_monkeys) - 1):
        G.add_edge(sorted_monkeys[i], sorted_monkeys[i + 1])

    # Create a vertical position layout for nodes
    pos = {monkey: (0, -i) for i, monkey in enumerate(sorted_monkeys)}
    # Horizontal
    # pos = {monkey: (i, 0) for i, monkey in enumerate(sorted_monkeys)}

    # Draw the graph without default labels and using a smaller node size
    plt.figure(figsize=(8, 6))  # Adjust figure size to be taller for vertical layout
    nx.draw(G, pos, with_labels=False, node_size=800, font_size=10, node_color="tomato",
            arrowsize=20, edge_color="dimgray", arrows=True)

    # Draw edges with same thickness and color
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)

    # Add custom labels to nodes showing their Elo ratings
    node_labels = nx.get_node_attributes(G, 'rating')
    custom_labels = {k: f"{k} ({int(v)})" for k, v in node_labels.items()}
    custom_labels = {k: f"{k}" for k, v in node_labels.items()}
    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=10, font_color="white")

    plt.title("Home Group's Cardinal Ranking Based on Elo Ratings")
    # Display the graph in a new window
    plt.show()
    plt.savefig('elo-rating.png')


if __name__ == '__main__':
    main()
