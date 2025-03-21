import dgl
import torch
import torch.nn.functional as F
from dgl.nn import GATConv
import json
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score
import random

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.gat3 = GATConv(hidden_dim * num_heads, out_dim, num_heads)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, g, inputs):
        h = self.gat1(g, inputs)
        h = h.flatten(1)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gat2(g, h)
        h = h.flatten(1)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gat3(g, h)
        h = h.mean(1)
        return h


def load_graphs_from_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        graph_data = json.load(file)

    graphs = []
    for graph_dict in graph_data:
        nx_graph = nx.node_link_graph(graph_dict)
        dgl_graph = dgl.from_networkx(nx_graph)

        dgl_graph = dgl.add_self_loop(dgl_graph)

        dgl_graph.ndata['_ID'] = torch.arange(dgl_graph.number_of_nodes())

        # Tính centrality và pagerank cho từng graph
        betweenness = nx.betweenness_centrality(nx_graph)
        pagerank = nx.pagerank(nx_graph)

        answer_nodes = [i for i, node in enumerate(nx_graph.nodes.values()) if node['label'] == 1]
        num_nodes = dgl_graph.number_of_nodes()

        features = []
        pos_dict = {'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'OTHER': 0}

        for i, node in enumerate(nx_graph.nodes.values()):
            bert_feat = node['embedding']
            degree_feat = [node['degree'] / 10.0]
            pos_feat = [pos_dict.get(node['pos'], 0)]

            betweenness_feat = [betweenness[i]]
            pagerank_feat = [pagerank[i]]

            dists = []
            for a in answer_nodes:
                try:
                    d = nx.shortest_path_length(nx_graph, source=i, target=a)
                    dists.append(d)
                except nx.NetworkXNoPath:
                    dists.append(num_nodes)  # Không có đường thì gán max

            if dists:
                dist = min(dists)
            else:
                dist = num_nodes  # Trường hợp không có answer node nào

            dist_feat = [dist / num_nodes]


            position_feat = [(i + 1) / num_nodes]

            full_feat = (
                bert_feat +
                degree_feat +
                pos_feat +
                betweenness_feat +
                pagerank_feat +
                dist_feat +
                position_feat
            )
            features.append(full_feat)

        # Tổng hợp lại toàn bộ feature của graph
        features = torch.tensor(features, dtype=torch.float32)

        labels = torch.tensor([node['label'] for node in nx_graph.nodes.values()], dtype=torch.long)

        dgl_graph.ndata['feat'] = features
        dgl_graph.ndata['label'] = labels

        graphs.append(dgl_graph)


    return graphs

def train_gat_batch(graphs, num_epochs=30, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    weights = torch.tensor([1.0, 9.0], device=device)  # Tăng weight class 1
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    batched_graph = dgl.batch(graphs).to(device)
    features = batched_graph.ndata['feat'].to(device)
    labels = batched_graph.ndata['label'].to(device)

    num_class_0 = (labels == 0).sum().item()
    num_class_1 = (labels == 1).sum().item()
    print(f"Class 0: {num_class_0}, Class 1: {num_class_1}")


    gat_model = GAT(in_dim=features.shape[1], hidden_dim=8, out_dim=2, num_heads=2).to(device)
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=lr)

    best_f1 = 0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        gat_model.train()

        logits = gat_model(batched_graph, features)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        precision = precision_score(labels_np, predictions, average='binary', zero_division=0)
        recall = recall_score(labels_np, predictions, average='binary', zero_division=0)
        f1 = f1_score(labels_np, predictions, average='binary', zero_division=0)

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    print("GAT model trained successfully!")

    torch.save(gat_model.state_dict(), 'gat_model.pth')
    print(" Model saved as 'gat_model.pth'")





def balance_graph_nodes(graph, target_ratio=10):
    labels = graph.ndata['label'].numpy()
    node_indices = list(range(len(labels)))

    class_0 = [i for i in node_indices if labels[i] == 0]
    class_1 = [i for i in node_indices if labels[i] == 1]

    if len(class_0) > len(class_1) * target_ratio:
        keep_class_0 = random.sample(class_0, len(class_1) * target_ratio)
        keep_indices = set(keep_class_0 + class_1)

        subgraph = graph.subgraph(list(keep_indices))
        subgraph.ndata['feat'] = graph.ndata['feat'][list(keep_indices)]
        subgraph.ndata['label'] = graph.ndata['label'][list(keep_indices)]

        # Fix lỗi schema
        subgraph.ndata['_ID'] = torch.arange(subgraph.number_of_nodes())

        return subgraph
    else:
        return graph


if __name__ == "__main__":
    graph_data_path = r'graph\train_graph_part1.json'
    graphs = load_graphs_from_json(graph_data_path)
    print(f"Loaded {len(graphs)} graphs.")

    # Cân bằng graph
    graphs = [balance_graph_nodes(g) for g in graphs]

    for g in graphs:
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.number_of_edges())



    # In thử 5 graph đầu + 5 graph cuối
    print("Balanced graph stats (first 5 and last 5):")
    for i, g in enumerate(graphs):
        if i < 5 or i >= len(graphs) - 5:
            labels = g.ndata['label'].numpy()
            print(f"Graph {i+1}: Total nodes {len(labels)}, Class 0: {(labels == 0).sum()}, Class 1: {(labels == 1).sum()}")

    train_gat_batch(graphs)





# Sử dụng batch training giúp tăng tốc độ huấn luyện và giảm bộ nhớ cần thiết.