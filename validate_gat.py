import dgl
import torch
import torch.nn.functional as F
import json
import networkx as nx
from sklearn.metrics import accuracy_score

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.gat1 = dgl.nn.GATConv(in_dim, hidden_dim, num_heads, allow_zero_in_degree=True)
        self.gat2 = dgl.nn.GATConv(hidden_dim * num_heads, out_dim, num_heads, allow_zero_in_degree=True)

    def forward(self, g, inputs):
        h = self.gat1(g, inputs)
        h = h.flatten(1)
        h = F.elu(h)
        h = self.gat2(g, h)
        h = h.mean(1)
        return h

def load_graphs_from_json(path, feature_dim=128):
    with open(path, 'r', encoding='utf-8') as file:
        graph_data = json.load(file)

    graphs = []
    for graph_dict in graph_data:
        nx_graph = nx.node_link_graph(graph_dict)
        dgl_graph = dgl.from_networkx(nx_graph)
        dgl_graph = dgl.add_self_loop(dgl_graph)

        num_nodes = dgl_graph.number_of_nodes()
        dgl_graph.ndata['feat'] = torch.rand(num_nodes, feature_dim)  # Dummy feature

        graphs.append(dgl_graph)

    return graphs

def evaluate_gat(model, graphs, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            features = graph.ndata['feat'].to(device)
            labels = torch.zeros(graph.number_of_nodes(), dtype=torch.long, device=device)  # Dummy labels (có data thật thì đổi chỗ này)

            logits = model(graph, features)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load graph để validate
    val_graphs = load_graphs_from_json('data\dev-v2.0.json')
    print(f"Loaded {len(val_graphs)} validation graphs.")

    # Khởi tạo model (phải khớp với lúc train về số chiều feature và head)
    model = GAT(in_dim=128, hidden_dim=8, out_dim=2, num_heads=2).to(device)

    # Nếu mày có lưu checkpoint (model.pth) thì load
    # model.load_state_dict(torch.load('gat_model.pth'))

    evaluate_gat(model, val_graphs, device)
