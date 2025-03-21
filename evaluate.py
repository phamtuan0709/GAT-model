import json
import torch
import dgl
import networkx as nx
from train_gat_batch import GAT
from load_data import build_graph, get_bert_token_embeddings, load_squad_dataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Load model đã train
in_dim, hidden_dim, out_dim, num_heads = 774, 8, 2, 2  # Cấu hình đúng như khi train
gat_model = GAT(in_dim, hidden_dim, out_dim, num_heads)
gat_model.load_state_dict(torch.load('gat_model.pth'))
gat_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
gat_model.to(device)



def preprocess_dev_data(dev_data):
    """Chuyển tập dev.json thành danh sách graph và câu trả lời thực tế"""
    graphs = []
    ground_truths = []
    
    for article in dev_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                
                # Chuyển câu hỏi thành graph
                graph = build_graph(question, [])
                dgl_graph = dgl.from_networkx(graph)
                
                # Thêm feature vào graph
                features = []
                pos_dict = {'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'OTHER': 0}

                for i, node in enumerate(graph.nodes.values()):
                    bert_feat = node.get('embedding', [0] * 768)  # BERT embedding
                    degree_feat = [node.get('degree', 0) / 10.0]  # Độ kết nối của node
                    pos_feat = [pos_dict.get(node.get('pos', 'OTHER'), 0)]  # POS tagging
                    betweenness_feat = [node.get('betweenness', 0)]
                    pagerank_feat = [node.get('pagerank', 0)]
                    dist_feat = [node.get('dist', 1.0)]  # Khoảng cách đến answer nodes
                    position_feat = [(i + 1) / graph.number_of_nodes()]  # Vị trí của token

                    full_feat = bert_feat + degree_feat + pos_feat + betweenness_feat + pagerank_feat + dist_feat + position_feat
                    features.append(full_feat)

                
                dgl_graph = dgl_graph.to(device)  # Chuyển cả graph lên GPU trước
                dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32).to(device)

                graphs.append(dgl_graph)
                ground_truths.append(answers)
    
    return graphs, ground_truths

def predict_answer(graph):
    """Dự đoán câu trả lời từ graph"""
    graph = dgl.add_self_loop(graph)
    features = graph.ndata['feat']
    
    with torch.no_grad():
        logits = gat_model(graph, features)
        predictions = logits.argmax(dim=1).cpu().numpy()

    # Lấy lại câu hỏi từ graph
    words = [graph.nodes[i]['text'] for i in range(graph.number_of_nodes())]

    # Lấy các token được dự đoán là câu trả lời
    predicted_answer_tokens = [i for i, label in enumerate(predictions) if label == 1]

    answer = " ".join([words[i] for i in predicted_answer_tokens if i < len(words)])

    return answer if answer else "N/A"


# Load dữ liệu dev.json
dev_data = load_squad_dataset(r"split_dev\dev_part1.json")
dev_data['data'] = dev_data['data'][:100]  # Chỉ chạy trên 100 câu đầu tiên để nhanh
graphs, ground_truths = preprocess_dev_data(dev_data)

# Chạy model trên tập dev
predictions = [predict_answer(graph) for graph in graphs]

# So sánh với ground truth
y_true = [1 if pred in ans else 0 for pred, ans in zip(predictions, ground_truths)]
y_pred = [1] * len(y_true)  # Vì model luôn dự đoán 1 câu trả lời, ta giả định có dự đoán

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"📊 Evaluation Results:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {sum(y_true) / len(y_true):.4f}")
print(f"Total questions: {len(y_true)}")