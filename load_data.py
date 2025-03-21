import json
import spacy
import networkx as nx
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import torch
import os

# Load spaCy model and BERT tokenizer/model
nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def load_squad_dataset(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def preprocess_data(squad_data):
    passages = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            answers = [ans['text'] for qa in paragraph['qas'] for ans in qa['answers']]
            passages.append((context, answers))
    return passages


def get_bert_token_embeddings(context):
    inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=512, return_offsets_mapping=True)

    # Tách offset_mapping ra để dùng riêng, KHÔNG đưa vào model
    offset_mapping = inputs.pop('offset_mapping')

    with torch.no_grad():
        outputs = bert_model(**inputs)

    token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
    offset_mapping = offset_mapping.squeeze(0).tolist()  # Để dùng cho mapping sau này


    return token_embeddings, offset_mapping

def build_graph(context, answers):
    doc = nlp(context)
    graph = nx.DiGraph()

    # Lấy toàn bộ BERT embedding 1 lần cho cả câu
    bert_embeddings, bert_tokens = get_bert_token_embeddings(context)

    for token in doc:
        graph.add_node(token.i, text=token.text, pos=token.pos_)

        if token.dep_ != 'ROOT':
            graph.add_edge(token.head.i, token.i, relation=token.dep_)

    # Thêm degree vào node attributes
    for node_id in graph.nodes:
        graph.nodes[node_id]['degree'] = graph.degree(node_id)

    # Ánh xạ token spaCy về BERT token (cần đoạn map khéo)
    token_embeddings, offset_mapping = get_bert_token_embeddings(context)

    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)

        bert_idx = None
        for idx, (start, end) in enumerate(offset_mapping):
            if start == token_start and end == token_end:
                bert_idx = idx
                break

        if bert_idx is not None:
            graph.nodes[token.i]['embedding'] = token_embeddings[bert_idx].tolist()
        else:
            graph.nodes[token.i]['embedding'] = [0] * 768  # fallback nếu không map được


    # Gắn label: 1 nếu token nằm trong answer, 0 nếu không
    answer_tokens = set()
    for ans in answers:
        for token in doc:
            if ans in token.text:
                answer_tokens.add(token.i)

    nx.set_node_attributes(graph, {n: 1 if n in answer_tokens else 0 for n in graph.nodes()}, 'label')

    return graph


def match_spacy_tokens_to_bert(spacy_doc, bert_tokens):
    mapping = {}
    bert_idx = 0

    for spacy_idx, token in enumerate(spacy_doc):
        spacy_token = token.text.lower()
        bert_token = bert_tokens[bert_idx]

        # Map đơn giản: trùng nguyên token
        if spacy_token == bert_token or bert_token.startswith('##'):
            mapping[spacy_idx] = bert_idx
            bert_idx += 1
        else:
            # Trong trường hợp không khớp, cố gắng tìm token tiếp theo
            while bert_idx < len(bert_tokens):
                bert_token = bert_tokens[bert_idx]
                if spacy_token in bert_token or bert_token.startswith('##'):
                    mapping[spacy_idx] = bert_idx
                    bert_idx += 1
                    break
                bert_idx += 1

    return mapping


def save_graphs(graphs, filename):
    graph_data = [nx.node_link_data(graph) for graph in graphs]
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(graph_data, file)


if __name__ == "__main__":
    folder_path = r'split_train'
    output_folder = r'graph'

    # Lấy danh sách tất cả file JSON trong thư mục
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        squad_data = load_squad_dataset(json_path)
        passages = preprocess_data(squad_data)

        graphs = []
        for context, answers in tqdm(passages, desc=f"Processing {json_file}"):
            graph = build_graph(context, answers)
            graphs.append(graph)

        output_path = os.path.join(output_folder, f"graph_{json_file}")
        save_graphs(graphs, output_path)
        print(f"✅ Graphs saved to {output_path}")
