import json
import os

# Config
input_file = r'data\dev-v2.0.json'   # File train gốc
output_folder = 'split_dev'  # Folder để chứa các file split
samples_per_file = 5000        # Số samples mỗi file, chỉnh tùy nhu cầu

# Tạo folder chứa file chia nhỏ
os.makedirs(output_folder, exist_ok=True)

# Đọc dữ liệu từ file gốc
with open(input_file, 'r', encoding='utf-8') as f:
    squad_data = json.load(f)

# Lấy danh sách articles (mỗi article chứa nhiều paragraphs & qas)
all_articles = squad_data['data']

# Chia nhỏ list articles ra theo số lượng samples
current_articles = []
current_count = 0
file_index = 1

def save_split_file(index, articles):
    split_data = {
        "version": squad_data.get("version", "unknown"),
        "data": articles
    }
    output_path = os.path.join(output_folder, f'dev_part{index}.json')
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(split_data, f_out, ensure_ascii=False, indent=2)
    print(f'Đã lưu {output_path} ({len(articles)} articles)')

# Bắt đầu chia
for article in all_articles:
    current_articles.append(article)

    # Đếm tổng số câu hỏi (qas) trong article này
    num_qas = sum(len(p['qas']) for p in article['paragraphs'])
    current_count += num_qas

    # Nếu đủ số lượng thì lưu ra file
    if current_count >= samples_per_file:
        save_split_file(file_index, current_articles)
        current_articles = []
        current_count = 0
        file_index += 1

# Nếu còn dư data sau vòng lặp thì lưu nốt
if current_articles:
    save_split_file(file_index, current_articles)

print(f'Tổng cộng đã chia thành {file_index} files')
