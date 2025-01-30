# train.py
import os
import json
import torch
import jieba
import math
import time
import sys
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

# ==================== 配置参数 ====================
CONFIG = {
    "data_path": r"E:\BaiduNetdiskDownload\wiki_zh_2019\wiki_zh",
    "max_seq_len": 64,
    "d_model": 256,
    "nhead": 4,
    "num_layers": 2,
    "batch_size": 8,
    "lr": 0.0001,
    "epochs": 1,
    "vocab_size": 10000,
    "min_word_count": 2,
    "train_ratio": 0.95,
    "num_workers": 0,
    "save_dir": "./wiki_model",
    "force_cpu": False
}

# ==================== 环境验证 ====================
def check_environment():
    print("="*40)
    print("正在验证运行环境...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"当前设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*40)
    time.sleep(1)

def list_files(directory):
    results = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            results.append(item_path)
        elif os.path.isdir(item_path):
            results.extend(list_files(item_path))
    return results

# ==================== 数据处理类 ====================
class WikiDataProcessor:
    def __init__(self):
        self.word_counter = Counter()
        self.special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']



    def process_files(self, file_dir):
        try:
            file_list = list_files(file_dir)
            print(f"找到 {len(file_list)} 个数据文件")
            
            for file_name in tqdm(file_list, desc="处理文件中"):
                file_path = os.path.join(file_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            try:
                                article = json.loads(line.strip())
                                self.process_article(article.get('text', ''))
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"\n文件 {file_name} 处理失败: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"\n目录处理失败: {str(e)}")
            raise

    def process_article(self, text):
        try:
            text = text.replace('\n', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
            words = jieba.lcut(text)
            self.word_counter.update(words)
        except Exception as e:
            print(f"文章处理异常: {str(e)}")

    def build_vocab(self):
        try:
            filtered_words = [w for w, c in self.word_counter.items() 
                            if c >= CONFIG["min_word_count"]]
            vocab = self.special_tokens + filtered_words[:CONFIG["vocab_size"]-len(self.special_tokens)]
            print(f"有效词汇数量: {len(filtered_words)}")
            return {word: idx for idx, word in enumerate(vocab)}
        except Exception as e:
            print(f"词表构建失败: {str(e)}")
            raise

# ==================== 数据集类 ====================
class WikiDataset(Dataset):
    def __init__(self, file_dir, vocab, mode='train'):
        self.vocab = vocab
        self.seq_len = CONFIG["max_seq_len"]
        self.mode = mode  # 关键修复1：保存mode为实例变量
        self.data = []
        
        try:
            print(f"\n开始加载{self.mode}数据集...")  # 关键修复2：使用self.mode
            self.load_data(file_dir)
            print(f"{self.mode}集加载完成，样本数: {len(self.data)}")
        except Exception as e:
            print(f"数据集初始化失败: {str(e)}")
            raise

    def load_data(self, file_dir):
        try:
            file_list = [f for f in os.listdir(file_dir) 
                       if os.path.isfile(os.path.join(file_dir, f))]
            
            # 关键修复3：使用self.mode
            for file_name in tqdm(file_list, desc=f"加载{self.mode}数据文件"):
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            article = json.loads(line)
                            tokens = self.text_to_ids(article.get('text', ''))
                            # 关键修复4：传递self.mode
                            if self.is_train_data(line, self.mode):
                                self.data.extend(self.create_sequences(tokens))
                        except:
                            continue
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise

    def is_train_data(self, line, mode):  # 关键修复5：添加mode参数
        is_train = (hash(line) % 100)/100 < CONFIG["train_ratio"]
        return (mode == 'train' and is_train) or (mode == 'valid' and not is_train)

    def text_to_ids(self, text):
        text = text.replace('\n', ' ').replace('\u3000', ' ')
        words = jieba.lcut(text)
        return [self.vocab.get(w, 1) for w in words]

    def create_sequences(self, tokens):
        sequences = []
        for i in range(0, len(tokens), self.seq_len):
            seq = tokens[i:i+self.seq_len]
            if len(seq) < self.seq_len:
                seq += [0]*(self.seq_len - len(seq))
            sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx][:-1]), torch.LongTensor(self.data[idx][1:])

# ==================== 模型类 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class WikiTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CONFIG["d_model"], padding_idx=0)
        self.pos_encoder = PositionalEncoding(CONFIG["d_model"])
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=CONFIG["d_model"],
            nhead=CONFIG["nhead"],
            dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, CONFIG["num_layers"])
        self.fc = nn.Linear(CONFIG["d_model"], vocab_size)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)

    def forward(self, src):
        src_emb = self.embedding(src) * math.sqrt(CONFIG["d_model"])
        src_emb = self.pos_encoder(src_emb)
        output = self.transformer(src_emb.permute(1, 0, 2))
        return self.fc(output.permute(1, 0, 2))

# ==================== 训练流程 ====================
def train():
    try:
        print("\n" + "="*40)
        print("初始化训练流程")
        check_environment()
        
        # 阶段1：数据处理
        print("\n[阶段1] 数据预处理")
        processor = WikiDataProcessor()
        processor.process_files(CONFIG["data_path"])
        vocab = processor.build_vocab()
        print(f"词表大小: {len(vocab)}")

        # 阶段2：数据集准备
        print("\n[阶段2] 准备数据集")
        train_dataset = WikiDataset(CONFIG["data_path"], vocab, 'train')
        valid_dataset = WikiDataset(CONFIG["data_path"], vocab, 'valid')

        # 阶段3：数据加载器
        print("\n[阶段3] 创建数据加载器")
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            pin_memory=True
        )

        # 设备配置
        device = torch.device("cpu") if CONFIG["force_cpu"] else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[阶段4] 初始化模型，使用设备: {device}")

        # 模型初始化
        model = WikiTransformer(len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

        # 阶段5：训练循环
        print("\n[阶段5] 开始训练")
        best_val_loss = float('inf')
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            
            for data, targets in progress_bar:
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 验证步骤
            val_loss = evaluate(model, valid_loader, device, criterion, len(vocab))
            print(f"\nEpoch {epoch} 结果:")
            print(f"- 训练损失: {total_loss/len(train_loader):.4f}")
            print(f"- 验证损失: {val_loss:.4f}")

            # 模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "model.pt"))
                print(f"模型已保存至 {CONFIG['save_dir']}")

        print("\n训练完成！")

    except Exception as e:
        print(f"\n发生致命错误: {str(e)}")
        raise

def evaluate(model, data_loader, device, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="验证中"):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    print("="*40)
    print("程序启动")
    start_time = time.time()
    
    try:
        train()
    except KeyboardInterrupt:
        print("\n用户中断训练")
    except Exception as e:
        print(f"\n未处理的异常: {str(e)}")
    finally:
        print(f"\n总运行时间: {time.time()-start_time:.2f}秒")
        print("="*40)