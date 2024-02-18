from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
import requests
import PyPDF2

# 下载 PDF 文件
url = "https://img.tdcktz.com/jsq/bp.pdf"
response = requests.get(url)
with open("business_plan.pdf", "wb") as f:
    f.write(response.content)

# 提取文本
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

business_plan_text = extract_text_from_pdf("business_plan.pdf")

# 结构化数据
business_plan_data = {
    "text": business_plan_text
}
import json
with open("business_plan_dataset.json", "w") as f:
    json.dump(business_plan_data, f)
# 加载商业计划书数据集（示例）
# business_plan_dataset = load_dataset("./business_plan_dataset.json")
business_plan_dataset = Dataset.from_dict({"text": [business_plan_text]})


# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 准备数据集
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_datasets = business_plan_dataset.map(tokenize_function, batched=True)

class CustomModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # 取CLS token的输出作为分类结果
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss
        return logits

# 加载BERT模型及分类头部
model = CustomModel(num_labels=3)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="epoch"
)

# 定义Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# 开始微调
trainer.train()

# 保存微调后的模型
trainer.save_model("./business_plan_innovation_model")

# 在测试集上评估微调后的模型
eval_results = trainer.evaluate(tokenized_datasets["test"])
print("Evaluation Results:", eval_results)
