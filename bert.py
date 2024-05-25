from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

input_file_path = "ICEWS05-15/entity"
output_file_path = "embedding_ICE/features.pt"

encoded_vectors = []


with open(input_file_path, "r", encoding="utf-8") as input_file:
    total_lines = sum(1 for _ in input_file)
col = 1

with open(input_file_path, "r", encoding="utf-8") as input_file:
    for line in tqdm(input_file, total=total_lines):
        text = line.strip()
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')


        with torch.no_grad():
            outputs = model(**encoded_input)
            pooled_output = outputs.pooler_output

        encoded_vectors.append(pooled_output[0])
        col = col + 1
print(len(encoded_vectors))

encoded_tensor = torch.cat(encoded_vectors, dim=0)
print(encoded_tensor.shape)

torch.save(encoded_tensor, output_file_path)
import torch

output_file_path = "embedding_ICE/features.pt"

encoded_tensor = torch.load(output_file_path)

print("Encoded Tensor Shape:", encoded_tensor.shape)
encoded_tensor = encoded_tensor.reshape(98851,-1).reshape(col, -1)
torch.save(encoded_tensor, 'embedding_ICE/features.pt')