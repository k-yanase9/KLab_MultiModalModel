import argparse
import sys
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

def cosine_similarity(a, b):
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    similarity = torch.mm(a_norm, b_norm.T)
    return similarity

# ArgumentParserオブジェクトを作成
parser = argparse.ArgumentParser(description='このスクリプトの説明')
parser.add_argument('--model_name', type=str, choices=['t5-11b', 't5-3b', 't5-small', 't5-base', 't5-large'], default='t5-large')
args = parser.parse_args()

print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map='auto', low_cpu_mem_usage=True)
print("Model loaded")

while True:
    print("First input: ", end="")
    input_text1 = input()

    # Tokenize the input text
    input1 = tokenizer.encode_plus(
        input_text1,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Generate language features
    with torch.no_grad():
        output1 = model.encoder(input1['input_ids'])

    # Extract the language features
    language_features1 = output1.last_hidden_state.squeeze(0)
    print(language_features1, language_features1.shape)
    print(sys.getsizeof(input_text1), sys.getsizeof(language_features1))

    print("Second input: ", end="")
    input_text2 = input()

    # Tokenize the input text
    input2 = tokenizer.encode_plus(
        input_text2,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Generate language features
    with torch.no_grad():
        output2 = model.encoder(input2['input_ids'])

    # Extract the language features
    language_features2 = output2.last_hidden_state.squeeze(0)
    print(language_features2, language_features2.shape)
    print(sys.getsizeof(input_text2), sys.getsizeof(language_features2))


    similarity_matrix = cosine_similarity(language_features1, language_features2)
    print(similarity_matrix)