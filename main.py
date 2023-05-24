import argparse
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ArgumentParserオブジェクトを作成
parser = argparse.ArgumentParser(description='このスクリプトの説明')
parser.add_argument('--model_name', type=str, choices=['t5-11b', 't5-3b', 't5-small', 't5-base', 't5-large'], default='t5-large')
args = parser.parse_args()

print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(args.model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
print("model loaded")

response = "AI: Please enter sentence."
print(response)
while True:
    print("You: ", end="")
    input_text = input()
    if input_text == 'end':
        break
    
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(device)

    # Generate language features
    with torch.no_grad():
        outputs = model.encoder(inputs['input_ids'])

    # Extract the language features
    language_features = outputs.last_hidden_state.squeeze(0)

    print(language_features)
    print(language_features.shape)
    print(sys.getsizeof(input_text), sys.getsizeof(language_features))