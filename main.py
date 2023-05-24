import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ArgumentParserオブジェクトを作成
parser = argparse.ArgumentParser(description='このスクリプトの説明')
parser.add_argument('--model_name', type=str, choices=['t5-11b', 't5-3b', 't5-small', 't5-base', 't5-large'], default='t5-11b')
args = parser.parse_args()

print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained(args.model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def translate_text(text):
    input_text = "translate Englist to Japanse: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    translated = model.generate(input_ids=input_ids, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

response = "AI: Please enter English sentence."
print(response)
while True:
    print("You: ", end="")
    input_text = input()
    if input_text == 'end':
        break
    response = translate_text(input_text)
    print(f"T5: {response}")
