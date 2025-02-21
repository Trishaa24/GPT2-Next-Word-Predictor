from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch

# Function to load GPT-2 model
def load_gpt2():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Function to load WikiText-103 dataset
def load_wikitext():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")  # WikiText-103 dataset
    return dataset["train"]["text"]  # Use the 'train' split of the dataset

# Function to train GPT-2 model using the WikiText-103 dataset
def train_gpt2(dataset):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    text_data = dataset[:5000]
    print(f"Loaded {len(text_data)} text samples.")  # Debugging statement

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    batch_size = 4
    
    if not text_data:
        print("No text data found! Exiting training...")
        return model, tokenizer
    
    for i in range(0, len(text_data), batch_size):
        batch_texts = text_data[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}")  # Debugging statement

        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=256, truncation=True, padding=True)

        inputs = {key: value.to(device) for key, value in inputs.items()}

        try:
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch {i//batch_size + 1} Loss: {loss.item()}")  # Debug loss
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")

    return model, tokenizer



# Function to load the trained model from disk
def load_trained_model():
    model = GPT2LMHeadModel.from_pretrained("trained_model")
    tokenizer = GPT2Tokenizer.from_pretrained("trained_model")
    return model, tokenizer

# Main function to simulate model training and selection
def main():
    print("Starting model training for next word prediction using GPT-2")

    dataset = load_wikitext()

    # Train GPT-2 model
    print("Training GPT-2 model")
    trained_model, tokenizer = train_gpt2(dataset)

    # Optionally, we could train and use BERT as shown below:
    # print("Training BERT model")
    # bert_model, bert_tokenizer = train_bert(dataset)

    return trained_model, tokenizer

if __name__ == "__main__":
    main()
