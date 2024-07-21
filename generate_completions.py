import torch
import argparse
import json
from train_module import EnhancedTransformer, build_vocab, generate_text, JSONLDataset

def load_model(checkpoint_path, vocab_size, d_model, nhead, num_layers):
    model = EnhancedTransformer(vocab_size, d_model, nhead, num_layers)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    return model

def load_jsonl_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            text = ' '.join(json_obj['content'])
            data.append({
                'text': text,
                'language': json_obj['language'],
                'url': json_obj['url'],
                'title': json_obj['metadata']['title']
            })
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate text completions using a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset used for training (JSONL format)")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    # Load the dataset to build the vocabulary
    data = load_jsonl_data(args.dataset)
    word2idx, idx2word = build_vocab(data, args.vocab_size)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.vocab_size, args.d_model, args.nhead, args.num_layers)
    model = model.to(device)
    model.eval()

    print("Model loaded. Ready for text completion.")
    print("Enter your prompt (or type 'quit' to exit):")

    while True:
        prompt = input("> ")
        if prompt.lower() == 'quit':
            break

        generated_text = generate_text(model, prompt, word2idx, idx2word, 
                                       max_length=args.max_length, 
                                       temperature=args.temperature)
        print("\nGenerated completion:")
        print(generated_text)
        print("\nEnter your next prompt (or type 'quit' to exit):")

if __name__ == "__main__":
    main()