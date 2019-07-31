import argparse, pickle, random
from data_loader import CwCDataset
from vocab import Vocabulary
from utils import *

def get_perplexity_factor(dev_dataset):
    vocab = dev_dataset.decoder_vocab
    dev_dl = dev_dataset.get_data_loader(batch_size=1, shuffle=True, num_workers=2)

    num_tokens = 0.0 # total number of tokens in validation data
    unk_tokens = 0.0 # total number of unk tokens in validation data
    unk_id = vocab("<unk>")

    for i, (_, _, decoder_outputs) in enumerate(dev_dl):
        target_outputs = to_var(decoder_outputs.target_outputs)
        num_tokens += len(target_outputs[0])
        for token in target_outputs[0]:
            if token.item() == unk_id:
                unk_tokens += 1

    unk_bucket_size = 0.0 # total number of unique tokens in training data represented as unk, i.e., not included in the vocab

    for token in vocab.word_counts:
        if token not in vocab.word2idx:
            unk_bucket_size += 1

    factor = unk_bucket_size ** (unk_tokens/num_tokens)

    return factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_path', type=str, default='../vocabulary/glove.42B.300d-lower-5r-speaker.pkl', help='path for vocabulary wrapper')
    parser.add_argument("--add_builder_utterances", default=False, action="store_true", help="Whether or not to include builder utterances in the datasets")
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    args = parser.parse_args()
        
    random.seed(args.seed)

    lower = "lower" in args.vocab_path.split("/")[-1]

    # load the vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    dev_dataset = CwCDataset(
        split="val", lower=lower,
        num_prev_utterances=1, data_path='../data/logs/',
        gold_configs_dir='../data/gold-configurations/', decoder_vocab=vocab, encoder_vocab=vocab,
        transform=None, add_builder_utterances=args.add_builder_utterances,
        load_dataset = True
    )

    print(get_perplexity_factor(dev_dataset))
