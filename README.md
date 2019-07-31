# Setup
- PyTorch 0.4.1


# Data
In `data`, the Minecraft Dialogue Corpus (without screenshots) is in `logs`. The target structures are in `gold-configurations`. `logs/splits.json` defines the train-test-val split by defining subsets of target structures. `embeddings` contains the glove embeddings we use. `lexicons` contains various handmade lexicons for specific use cases you'll encounter later.

# Data preprocessing
`python/data_loader.py` defines a custom dataset class `CwCDataset`. See example usage in the `__main__` block. The data loader for the class returns an example for every Architect utterance (and optionally every Builder utterance). Such examples will be used for training and testing models.

## Generated datasets
Generated datasets obtained from data preprocessing are stored on our lab machines at `/shared/data/cwc/scratch/lower-no_perspective_coords-fixed`. You'll need to use these to train/test models on our data.

# Vocabularies
[`vocab.py`](python/vocab.py) should be used to generate vocab files. Generated ones are stored in the `vocabulary` directory.

## Creating a vocabulary file
[`vocab.py`](python/vocab.py) creates a vocabulary based on a pretrained word embeddings file (if given) and the words in the dataset. It creates an instantiation of a `Vocabulary` class which has functions to retrieve words by their IDs and vice versa. It also contains fields for vocabulary sizes that can then be used to initialize sizes of LSTM layers. Finally, the `Vocabulary` also contains a field, `self.word_embeddings`, that is directly a `torch.nn.Embedding` layer that can be used to encode word IDs in your model.

To create a vocabulary that will be saved to disk, call `vocab.py` with the following arguments:
* `--vector_filename`: path to the pretrained word embeddings file you want to use. As of now, this script can only parse embeddings in Glove pretrained embedding txt files. In the future, support for word2vec, fastText, and other pretrained embeddings files should be included. If left blank, the resulting vocabulary will return an `Embedding` layer that contains just the words in the training dataset and will be trained (i.e., `requires_grad=True`).
* `--embed_size`: size of the word embeddings. This should be set to the size of the pretrained embeddings you are using, or to a desired embedding size if not using pretrained embeddings.
* `--oov_as_unk`: by default, any words found in the dataset that do not contain pretrained word embeddings and also appear frequently enough in the dataset (over some threshold) are added as random vectors to the vocabulary. To disable this feature, and thus treat such words as `<unk>` tokens, enable this flag.
* `--lower`: lowercase the tokens in the dataset.
* `--keep_all_embeddings`: by default, words that are never seen in the dataset are not kept as word embeddings in the final vocabulary dictionary. This greatly speeds up processing time and reduces the size of the vocabulary that is needed in memory and on disk. For an interactive demo, however, the embeddings for other unseen words should be kept. To do so, enable this flag.
* `--train_embeddings`: by default, if a pretrained embeddings file has been loaded, the resulting `torch.nn.Embedding` layer that is initialized is frozen, i.e., `requires_grad=False`. To enable fine-tuning of the embedding layer, enable this flag.
* `--use_speaker_tokens`: by default, the vocabulary is initialized with generic start- and end-of-sentence tokens, i.e., `<s>` and `</s>`. In order to use speaker-specific tokens, e.g. `<architect>` and `</architect>`, enable this flag.
* `--threshold`: the rare word threshold, below which items in the dataset that do not have pretrained embeddings are treated as `<unk>`.
* `--verbose`: prints the entire dictionary of the vocabulary when initialization is finished.

The resulting vocabulary will be pickled and saved to `cwc-minecraft-models/vocabulary/` and will have a filename that is a combination of the pretrained word embeddings filename and various parameters you have specified (e.g. lowercasing, rare word threshold, and use of speaker tokens). A log of the console messages printed during vocabulary creation is also saved to the same directory.

## Using a generated vocabulary file
The vocabulary can be easily loaded using `pickle.load()`. The `Vocabulary`'s `self.word_embeddings` field, which contains embeddings for all words in the entire dataset, should be directly used as a `torch.nn.Embedding` layer in your model. The embedding for the `<unk>` token is defined to be the average of all word embeddings for words seen in the training set.

Calling the vocabulary will allow you to look up a word ID for a given word using either the words-to-IDs dictionary based on the entire dataset or only that of the train set, the latter of which should be used for decoding. When calling the vocabulary for word lookup, lookups for input words should be simply called as `vocabulary(word)`, while lookups for output words should use `vocabulary(word, split='train')` to disallow unseen tokens from being generated at the output. **Additionally, the output size for your model's final `Linear` layer should use `vocabulary.num_train_tokens` in order for this word lookup to function correctly.**

Similarly, you can use the `ids2words()` function in `utils.py` during decoding to translate a list of word IDs to their respective words as a string. When translating a list of input word IDs, you should use `ids2words(vocabulary, sampled_ids)`, while translating a list of output word IDs (as generated by a sampling function) should use `ids2words(vocabulary, sampled_ids, split='train')`.

# Models
In `python`, models are in the following directories' `model.py` file -- `cnn_3d`, `seq2seq_all_inputs`, `seq2seq_attn` and `seq2seq_world_state`.
- `seq2seq_attn`: An encoder RNN, a decoder RNN, attention
- `seq2seq_world_state`: Global block counters encoder, Regional block counters encoder
- `seq2seq_all_inputs`: Integrated encoder models using an encoder RNN and the block counters encoders

To define your own model, following similar coding patterns and add it to the appropriate directory's `model.py` or create a new such directory if needed.

# Training

## train.py
[`train.py`](python/train.py) trains a given model with given hyperparameters for a certain number of epochs, or until the model's performance on the development set steadily degrades for a given number of epochs.

In order to get a model working with the `train.py` script, portion of code that should be modified is tagged with an `""" IMPLEMENT ME FOR NEW MODELS """` comment. Your implementation should be encapsulated in an if-block that executes when the model name, `args.model`, matches the name of your model. Each if-block is basically to import and define the right kind of encoder model based on `args.model`.

Finally, don't forget to add any new hyperparameters you need to initialize your model to the argparse.ArgumentParser() that is initialized in the `__main__` block. Once you do so, you can test that your model can be trained for one set of hyperparameters by simply running train.py with your desired arguments.

You can now use `train.py <model name>` directly to train a single model with specified hyperparameters. Parameters must be passed in to the script when calling it from the command line. A given model is saved to `cwc-minecraft-models/models/<model name>/<yyyymmdd>/<timestamp-ms>/`, where the timestamps are determined by the start time of the script. In this directory, you will find:
* `config.txt`: lists all the hyperparameters and arguments used for the trained model
* `eval.txt`: a small snippet containing details of when the best model was found, as well as its performance on the validation set as defined by your `EvaluationResult`
* `<model name>-epoch-n.pkl`: saved models per n epochs, as specified (by default, a model is saved every epoch)
* `<model name>-epoch-n-best.pkl`: the best model found across all epochs (the model achieving smallest loss on the validation set)
* `<model name>_train.log`: console log of the training script

To see the command-line arguments and their default values, refer to the `__main__` block of `train.py`.

## trainer.py

[`trainer.py`](python/trainer.py) is a helper script that can initialize and train your model with different combinations of hyperparameters in succession. It is intended to be used to train multiple models in sequence on a single GPU without need for intervention. It first reads in a specified configuration file that defines the different hyperparameter options to be used, then trains a model for each combination of hyperparameters.

There are a few modifications that must be made to `trainer.py` for it to be used with a newly defined model:
* The `parse_value` function should be modified such that newly defined hyperparameter values are parsed into the correct python type that is expected by your model. Since the trainer reads values from a file, values default to `string`, with the `"None"` string being translated to the `None` type. If you wish for different behavior for other hyperparameters, you should specify it here, e.g.:
```
if '_size' in param or '_layers' in param or 'num_prev_utterances' in param:
  return int(value)
```
* The `flatten_combined_params` function should be modified, if desired, to ignore certain combinations of hyperparameters if you deem them redundant. E.g.:
```
if model_name in ['lm', 'seq2seq']:
  if not config.get("linear_size") and config.get("nonlinearity") or config.get("linear_size") and not config.get("nonlinearity"):
    continue
```

To run the trainer, simply call `trainer.py <model name> <path to hyperparameter config file>`, where the hyperparameter configuration file lists a hyperparameter, by name, per line, with the values you would like to use as whitespace-separated options following the names. **Note: it is very important that the hyperparameter names in this file exactly match the names of the hyperparameter fields that you are expecting** (e.g., `hidden_size` and `args.hidden_size`). An example hyperparameter configuration file can be found [here](config/seq2seq/hyperparameters-1.config). A log of the trainer script's console messages is saved to `cwc-minecraft-models/models/<model name>/<yyyymmdd>/<model name>_trainer_<timestamp-ms>.log`, where the timestamps are determined by the start time of the script.

All hyperparameter configuration files should be stored in the `config/<model name>` directory.

# Generation
Should always be done on a CPU for efficiency reasons.

## generate_seq2seq.py
Runs generation and evaluation using a trained model. The CLAs should guide you.

Output:
- `raw_sentences*.txt` -- generated utterances for each example
- `generated_sentences*.txt` -- Some stats, BLEU scores, generated utterances for each example accompanied by some of the model inputs and the ground truth utterance
- Some other less human-readable files for downstream purposes

## generator_seq2seq.py
Meta script that runs generation and evaluation for each model in a collection of models sequentially

## generator_seq2seq.sh
Meta script that runs generation and evaluation for each model in a collection of models parallely

## beam_grid_search.sh
Meta script that runs generation and evaluation using a trained model for a whole grid of gamma and beam size values.

# Training and generation together

## train_and_generate.sh
Meta script that runs training using `trainer.py` on GPU and once all models are done training, switches to CPU and runs generation and evaluation for each using `generator_seq2seq.py`. You'll typically need this script for begin efficient.

# Compiling results plus modified BLEU metrics -- get_evals.py
Meta script that compiles results from generation/eval runs for all models in a directory for models saved on a specific date into a single csv file. Also computes modified BLEU metrics.

Output:
- `cumulative_evals.csv`-- The csv file (use Google Sheets to work with it)

# Limitations
Following are some limitations/stuff not supported currently in this repo:
- No support for LSTMs (implemented for the most part but broken)
- Batch size is always 1
- No support for attention (only a small part implemented)
- Only teacher forcing mode of training supported
