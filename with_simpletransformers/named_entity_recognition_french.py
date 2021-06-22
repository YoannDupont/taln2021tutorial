"""Train some NER model on French data.

Currently, only CamemBERT can be used to train on french data, some options are
investigated to use other transformer-based LM.
"""

import argparse
import sys
import time
import torch
import pandas as pd

from simpletransformers.ner import NERModel, NERArgs


def read_conll(path, columns=None):
    """Read a CoNLL file with no metadata and returns the corpus represented as
    a pandas dataFrame. The dataFrame has three columns: sentence_id, words and
    labels. This is the expected corpus format with simpletransformers.
    """

    columns = columns or [0, -1]
    data = []
    sent_id = 0
    with open(path) as input_stream:
        for line in input_stream:
            line = line.strip()
            if not line:
                sent_id += 1
                continue
            parts = line.split("\t")
            relevant = [parts[column] for column in columns]
            data.append([sent_id] + relevant)
    df = pd.DataFrame(data, columns=["sentence_id", "words", "labels"])
    return data, df


def read_presto(path, columns=None):
    """Read a presto file and returns the corpus represented as a pandas
    dataFrame. The dataFrame has three columns: sentence_id, words and labels.
    This is the expected corpus format with simpletransformers.
    """

    fields = ["form", "lemma", "POS", "O", "O", "O", "O", "_"]
    columns = columns or [0, -1]
    data = []
    sent_id = 0
    prev = None
    with open(path) as input_stream:
        for line in input_stream:
            line = line.strip()
            parts = line.split("\t")
            if not line:
                sent_id += 1
                prev = line
                continue
            elif parts[0].endswith(".tsv") and parts[1] == "xxx":
                if prev:
                    sent_id += 1
                prev = line
                continue
            elif parts == fields:
                prev = line
                continue
            relevant = [parts[column] for column in columns]
            data.append([sent_id] + relevant)
            prev = line

    df = pd.DataFrame(data, columns=["sentence_id", "words", "labels"])
    return data, df


def read_data(path, data_format, columns=None):
    """Read a file to give the pandas dataFrame simpletransformers expects to
    work with.
    """

    func = format2function[data_format.lower()]
    return func(path, columns=columns)


def evaluate(model, df):
    """Print precision, recall and F1-score for using the model `model` on the
    gold corpus `df`."""

    result, model_outputs, predictions = model.eval_model(df)

    print(f"P={result['precision']*100:.2f}")
    print(f"R={result['recall']*100:.2f}")
    print(f"F={result['f1_score']*100:.2f}")


format2function = {
    "conll": read_conll,
    "presto": read_presto,
}


#
# args for the model
#

model_args = NERArgs()
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.num_train_epochs = 5
model_args.use_multiprocessing = False
model_args.multiprocessing_chunksize = 1
model_args.process_count = 0
model_args.train_batch_size = 8
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.dataloader_num_workers = 0
# TODO: in real world, change this to evaluate with a given number of training steps instead of each epoch
# following the "gainz fer teh paperz" heuristic
model_args.evaluate_during_training_steps = 0
model_args.evaluate_each_epoch = False
model_args.evaluate_during_training = (
    model_args.evaluate_during_training_steps > 0 or model_args.evaluate_each_epoch
)
model_args.save_steps = -1


def main(
    train_path,
    valid_path=None,
    eval_path=None,
    reload_model=False,
    n_epochs=5,
    data_format="conll",
    word_column=0,
    tag_column=-1,
):
    #
    # Creating train_df, valid_df and eval_df
    #

    start = time.time()
    labels_set = set()  # libraries require the list of labels, we will infer it from the dataset
    columns = [word_column, tag_column]

    if reload_model:
        model_name = str(pathlib.Path(model_args.output_dir) / "best_model")
    else:
        model_name = "camembert-base"

    print("reading train data...")
    start_read = time.time()
    train_data, train_df = read_data(train_path, data_format, columns=columns)
    labels_set.update(set(train_df["labels"]))
    print("done in", time.time() - start_read, "s")

    if valid_path:
        print("reading valid data...")
        start_read = time.time()
        valid_data, valid_df = read_data(valid_path, data_format, columns=columns)
        labels_set.update(set(valid_df["labels"]))
        print("done in", time.time() - start_read, "s")
    else:
        print("no validation data...")
        valid_data, valid_df = [], pd.DataFrame()

    if eval_path:
        eval_data, eval_df = read_data(eval_path, data_format, columns=columns)
        labels_set.update(set(eval_df["labels"]))
    else:
        eval_data, eval_df = [], pd.DataFrame()

    model_args.labels_list = sorted(set(labels_set))
    model_args.num_train_epochs = n_epochs

    #
    # Create a NERModel and train / eval
    #

    model = NERModel(
        "camembert",
        model_name,
        args=model_args,
        use_cuda=torch.cuda.is_available(),
    )

    try:
        model.train_model(
            train_df,
            eval_data=valid_df,
        )
    except KeyboardInterrupt:  # might take some time, allow the user to cut short
        pass

    if not valid_df.empty:
        print()
        print("#" + "="*31)
        print("# valid")
        print("#" + "="*31)
        print()
        evaluate(model, valid_df)

    if not eval_df.empty:
        print()
        print("#" + "="*31)
        print("# eval")
        print("#" + "="*31)
        print()
        evaluate(model, eval_df)

    if valid_df.empty and eval_df.empty:
        print()
        print("#" + "="*31)
        print("# train")
        print("#" + "="*31)
        print()
        evaluate(model, train_df)

    end = time.time()

    print()
    print()
    print(end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_path", help="Path to train file.")
    parser.add_argument("--valid-path", help="Path to valid file.")
    parser.add_argument("--eval-path", help="Path to eval file.")
    # parser.add_argument("--lm-type", default="camembert", help="The name of the language model type to use (default: %(default)s).")
    # parser.add_argument("--lm-name", default="camembert-base", help="The name of the language model name to use (default: %(default)s).")
    parser.add_argument("--reload-model", action="store_true", help="Reload previous best model.")
    parser.add_argument("-e", "--n-epochs", type=int, default=2, help="Index of the word column (default: %(default)s).")
    parser.add_argument("-f", "--data-format", choices=("conll", "presto"), default="conll", help="Format of the data (default: %(default)s).")
    parser.add_argument("--word-column", type=int, default=0, help="Index of the word column (default: %(default)s).")
    parser.add_argument("-t", "--tag-column", type=int, default=-1, help="Index of the tag column (default: %(default)s).")
    args = parser.parse_args()

    main(**vars(args))
