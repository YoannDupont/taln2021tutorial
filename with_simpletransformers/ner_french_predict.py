"""Apply a simpletranformers model on some text data given as a text stream read
from the standard input.

Currently uses SEM to pre-/post- process data, should consider using SpaCy to
tokenize input and output data in a BRAT format.
"""

import argparse
import sys
import time
import torch
import pandas as pd

import sem.storage
import sem.modules.segmentation
import sem.modules.export
import sem.modules.label_consistency

from simpletransformers.ner import NERModel


segmenter = sem.modules.segmentation.SEMModule("fr")
exporter = sem.modules.export.SEMModule("html", ner_column="NER")
consistency = sem.modules.label_consistency.SEMModule("NER")


def main(model_path):
    model = NERModel(
        "camembert",
        model_path,
        use_cuda=torch.cuda.is_available(),
    )

    text = sys.stdin.read()
    doc = sem.storage.Document("document", text)
    segmenter.process_document(doc)
    tokens = [text[w.lb: w.ub] for w in doc.segmentation("tokens")]
    sentences = [tokens[s.lb: s.ub] for s in doc.segmentation("sentences")]
    predictions, raw_outputs = model.predict(sentences, split_on_space=False)
    tagss = []

    for i, pred in enumerate(predictions):
        tags = [val for val in [list(l.values())[0] for l in pred]]
        doc.corpus.sentences[i].add(tags, "NER")
        tagss.append(tags)

    consistency.process_document(doc)

    doc._annotations["NER"] = sem.storage.chunk_annotation_from_corpus(
        doc.corpus, "NER", "NER", reference=doc.segmentation("tokens")
    )

    exporter.process_document(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_path", help="Path to train file.")
    args = parser.parse_args()

    main(**vars(args))
