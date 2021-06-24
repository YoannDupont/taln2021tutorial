# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Presto Corpus"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{gabay:hal-03187097,
  TITLE = {{A dataset for automatic detection of places in (early) modern French texts}},
  AUTHOR = {Gabay, Simon and Ortiz Su{\'a}rez, Pedro Javier},
  URL = {https://hal.archives-ouvertes.fr/hal-03187097},
  BOOKTITLE = {{NASSCFL 2021 - 50th Annual North American Society for Seventeenth-Century French Literature Conference}},
  ADDRESS = {Iowa City / Virtual, United States},
  ORGANIZATION = {{NASSCFL}},
  PAGES = {5},
  YEAR = {2021},
  MONTH = May,
  PDF = {https://hal.archives-ouvertes.fr/hal-03187097/file/NASSCFL.pdf},
  HAL_ID = {hal-03187097},
  HAL_VERSION = {v1},
}
"""

_DESCRIPTION = """\
Linguistically annotated corpora of modern French (16-18th c.)
"""



class PrestoConfig(datasets.BuilderConfig):
    """BuilderConfig for Presto"""

    def __init__(self, **kwargs):
        """BuilderConfig for Presto.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PrestoConfig, self).__init__(**kwargs)


class Presto(datasets.GeneratorBasedBuilder):
    """Presto dataset."""

    BUILDER_CONFIGS = [
        PrestoConfig(name="Presto", version=datasets.Version("1.0.0"), description="Presto dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "B-event",
                                "B-func",
                                "B-loc",
                                "B-org",
                                "B-pers",
                                "B-prod",
                                "B-time",
                                "I-event",
                                "I-func",
                                "I-loc",
                                "I-org",
                                "I-pers",
                                "I-prod",
                                "I-time",
                                "O",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/e-ditiones/LEM17",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """The `data_files` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].
        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datasets.Split` enum.
        """

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_files["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self.config.data_files["test"]}),
        ]
        

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # presto tokens are tab separated
                    splits = line.split('\t')
                    tokens.append(splits[0])
                    ner_tags.append(splits[3].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }