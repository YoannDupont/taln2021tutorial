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
"""CLEF HIPE Corpus"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{ehrmann_extended_2020,
  title = {Extended {Overview} of {CLEF HIPE} 2020: {Named Entity Processing} on {Historical Newspapers}},
  booktitle = {{CLEF 2020 Working Notes}. {Working Notes} of {CLEF} 2020 - {Conference} and {Labs} of the {Evaluation Forum}},
  author = {Ehrmann, Maud and Romanello, Matteo and Fl{\"u}ckiger, Alex and Clematide, Simon},
  editor = {Cappellato, Linda and Eickhoff, Carsten and Ferro, Nicola and N{\'e}v{\'e}ol, Aur{\'e}lie},
  year = {2020},
  volume = {2696},
  pages = {38},
  publisher = {{CEUR-WS}},
  address = {{Thessaloniki, Greece}},
  doi = {10.5281/zenodo.4117566},
  url = {https://infoscience.epfl.ch/record/281054},
}
}
"""

_DESCRIPTION = """\
HIPE (Identifying Historical People, Places and other Entities) is a evaluation campaign on named entity processing on historical newspapers in French, German and English, which was organized in the context of the impresso project and run as a CLEF 2020 Evaluation Lab.
"""



class PrestoConfig(datasets.BuilderConfig):
    """BuilderConfig for HIPE"""

    def __init__(self, **kwargs):
        """BuilderConfig for HIPE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PrestoConfig, self).__init__(**kwargs)


class Presto(datasets.GeneratorBasedBuilder):
    """HIPE dataset."""

    BUILDER_CONFIGS = [
        PrestoConfig(name="HIPE", version=datasets.Version("1.0.0"), description="HIPE"),
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
                                'I-time',
                                'B-org',
                                'I-prod',
                                'I-pers',
                                'B-time',
                                'I-loc',
                                'B-loc',
                                'B-comp',
                                'B-prod',
                                'O',
                                'B-pers',
                                'I-org',
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://impresso.github.io/CLEF-HIPE-2020/",
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
