# Training some french NER model using simpletransformers library

## External ressources required

[python3 (already installed on UNIX systems)](https://www.python.org)

[simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)

[SEM (for now)](https://github.com/YoannDupont/SEM)

## Installation

First, install virtualenv for python, create a new virtual environment called
`cantal` and activate it (afterwards, only the third line will be required).

```
pip3 install virtualenv
python3 -m virtualenv ${HOME}/venvs/cantal
source ${HOME}/venvs/cantal/bin/activate
```

Then, install simpletransformers :

```
pip install simpletransformers
```

To install SEM, first clone it from github and go in the `dev` branch:

```
git clone git@github.com:YoannDupont/SEM.git
cd SEM/
git checkout dev
```

Then, install SEM using the [installation procedure](https://github.com/YoannDupont/SEM/blob/dev/install.md).

## Train some model

Training will be done using the `named_entity_recognition_french.py` script. To
see the help of this script, launch:

```
python ./named_entity_recognition_french.py -h
```

If you want to train on some `CoNLL` data (this is the simplest command to do
so, you can modify options of course):

```
python ./named_entity_recognition_french.py <conll_file>
```

If you want to train on some `presto` data (this is the simplest command to do
so, you can modify options of course):

```
python ./named_entity_recognition_french.py <presto_file> -f presto
```

You can notice the only difference is the option `-f presto`. By default, this
script accepts CoNLL files.

## Apply a trained model

To apply a trained model on a random sentence, use the command:

```
echo "Je suis chez ce cher Serge." | python ./ner_french_predict.py path/to/best_model
```

To apply a trained model on a text file, use the command:

```
cat <inputfile> | python ./ner_french_predict_sem.py path/to/best_model
```

This will display on the terminal the output of the NER.
