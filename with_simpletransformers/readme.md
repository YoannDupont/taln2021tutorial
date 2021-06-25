Pour la version anglaise, voir [readme-en.md](readme-en.md)

# Entraîner un model de REN avec la bibliothèque simpletransformers

## Ressources externes requises

[python3 (déjà installé sur les systèmes UNIX)](https://www.python.org)

[simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)

[SEM (temporairement)](https://github.com/YoannDupont/SEM)

## Installation

Installez virtualenv pour python, créez un nouvel environnement virtuel appelé
`cantal` et activez-le (après, vous pourrez uniquement activer l'environnement).

```
pip3 install virtualenv
python3 -m virtualenv ${HOME}/venvs/cantal
source ${HOME}/venvs/cantal/bin/activate
```

Installez la plupart des librairies nécessaires (dont simpletransformers) :

```
pip install -r requirements.txt
```

Pour installer SEM, clonez-le depuis github, allez dans la branche `dev` et
suivez la procédure d'installation :

```
git clone https://github.com/YoannDupont/SEM.git
cd SEM/
git fetch
git checkout dev
source ${HOME}/venvs/cantal/bin/activate
pip install -r requirements.txt
python ./setup.py install
```

Puis installez SEM à l'aide de la [procédure d'installation](https://github.com/YoannDupont/SEM/blob/dev/install.md).

## Entraînez un modèle

Avant d'exécuter les commandes, placez-vous dans le dossier :

```
cd with_simpletransformers
```

L'entraînement se fait via le script `named_entity_recognition_french.py`. Pour
voir l'aide du script, lancez la commande :

```
python ./named_entity_recognition_french.py -h
```

Pour entraîner un modèle sur des données `CoNLL` (command la plus simple, il y a
des options):

```
python ./named_entity_recognition_french.py <conll_file>
```

Pour entraîner un modèle sur des données `presto` (command la plus simple, il y a
des options)

```
python ./named_entity_recognition_french.py <presto_file> -f presto
```

La seule différence est l'option `-f presto`. Par défaut, le script attend des
fichiers CoNLL.

## Appliquez un modèle entraîné

Avant d'exécuter les commandes, placez-vous dans le dossier :

```
cd with_simpletransformers
```

Pour appliquer un modèle entraîné sur une phrase, lancez la commande :

```
echo "Je suis chez ce cher Serge." | python ./ner_french_predict.py chemin/vers/dossier_modele
```

Pour appliquer un modèle entraîné sur un fichier texte, lancez la commande :

```
cat <inputfile> | python ./ner_french_predict_sem.py chemin/vers/dossier_modele
```

Le script affichera dans le terminal le résultat de l'annotation en entités nommées.

Pour que la visualisation en HTML se fasse correctement, vous pouvez rediriger
la sortie vers un fichier dans un dossier avec les feuilles de style CSS requises :

```
cat <inputfile> | python ./ner_french_predict_sem.py chemin/vers/dossier_modele > ../sample_output/resultat.html
```
