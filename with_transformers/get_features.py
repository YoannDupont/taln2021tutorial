def print_features(filepath):
    with open(filepath, encoding="utf-8") as f:
        ner_tags = set()
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                continue
            else:
                # FTM tokens are tab separated
                splits = line.split('\t')
                ner_tags.add(splits[3].rstrip())
        # last example
        print(ner_tags)
        
print_features("path/du/fichier")
