# Herodotos Project NER Annotation and Tagger

Latin texts annotated for named entities as part of the Herodotos Project (Ohio State University / Ghent University) as well as the BiLSTM-CRF (Lample et al., 2016) NER tagger trained on said annotation (see the [Humanities Entity Recognizer](https://github.com/alexerdmann/HER) for more details on the NER component).

The data files in the Annotation directory were annotated for named entities by a team of Latin experts at the Ohio State University. Greek annotation data is forthcoming as well. Texts presently included are excerpts from Caesar's Wars, both Gallic (GW) and Civil (CW), the Plinies' writings, both Elder and Younger, and Ovid's Ars Amatoria.

Names of peoples are annotated as GRP; names of persons are annotated as PRS; and names of geographical places are annotated as GEO in the [BIO](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) scheme.

Further information, including our splits for training and testing, can be found in the relevant paper in this directory, Erdmann et al. (2016), "Challenges and Solutions for Latin Named Entity Recognition". This is the paper that should be cited if you wish to use the data. An updated paper is currently under review.. more to follow.

Also included is the Herodotos Project Latin NER Tagger trained on the entire set of Latin data included in this repository using the BiLSTM-CRF architecture of Lample et al. (2016), "Neural Architectures for Named Entity Recognition". For a quick demo, you can test it with the included sample data like so:

```
cat Herodotos_Project_Latin_NER_tagger/sample.tok.in | python Herodotos_Project_Latin_NER_tagger/tagger.py > sample.out.tags
```

To work, you must use Python 3 with the theano library installed. Furthermore, the input should ideally be tokenized (we separated clictics and punctuation during training) with one sentence per line. The output will return all identified named entities for each line as triples: (character offset within the corresponding line where the named entity starts, the full span of the named entity, the label of the named entity).

Please contact erdmann.6@osu.edu or any of the co-authors with questions regarding this data.
