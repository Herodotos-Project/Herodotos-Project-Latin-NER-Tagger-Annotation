# Herodotos Project NER Annotation and Tagger

Texts annotated for named entities as part of the Herodotos Project (Ohio State University / Ghent University) as well as the BiLSTM-CRF (Lample et al., 2016) NER tagger trained on said annotation (see the [Humanities Entity Recognizer](https://github.com/alexerdmann/HER) for more details on the NER component).

**All texts are in Latin**, from the [Latin Library Collection](https://www.thelatinlibrary.com) (collected by [CLTK](https://github.com/cltk/latin_text_latin_library)) or [Perseus Latin Collection](http://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:Greco-Roman). **Greek will be added soon**.

The data files in the Annotation directory were annotated for named entities by a team of Latin experts at the Ohio State University. Greek annotation data is forthcoming as well. Texts presently included are excerpts from Caesar's Wars, both Gallic (GW) and Civil (CW), the Plinies' writings, both Elder and Younger, and Ovid's Ars Amatoria.

Names of peoples are annotated as GRP; names of persons are annotated as PRS; and names of geographical places are annotated as GEO in the [BIO](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) scheme.

Further information, including our splits for training and testing, can be found in the relevant paper, [Erdmann et al. (2016), "Challenges and Solutions for Latin Named Entity Recognition"](http://www.aclweb.org/anthology/W16-4012). An updated paper is currently under review.. more to follow.

Also included is the Herodotos Project Latin NER Tagger trained on the entire set of Latin data included in this repository using the BiLSTM-CRF architecture of Lample et al. (2016), "Neural Architectures for Named Entity Recognition". For a quick demo, you can test it with the included sample data like so:

```
cd Herodotos_Project_Latin_NER_tagger
cat sample.in.tok | python tagger.py > sample.out.tags
```

To work, you must use Python 3 with the theano library installed. Furthermore, the input should ideally be tokenized (we separated clictics and punctuation during training) with one sentence per line. The output will return all identified named entities for each line as triples: (character offset within the corresponding line where the named entity starts, the full span of the named entity, the label of the named entity).

Alternative supported input formats can be specified with the ```--inputFormat``` option. They include the conll and crfsuite formats. Conll is one token per line, followed by a tab, then its label (though since we're predicting the label, it doesn't matter what you actually put as the label). Sentence breaks are indicated by a blank line. See *sample.in.conll* for an example.
```
cat sample.in.conll | python tagger.py --inputFormat conll > sample.out.tags
```
Crfsuite formatting is the same as conll but the token-label order is reversed. See *sample.in.crf* for an example.
```
cat sample.in.crf | python tagger.py --inputFormat crf > sample.out.tags
```

You can also request different output formats via the ```--outputFormat``` option. The following example will output to crfsuite format:
```
cat sample.in.tok | python tagger.py --outputFormat crf > sample.out.crf
```
And this will output to conll format:
```
cat sample.in.tok | python tagger.py --outputFormat conll > sample.out.conll
```
Alternatively, you can print out a list of all unique entities identified by label with the list option:
```
cat sample.in.tok | python tagger.py --outputFormat list > sample.out.list
```
And of course, any combination of input and output formats is supported.

Please contact erdmann.6@osu.edu or any of the co-authors with questions regarding this data. If you find this work useful in any way, please cite [Erdmann et al. (2016)](http://www.aclweb.org/anthology/W16-4012).
