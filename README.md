# Herodotos Project NER Annotation and Tagger

This repository includes texts annotated for named entities as part of the [Herodotos Project](https://u.osu.edu/herodotos/) (Ohio State University / Ghent University) as well as a BiLSTM-CRF ([Lample et al., 2016](https://arxiv.org/abs/1603.01360)) NER tagger pre-trained on said annotation. Please check out the [Humanities Entity Recognizer](https://github.com/alexerdmann/HER) for more details on how it was trained.

## Annotation

**All texts are in Latin** taken from the [Latin Library Collection](https://www.thelatinlibrary.com) (collected by [CLTK](https://github.com/cltk/latin_text_latin_library)) or the [Perseus Latin Collection](http://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:Greco-Roman). **Greek will be added soon**.

The data files in the *Annotation* directory were annotated for named entities by a team of Classics experts at Ohio State University. Texts presently included are excerpts from Caesar's Wars, both Gallic (GW) and Civil (CW), the Plinies' writings, both Elder and Younger, and Ovid's Ars Amatoria.

Names of peoples are annotated as GRP; names of persons are annotated as PRS; and names of geographical places are annotated as GEO in the [BIO](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) scheme.

Further information on the corpus, including splits for training and testing, can be found in Erdmann et al. (2016), "[Challenges and Solutions for Latin Named Entity Recognition](http://www.aclweb.org/anthology/W16-4012)." For citation purposes however, please see the Acknowledgments section below for the more recent/relevant publication to cite. 

## Tagger

The Herodotos Project Latin NER Tagger is trained on the entire set of Latin data included in this repository using the BiLSTM-CRF architecture of Lample et al. (2016), "[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)".

### Prerequisites

To run the tagger, make sure the below packages and any dependencies have been installed.

* [Python 3](https://www.python.org/downloads/)
* [Theano](https://github.com/Theano/Theano)

### Usage

The tagger can be called with the following commands:

```
cd Herodotos_Project_Latin_NER_tagger
python tagger.py --input sample.in.tok > sample.out.tags
```

The input should already be tokenized with clitics separated and one sentence per line, as in *sample.in.tok*. Near optimal performance can be achieved by simply using punctuation-and-white-space or even just white-space tokenization due to the relative infrequency of Latin cliticization and the tagger's robust handling of character-level features. The output will return all identified named entities for each line as triples. Each triple contains the following information: (1) character offset within the corresponding line where the named entity starts (2) the full span of the named entity (3) the label of the named entity.

Alternative supported input formats can be specified with the ```--inputFormat``` option. They include the conll and [crfsuite](http://www.chokkan.org/software/crfsuite/) formats. Conll is one token per line, followed by a tab, then its label (though since we're predicting the label, it doesn't matter what you actually put as the label). Sentence breaks are indicated by a blank line. See *sample.in.conll* for an example.

```
python tagger.py --input sample.in.conll --inputFormat conll > sample.out.tags
```
If you are on a Mac OS system and get errors when running this command, add `theano.config.gcc.cxxflags = "-Wno-c++11-narrowing` to the tagger.py file, to supress compiler warnings.

Crfsuite formatting is the same as conll but the token-label order is reversed. See *sample.in.crf* for an example.

```
python tagger.py --input sample.in.crf --inputFormat crf > sample.out.tags
```

You can also request different output formats via the ```--outputFormat``` option. The following example will output to crfsuite format:

```
python tagger.py --input sample.in.tok --outputFormat crf > sample.out.crf
```

And this will output to conll format:

```
python tagger.py sample.in.tok --outputFormat conll > sample.out.conll
```

Alternatively, you can print out a list of all unique entities identified by label with the list option:

```
python tagger.py sample.in.tok --outputFormat list > sample.out.list
```

And of course, any combination of input and output formats is supported.

## Acknowledgments

If you find either the tagger or the data useful in any way, please cite our forthcoming publication:

* Alexander Erdmann, David Joseph Wrisley, Benjamin Allen, Christopher Brown, Sophie Cohen Bodénès, Micha Elsner, Yukun Feng, Brian Joseph, Béatrice Joyeaux-Prunel and Marie-Catherine de Marneffe. 2019. “[Practical, Efficient, and Customizable Active Learning for Named Entity Recognition in the Digital Humanities](https://github.com/alexerdmann/HER/blob/master/HER_NAACL2019_preprint.pdf).” In *Proceedings of North American Association of Computational Linguistics (NAACL 2019)*. Minneapolis, Minnesota.

Contact ae1541@nyu.edu or any of the co-authors with questions regarding this repository. 
