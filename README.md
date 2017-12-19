# Herodotos_Project_Annotation
Latin texts annotated for named entities as part of the Herodotos Project (Ohio State University / Ghent University)

The data files in this directory were annotated for named entities by Alex Erdmann, Petra Ajaka, Dr. Christopher Brown, and Dr. Brian Joseph at the Ohio State University. They are taken from Caesar's Gallic Wars (GW), Pliny's Epistulae, and Ovid's Ars Amatoria.

Names of peoples are annotated as GRP; names of persons are annotated as PRS; and names of geographical places are annotated as GEO. These classes can also take any of 3 suffixes (or no suffix), i.e., GEOU is a single-word, unitary geographical place name, like "Greece", whereas a multi-word named entity like "Gaius Julius Caesar" would take the following label sequence: PRSF PRS PRSL, with F and L standing for first and last respectively. A lack of suffix denotes that the word is a non-initial and non-final element of a multi-word named entity. All non-named entities take the label "0".

Further information, including our splits for training and testing, can be found in the relevant paper in this directory, "Challenges and Solutions for Latin Named Entity Recognition". This is the paper that should be cited if you wish to use the data.

Please contact erdmann.6@osu.edu or any of the co-authors with questions regarding this data.
