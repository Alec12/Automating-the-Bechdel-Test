#### Automating the Bechdel Test ####

__Alec Naidoo, Zachary Fenton__

![Original-Comic-Strip](/assets/img/bechdel-comic-origin.jpeg)

## Abstract

The Bechdel test is a rudimentary test that has been applied in the social sciences for assessing gender bias in popular culture, in particular the underrepresentation of women in movies. Recent advancements in language models open the door to the possibility of automating the Bechdel test. In this paper, we present two NLP approaches to the automation task. We train a transformer based model with chunking on screenplays that are labeled with binary Bechdel scores (pass/fail). We also take a coreference resolution approach to identify characters within screenplay dialogue to evaluate the Bechdel criteria. Our results show that both methods improve on baseline models and give confidence to the idea that automating the Bechdel test, with modern NLP procedures, is possible.

## Introduction
The Bechdel Test was created in 1985 by Alison Bechdel in their comic titled “The Rule”. The comic strip portrays two women discussing the requirements to see a movie. These rules are ultimately translated into what is now known as the Bechdel test, a simple test to measure the representation of women in movies and literature.
In order to pass the Bechdel Test, three criteria must be met; (1) the piece of media must have at least two __named__ female characters, (2) they must __talk__ to each other, (3) and they must discuss something __other than a man__. Previous studies have utilized a machine learning approach to apply the Bechdel Test to assess gender equality. For example the work of Scheiner-Fisher and Russell III (2012) have applied ML to asses male bias in history curriculum and Garcia et al. (2014) have applied it to study social media posts (i.e. Twitter, Myspace). With the increasing presence of machine learning, automating the Bechdel Test and applying it to movie scripts has become an achievable reality.

In this paper, we build off of the work of Agarwal et al. (2015) in automating the Bechdel Test, specifically as it is applied to movie scripts. We first built a baseline model by replicating the work of Umairican (2014) and obtained an accuracy score of 0.558 and an F1 score of 0.558. Our goal for this paper is to build an automatic Bechdel test that will achieve higher accuracy and F1 while overcoming the challenges of applying transformer models to long document sizes of screenplays. With the advancement of language models, we approach the goal of creating an automatic Bechdel test by first utilising transformer based models and then a character coreference approach.

Our full paper can be found [here](https://github.com/Alec12/Automating-the-Bechdel-Test/blob/main/Automatic%20Bechdel.pdf)

The Bechdel test was originally created by Alison Bechdel. It appeared in a comic stop titled "The Rule" that was published in 1985. A great article on the test can be found [here](https://lithub.com/read-the-1985-comic-strip-that-inspired-the-bechdel-test/).



In the [coref_submit folder](https://github.com/Alec12/Automating-the-Bechdel-Test/tree/main/coref_submit) is the code for the Co-Reference Resolution approach, specifically in [this notebook](https://github.com/Alec12/Automating-the-Bechdel-Test/241113_experiment.ipynb).

In the [screenplayparser folder](https://github.com/Alec12/Automating-the-Bechdel-Test/screenplayparser), you can find the raw files for parsing screenplays. To run them, use [parser_run.ipynb](https://github.com/Alec12/Automating-the-Bechdel-Test/blob/main/parser_run.ipynb)

Source for raw dataset - [Link](https://huggingface.co/datasets/mocboch/movie_scripts/tree/main)

Screenplay parser + coreference resolution original work is adopted from [__sabyasachee__](https://github.com/usc-sail/mica-screenplay-parser/tree/main). Parser is missing the model.pt file for pre-trained model weights. It can be downloaded [here](https://github.com/usc-sail/mica-screenplay-parser/tree/main/screenplayparser) and [here] (https://github.com/usc-sail/mica-character-coref.git)


