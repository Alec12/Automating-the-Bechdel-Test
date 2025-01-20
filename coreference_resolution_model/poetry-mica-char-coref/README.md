# Character Coreference Resolution in Movie Screenplays

## Introduction

This repository contains the source code and data for training a coreference resolution model designed to identify character references in movie screenplays, enabling automated scoring for the Bechdel Test. The approach builds on the research of Baruah and Narayanan (2023), who adapted Dobrovolskii’s (2021) word-level coreference resolution model to the unique challenges of screenplay formats, including handling long document sizes.

### How It Works
The model operates by encoding tokens into word representations using a pretrained RoBERTa transformer model, augmented with features such as part-of-speech tags, named entities, and screenplay structural tags. A bi-directional RNN generates hidden layer vectors, which are fed into a feed-forward neural network to compute character and antecedent scores for each word representation. Overlapping subdocuments (maximum 5,120 tokens with 2,048-token overlaps) are used to process long screenplays, ensuring continuity of coreference clusters across sections. If an antecedent pair exists in overlapping regions, scores are averaged for consistency.

### Bechdel Test Scoring
Using the trained model, coreference resolution is applied to identify character references within screenplay dialogue. By analyzing these references, the model determines conversations between women and evaluates their topics to automate the three criteria of the Bechdel Test:
1. Presence of two women in a conversation.
2. The conversation is not about a man.
3. At least one scene meets these criteria for the screenplay to pass.

Pretrained weights fine-tuned on six scripts are utilized to perform inference, efficiently handling new datasets while maintaining high accuracy. Scenes are parsed to identify dialogue participants, topics, and coreference-identified entities, allowing detailed scoring for Bechdel compliance.

### Key Points
- **Fusion-Based Coreference Modeling:** Adapts long screenplays into smaller subdocuments for accurate and efficient processing, with continuity ensured through overlapping token regions.
- **Structural and Semantic Analysis:** Integrates screenplay structural tags with advanced character identification to capture nuanced interactions within scenes.
- **Automated Scoring Heuristics:** Combines model predictions with rule-based criteria to evaluate Bechdel compliance at a granular level, assigning passing scores where applicable.

This approach provides a scalable, efficient, and interpretable solution for automating Bechdel Test scoring in movie screenplays, addressing challenges of document length and complex character relationships.


## Training Data

The __MovieCoref__ corpus is saved to the [data](data/) directory.

The __MovieCoref__ coreference annotations and screenplay documents can be found in the [data/labels](data/labels/) 
and [data/screenplay](data/screenplay/) directories, respectively.
[data/movies.txt](data/movies.txt) contains the name of the movies and their raters.
[data/parse.csv](data/parse.csv) contains the line-level structural tags of the screenplays, obtained after 
screenplay parsing.

The [data/addsays](data/addsays/), [data/nocharacters](data/nocharacters/), and [data/regular](data/regular/) 
are JSON-preprocessed versions of the __MovieCoref__ corpus, which are more convenient for programmatic usage.

- The _addsays_ version adds the word "says" between speakers and their utterance.
- The _nocharacters_ version omits the speakers entirely from the screenplay.
- The _regular_ version does not make any changes.

We recommend using the _regular_ version.

### Training

Preprocess the screenplay documents and the annotation labels.
This creates preprocessed JSON files.

```
python preprocess.py --gold
bash train.sh
```

## Prediction

Given a movie script for which you want to find the character coreference clusters, you need to first parse it to
get the line-level tags.
You can use the [Movie Screenplay Parser](https://github.com/usc-sail/mica-screenplay-parser) to do this task.
Refer to its repository regarding how to use it.

```python
from movie_coref import preprocess
from movie_coref.movie_coref import MovieCoreference

# SCRIPT_FILE and PARSE_FILE are the filepaths to the script and parse file
# you can preprocess more than one movie by providing multiple script and parse files
# movie_data will be a list of processed movie scripts
movie_data = preprocess([SCRIPT_FILE], [PARSE_FILE])

# Instantiate the coreference model and pass the preprocessed data
movie_coref = MovieCoreference(
        preprocessed_data=movie_data,
        weights_file=WEIGHTS_FILE,
        hierarchical=HIERARCHICAL,
        document_len=SUBDOC_LEN,
        overlap_len=OVERLAP_LEN,
        n_representative_mentions=REPK,
        )

# Perform the inference
# movie_data will contain the predicted clusters
movie_data = movie_coref.predict()
```

Set WEIGHTS_FILE to `data/Mar09_01:31:43PM_24839/movie_coref.pt`. <br>
Set HIERARCHICAL to `True` if you wish to use the hierarchical mode, otherwise the fusion-based mode will be used
(refer to the paper).

SUBDOC_LEN is the length of the subdocuments into which the movie script is split. <br>
OVERLAP_LEN is the overlap length between the subdocuments for the fusion mode. <br>
REPK is the number of representatives per cluster for the hierarchical mode.
All details of these hyperparameters can be found in the paper.

Default values of SUBDOC_LEN, OVERLAP_LEN, and REPK are 5120, 2048, and 3.

## Inference

To perform inference for automating the Bechdel Test, follow the steps below to set up the environment and run the necessary processes:  

### 1. Set Up Your Environment  

Ensure you have **Poetry** installed to manage dependencies. If you don’t already have it, you can install it by following the instructions on their [official website](https://python-poetry.org/docs/).  

Once installed, navigate to the project directory in your terminal and run:  

```bash
poetry install
```  

This will create a virtual environment and install all the required dependencies listed in the `pyproject.toml` file.  

### 2. Run the Inference Notebook  

The inference process is outlined in the **`241113_experiment.ipynb`** notebook. This notebook provides:  
- **Steps taken** for setting up the automated pipeline for Bechdel Test inference using coreference resolution.  
- Guidance for loading the trained model and required data.  
- Code for preprocessing input data, running predictions, and analyzing results.  

To open the notebook, run the following command:  

```bash
poetry run jupyter notebook
```  

Then, navigate to the `241113_experiment.ipynb` file in your browser and follow the documented steps.  

### 3. Input and Output Details  

- **Input:** The notebook expects movie scripts or text files formatted according to the requirements specified in the preprocessing section. Please use `coreference_resolution_model/export_pipeline.py` for utility functions to transform movie scripts into the proper format before inference.
- **Output:** The model will predict whether a given movie script passes the Bechdel Test based on its trained architecture. Results will include a binary prediction (`1` for pass, `0` for fail).  

### 4. Additional Notes  

- If you encounter any issues with dependencies or execution, ensure your Poetry environment is active by running:  
  ```bash
  poetry shell
  ```  
- Refer to the `README.md` or the notebook for troubleshooting common errors.  