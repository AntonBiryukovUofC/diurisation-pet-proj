Diarisation Pet Proj
==============================
Run `test_diarisation.py` from the project's root folder. Speaker IDs will show up in a text file under `data/processed/`.
Name of that file, as well as a bunch of other input parameters are stored in a config file under `models/config.ini`.
Please read it carefully as it contains a lot of information critical for understanding of the workflow.


For Flask-app to use Bootstrap 4, install `bootstrap-flask`, not `flask-bootstrap`

## TODOs:

- ~~Visualization - tie `wav`, `rttm` together, create series of pngs~~. -- done by __Anton__
- ~~Tie the `png` together into a video, concatenate it with the original video of Zuck fighting with Cruz~~
- ~~Get the `VoxCeleb`/`VoxCeleb2` data, calculate all the embeddings, store them with labels~~ (done by __Anton__), **OR**
- ~~Using all above, select an appropriate model to be used as a feature extractor, and get embeddings of a given `.wav` 
in a sliding window fashion~~ -- done by __Dan__
- ~~Figure out how to tie metadata, and pull out Male/Female ids~~ - done by __Dan__
- ~~Set up KNN in the embedding space with an appropriate metric / pick `top_n` from the prediction vector~~ - done by __Dan & Anton__
- ~~Develop an interactive web-friendly visualization~~ - done by __Anton__
- ~~Apply Speaker Classification to `.wav` in a rolling window, get top N predictions~~ - done by __Dan__
- Build a classifier of masculinity/femininity (could be simply an average of top 10 / N predictions' gender)
- ~~Wrap everything into a nice set of functions~~


## TODOs for Anton:

- Tie a microphone recording module to the visualization via Bokeh server app
- Start working on a Flask application and think about how to deploy it 
 

## TODOs for Dan:

- Create a CLI wrapper for the set of scripts, that takes a path to an audiofile as an input argument, or a name of a preset `.wav` file, and launches a `bokeh` application locally on the user's machine. Rely on `click` package for settting up arguments

- Create a quick `gif` animation of running the sript in the terminal - use https://github.com/faressoft/terminalizer to help you create a `gif`. Gif would show the following:
    - Clone repo,
    - Run CLI wrapper with a `--help` parameter, that would show some info about the script usage, as well as possible values for preset examples
    - `ls` in the terminal for a custom `wav`
    - Run CLI on some custom `wav` file

- Clean this README, provide a nice overview of what the project does and how it achieves the goal - mention steps like:
    - VAD for voice activation detection
    - Speech Activity
    - Diarization clustering
    - VoxCeleb v1 dataset, metadata example, how the validation data was prepared (point to the right folder in the repo that has the codes, and the weights) -- this step is quite important actually, as people might be interested in downloading our weights
    - How the embeddings and the final layer were trained


## What did not work

- Figuring out if we can avoid downloading data and just use a pre-trained model from somewhere that produces 
    **embeddings with similarity property** -- **did not work, so we share our own models here trained on VoxCeleb1**


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

## How to use split.py for speaker prediction:

- Initialize arguments and params as seen in `split_test.ipynb`
- Create network using `network = src.vggvoxvlad.split.make_network(weight_path, args, input_dim=(257, None, 1), num_class=1251)` where `weight_path` is a `.h5` file
- Create a list of dataframes using `result_list = src.vggvoxvlad.split.voxceleb1_split(path, network, split_seconds=3, n=3,win_length=400, sr=16000, hop_length=160,n_fft=512, spec_len=250, n_classes=1251)` where `path` is a `.wav` file
- Each dataframe contains the headers `Time (s)`,     `Speaker`, `Probability`, `Country` and  `Gender`
- Each dataframe will display the three speakers with the highest predicted probability (to change this, change the `n` parameter)
- `voxceleb1_split()` will predict a speaker every three seconds by default (to change this, change the `split_seconds` parameter)
- To show a bargraph of the results, use `plot_split(result_list, num_speakers=2)` where `num_speakers` is the actual number of speakers in the `.wav` file


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
