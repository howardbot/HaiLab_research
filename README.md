# HaiLab Neural Decoding Project
This project aims to decode neural activity from two brain areas (CIP and V3A) using multi-unit electrophysiological recordings. 
The main goal is to train classifiers that can distinguish either the type of visual stimulus presented (e.g., slant) or the brain region (CIP vs V3A) from neural response data by using SVM.
## Project Structure

The `data/` folder contains `.mat` files representing neural recording sessions from two brain regions: CIP and V3A. Each `.mat` file stores a list of trials in the `TInfo_new` structure, which includes spike times (per neuron per tetrode), event timestamps (`EventT`), and stimulus information such as slant, tilt, and fixation distance.

The `src/` directory holds all the core processing modules. `Loader.py` handles the loading and parsing of `.mat` files, converting them into usable Python objects. `Features.py` is responsible for extracting firing rate features by calculating spike counts in a fixed time window (default: 0–200ms post-stimulus). `Models.py` implements a classification pipeline using a linear SVM, including standardized preprocessing, cross-validation, and confusion matrix visualization.

The `main.py` script orchestrates the entire decoding process. It loads the neural data, extracts features, assigns labels based on file names (e.g., CIP vs V3A), and trains a classifier. Additionally, the project includes a MATLAB script, `createSpikeMatrix.m`, which constructs a binned spike matrix from the raw data. This is useful if you wish to apply time-series models like LSTMs or convolutional networks.


