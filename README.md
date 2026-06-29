# Nepali Transliterator (NepaliXlit)

Adaptation of [IndicXlit](https://github.com/AI4Bharat/IndicXlit) for Nepali systems. This repository contains both a web application and a command-line interface (CLI) for performing Nepali transliteration.

---

## Tooling and Setup

The project is configured using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver. The project targets **Python 3.10** to maintain compatibility with deep learning libraries like `fairseq` and `torch`.

### Prerequisites

Install `uv` (if you haven't already):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To initialize the virtual environment and install all dependencies:
```bash
uv sync
```

---

## Running the Web Application

### Option 1: Running Locally (Recommended)

1. Sync the environment and install packages:
   ```bash
   uv sync
   ```

2. Run the application server:
   ```bash
   uv run python app/app.py
   ```
   *Your app will be live at `http://127.0.0.1:8000`*
   *Note: On the first run, the app will automatically download the required models and initialize the system. This may take a few minutes.*

### Option 2: Running with Docker

You can package and run the application inside a container using the provided Dockerfile which installs packages using standard pip.

1. Build the Docker Image:
   ```bash
   docker build -t nepalixlit .
   ```

2. Run the Docker Container:
   ```bash
   docker run -p 8000:8000 nepalixlit
   ```
   *Your app will be live at `http://127.0.0.1:8000`*

---

## Running the CLI Inference

You can run batch or single transliterations directly from the command line using the scripts in the `cli` folder.

1. Navigate to the CLI directory:
   ```bash
   cd cli
   ```

2. Download the models and dictionaries:
   ```bash
   # Download the model files
   uv run gdown 1v0RQU9BMhQJNzsesp_2BN1sImCbdw77d --output nepalixlit-en-ne.zip
   unzip nepalixlit-en-ne.zip

   # Download the Unigram dictionaries for reranking from the IndicXlit release
   wget https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/word_prob_dicts.zip
   unzip word_prob_dicts.zip
   ```

3. Create the input/output directories and save words:
   ```bash
   mkdir -p source output
   echo "namaste k xa haalkhabar" > source/input.txt
   ```

4. Run the inference script with `uv`:
   ```bash
   # Parameters:
   # -l = Language code (e.g., 'ne')
   # -i = Path to input file
   # -b = Beam size
   # -n = Number of best candidates
   # -r = Rerank flag (1 or 0)
   uv run bash transliterate_sentence.sh -l 'ne' -i 'source/input.txt' -b 5 -n 5 -r 1
   ```

5. View the output:
   ```bash
   cat output/final_transliteration.txt
   ```

---

## Bug Fixes and Details

- **Fixed TensorFlow Addons (TFA) EOL issues**:
  - The repository previously depended on `urduhack` which transitively required `tensorflow` and `tensorflow-addons`. Since `tensorflow-addons` reached End-of-Life (see [TensorFlow Addons Issue #2807](https://github.com/tensorflow/addons/issues/2807)), it caused installation failures.
  - We removed `urduhack` and replaced the Shahmukhi normalizer with a zero-dependency, pure-python character translation mapping inside [base_engine.py](file:///Users/pratham/Documents/GitHub/NepaliXlit/app/transliteration/transformer/base_engine.py). This completely bypassed any need for TensorFlow or TFA, resulting in faster load times and smaller footprints.
  - The custom normalization mapping is sourced directly from the original `urduhack` repository files:
    - [character.py](https://github.com/urduhack/urduhack/blob/master/urduhack/normalization/character.py)
    - [urdu_characters.py](https://github.com/urduhack/urduhack/blob/master/urduhack/urdu_characters.py)
- **Fixed PyTorch 2.6+ Checkpoint Loading**:
  - Configured PyTorch to support loading the fairseq checkpoint containing custom python objects under PyTorch 2.6+ by disabling pickle restrictions during loading.
- **Fixed TemplateResponse Signature**:
  - Updated the Starlette `TemplateResponse` call signature in [app.py](file:///Users/pratham/Documents/GitHub/NepaliXlit/app/app.py) to match Starlette 1.0.0+ specification (passing `request` as the first argument) to prevent dictionary hashing type errors.
