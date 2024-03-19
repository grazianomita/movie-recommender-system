# Movie Recommender System

This Python project implements a **content-based movie recommender system** using word embeddings from **Hugging Face 
pipelines** and **Faiss** for efficient semantic search. Raw data, embeddings and indexes are provided into the `data` 
directory for faster setup.

- Movies raw data is downloaded from: https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
- Be aware that computing the embeddings from scratch (eventually on a new raw movies dataset) might take significant time.
- Alternative embedding models, such as those provided by OpenAI, are also a valid choice.
- Alternative indexes like Pinecone or pgvector can replace Faiss depending on specific requirements and preferences.


## Getting Started

### Prerequisites

- Python 3.6+
- Pip (Python package manager)

### Installation

1. Clone the repository:

   ```bash
   foo@bar:~$ git clone <repo_url>
   foo@bar:~$ cd movie_recommender_system
   ```

2. Create a virtual environment and activate it:

   ```bash
   foo@bar:~$ python3 -m venv venv
   foo@bar:~$ source venv/bin/activate  # Linux/macOS
   # Or, for Windows:
   # venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   foo@bar:~$ pip install -r requirements.txt
   ```

### Create and configure a virtual environment

```bash
   foo@bar:~$ python -m venv venv
   foo@bar:~$ source venv/bin/activate
   foo@bar:~$ pip install -r requirements.txt
```

## Usage

### Running the Application

You can interactively run the code via the jupyter notebook.
You can also run the code via `main.py`. 

#### Input parameters

You can check the required parameters by running the following command:

```bash
   foo@bar:~$ python main.py --help
```

#### Running example

```bash
   foo@bar:~$ python main.py --query "<MovieTitle><MovieSummary>" -k 5 --data-dir "data" --embeddings-path "data/embeddings.npy" --faiss-index-path "data/movies.index"
```
You can also try to get recommendations with a custom query message, but expect results to be less accurate.

### Running Tests

To run tests using pytest be sure you are in the root directory of the project and execute:

```bash
foo@bar:~$ pytest
foo@bar:~$ pytest --cov=. --cov-report=html
```

### Generate documentation

```bash
foo@bar:~$ pip install sphinx
foo@bar:~$ pip install sphinx-rtd-theme
foo@bar:~$ mkdir docs
foo@bar:~$ cd docs
foo@bar:~$ sphinx-quickstart (separate build n)
foo@bar:~$ cd ..
foo@bar:~$ sphinx-apidoc -o docs src
foo@bar:~$ cd docs
foo@bar:~$ sphinx-build -b html -v . _build
```
