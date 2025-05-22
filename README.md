# VitaGraph

This repository contains the codebase and resources for the paper **VitaGraph: Building a Knowledge Graph for Biologically Relevant Learning Tasks**.

## 📦 VitaGraph

To generate VitaGraph allow the script vitagraph.sh to be executable and then run it:

   ```bash
   chmod +x vitagraph.sh
   ./vitagraph.sh
   ```

Before running the script make sure to have downloaded the auxiliary.zip archive from VitaGraph's Kaggle webpage.

This will:

- Download the original drkg.tar.gz archive from the official DGL repository.
- Extract its contents to the dataset/ directory.
- Automatically remove the compressed archive and unnecessary files such as:
  - embed/, relation_glossary.tsv, entity2src.tsv
- Extract the auxiliry files present in auxiliary.zip
- Generate Vitagraph

## 🏋️‍♀️ Models training and testing

In ablation_study.py is present the code used to execute the ablation study on all the possible datasets versions, tasks, and models.

## 🔧 Requirements

run:

  ```bash
    pip install -r requirements.txt
   ```
