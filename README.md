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
- Extract the auxiliary files present in auxiliary.zip
- Generate Vitagraph

## 🏋️‍♀️ Models training and testing

In ablation_study.py is present the code used to execute the ablation study on all the possible datasets versions, tasks, and models.
To run a specific experiment on a dataset and a given task run: 
```
  python ablation_study.py -t <Insert task here> -d <insert dataset here> -m <insert model here> --patience 100 --runs 1 --quiet
   ```
e.g.,
```
  python ablation_study.py -t "Compound-Gene,Gene-Compound" -d "vitagraph" -m "rgcn" --patience 10 --runs 1 --quiet
   ```

If you want to run **ALL** experiments execute the run_all_ablation.sh script:
```
  bash run_all_ablation.sh
   ```

## 🔧 Requirements

run:

  ```bash
    pip install -r requirements.txt
   ```
