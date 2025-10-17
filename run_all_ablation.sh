#!/bin/bash

set -e

tasks=("Compound-Gene,Gene-Compound" "Compound-SideEffect" "Gene-Gene")
datasets=("drkg" "vitagraph_no_features" "vitagraph")
models=("rgcn" "rgat" "compgcn")

for task in "${tasks[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      echo "🚀 Running > Task=$task | Dataset=$dataset | Model=$model"
      python ablation_study.py -t "$task" -d "$dataset" -m "$model" --patience 100 --runs 10 --quiet
      echo "✅ Completed"
    done
  done
done