#!/bin/bash
set -e

combinations=(
  "Compound-SideEffect vitagraph_no_features compgcn"
  "Compound-SideEffect vitagraph rgcn"
  "Compound-SideEffect vitagraph rgat"
  "Compound-SideEffect vitagraph compgcn"
  "Gene-Gene drkg rgcn"
  "Gene-Gene drkg rgat"
  "Gene-Gene drkg compgcn"
  "Gene-Gene vitagraph_no_features rgcn"
  "Gene-Gene vitagraph_no_features rgat"
  "Gene-Gene vitagraph_no_features compgcn"
  "Gene-Gene vitagraph rgcn"
  "Gene-Gene vitagraph rgat"
  "Gene-Gene vitagraph compgcn"
)

for combo in "${combinations[@]}"; do
  read -r task dataset model <<< "$combo"
  echo "🚀 Running > Task=$task | Dataset=$dataset | Model=$model"
  python ablation_study.py -t "$task" -d "$dataset" -m "$model" --patience 10 --runs 2 --quiet
  echo "✅ Completed"
done