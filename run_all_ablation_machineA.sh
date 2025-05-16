#!/bin/bash
set -e

combinations=(
  "Compound-Gene,Gene-Compound drkg rgcn"
  "Compound-Gene,Gene-Compound drkg rgat"
  "Compound-Gene,Gene-Compound drkg compgcn"
  "Compound-Gene,Gene-Compound vitagraph_no_features rgcn"
  "Compound-Gene,Gene-Compound vitagraph_no_features rgat"
  "Compound-Gene,Gene-Compound vitagraph_no_features compgcn"
  "Compound-Gene,Gene-Compound vitagraph rgcn"
  "Compound-Gene,Gene-Compound vitagraph rgat"
  "Compound-Gene,Gene-Compound vitagraph compgcn"
  "Compound-SideEffect drkg rgcn"
  "Compound-SideEffect drkg rgat"
  "Compound-SideEffect drkg compgcn"
  "Compound-SideEffect vitagraph_no_features rgcn"
  "Compound-SideEffect vitagraph_no_features rgat"
)

for combo in "${combinations[@]}"; do
  read -r task dataset model <<< "$combo"
  echo "🚀 Running > Task=$task | Dataset=$dataset | Model=$model"
  python ablation_study.py -t "$task" -d "$dataset" -m "$model" --quiet
  echo "✅ Completed"
done