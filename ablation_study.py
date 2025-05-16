################# SHHHH MAN-AT-WORK ####################
#           _
#     _____|~~\_____      _____________
# _-~               \    |    \
# _-    | )     \    |__/   \   \
# _-         )   |   |  |     \  \
# _-    | )     /    |--|      |  |
# __-_______________ /__/_______|  |_________
# (                |----         |  |
# `---------------'--\\\\      .`--'
#                             `||||
import warnings
warnings.simplefilter(action='ignore')

import os
import json
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.utils import set_seed, load_data, select_target_triplets, entities2id_offset, rel2id_offset, edge_ind_to_id, entities_features_flattening,\
                      set_target_label, triple_sampling, graph_to_undirect, negative_sampling, add_self_loops, evaluation_metrics

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn

BASE_SEED = 42

models_params_path = './configurations/models_params.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(dataset_version, task, validation_size, test_size, quiet, seed):  

  features_paths = {}
  if dataset_version == 'vitagraph':
    edge_index_path = './dataset/vitagraph/vitagraph.tsv'
    features_paths = {
      'Gene': './dataset/vitagraph/gene_features.csv',
      'Compound': './dataset/vitagraph/cmp_features.csv'
    }
  elif dataset_version == 'vitagraph_no_features':
    edge_index_path = './dataset/vitagraph_no_features/vitagraph_no_features.tsv'
  elif dataset_version == 'drkg':
    edge_index_path = './dataset/drkg/drkg.tsv'
  
  # features_paths = {
  #   'Gene': gene_features_path,
  #   'Compound': compound_features_path
  # }

  edge_index, node_features_per_type = load_data(
    edge_index_path,
    features_paths,
    quiet
  )

  ent2id, all_nodes_per_type  = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id                 = rel2id_offset(edge_index)
  indexed_edge_index          = edge_ind_to_id(edge_index, ent2id, relation2id)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)
  indexed_edge_index          = set_target_label(indexed_edge_index, [ x for x in task.split(',')])

  non_target_triplets, target_triplets        = select_target_triplets(indexed_edge_index)
  train_triplets, val_triplets, test_triplets = triple_sampling(
                                                                target_triplets.loc[:,["head","interaction","tail"]].values, 
                                                                validation_size, 
                                                                test_size, 
                                                                quiet,
                                                                seed
                                                              )

  non_target_triplets = non_target_triplets.loc[:,["head","interaction","tail"]].values

  # print(f" non target {non_target_triplets.shape}")
  train_index = np.concatenate([non_target_triplets,train_triplets], axis=0)
  
  # print(f" target triplets{target_triplets.shape}")
  num_nodes_per_type = {node_type:len(nodes) for node_type,nodes in all_nodes_per_type.items()}

  num_entities = 0
  for features in num_nodes_per_type.values():
      num_entities += features

  # print(f" non target+target {train_index.shape}")
  train_index = graph_to_undirect(train_index, len(relation2id))
  # print(f" undirected {train_index.shape}")
  train_index = add_self_loops(train_index, num_entities, len(relation2id))
  # print(f" loops {train_index.shape}")
  
  train_triplets, val_triplets, test_triplets = torch.tensor(train_triplets), torch.tensor(val_triplets), torch.tensor(test_triplets)
  
  train_val_triplets = torch.cat([train_triplets, val_triplets], axis=0)
  train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], axis=0)

  train_val_triplets = train_val_triplets.to(device)
  train_val_test_triplets = train_val_test_triplets.to(device)
        
  in_channels_dict = {node_type:(features.shape[1] if features!= None else None) for node_type, features in flattened_features_per_type.items() }
  num_relations = len(relation2id)

  return in_channels_dict, num_nodes_per_type, num_entities, num_relations, train_triplets, train_index, flattened_features_per_type, val_triplets, train_val_triplets, test_triplets, train_val_test_triplets

def get_model(model_name, dataset_version, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations):
  with open(models_params_path, 'r') as f:
    models_params = json.load(f)
  model_params = models_params[task][dataset_version][model_name]

  conv_hidden_channels = {f'layer_{x}':model_params[f'layer_{x}']  for x in range(model_params['conv_layer_num'])} 

  if model_name == 'rgcn':
    model = rgcn(
      in_channels_dict,
      None,
      model_params['mlp_out_layer'],
      conv_hidden_channels,
      num_nodes_per_type,
      num_entities,
      num_relations+1,
      model_params['conv_layer_num'],
      model_params['num_bases'],
      activation_function=F.relu,
      device=device
    )
  elif model_name == 'rgat':
    model = rgat(
      in_channels_dict,
      None,
      model_params['mlp_out_layer'],
      conv_hidden_channels,
      num_nodes_per_type,
      num_entities,
      num_relations+1,
      conv_num_layers=model_params['conv_layer_num'],
      num_bases=model_params['num_bases'],
      activation_function=F.relu,
      device=device
    )
  elif model_name == 'compgcn':
    model = compgcn(
      in_channels_dict,
      mlp_out_emb_size=model_params['mlp_out_layer'],
      conv_hidden_channels=conv_hidden_channels,
      num_nodes_per_type=num_nodes_per_type,
      num_entities=num_entities,
      num_relations=num_relations,
      dropout=model_params['dropout'],
      conv_num_layers=model_params['conv_layer_num'],
      opn=model_params['opn'],
    )
  else:
    return None
  
  return model, model_params['learning_rate'], model_params['regularization'], model_params['grad_norm']

def train(model, optimizer, gradnorm, reg_param, x_dict, index , target_triplets, target_labels, change_points=None):
  metrics = {"Auroc":{},"Auprc":{},"Loss":{}}
  model.train()
  optimizer.zero_grad()
  if change_points != None:
    out = model(x_dict, index, change_points)
  else:
    out = model(x_dict, index)
  
  scores  = model.distmult(out, target_triplets)
  loss    = model.score_loss(scores, target_labels)
  # print(loss.item())
  reg_loss = (reg_param * model.reg_loss(out, target_triplets))
  loss =  loss  + reg_loss
  scores = torch.sigmoid(scores)

  metrics["Loss"]   = loss.item()
  metrics["Auroc"]  = binary_auroc(scores, target_labels).item()
  metrics["Auprc"]  = binary_auprc(scores, target_labels).item()

  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), gradnorm)
  optimizer.step()

  return metrics

def test(model, reg_param, x_dict, index , target_triplets, target_labels, train_val_triplets, change_points=None):    
  metrics = {"Auroc":{},"Auprc":{},"Loss":{}}

  model.eval()
  with torch.no_grad():
    if change_points != None:
      out = model(x_dict, index, change_points)
    else:
      out = model(x_dict, index)
    scores = model.distmult(out, target_triplets)
    loss = model.score_loss(scores, target_labels)
    reg_loss = (reg_param * model.reg_loss(out, target_triplets))
    loss =  loss  + reg_loss
    scores = torch.sigmoid(scores)
  
  mrr,hits = evaluation_metrics(model, out, train_val_triplets, target_triplets[target_labels.to(torch.int).view(-1)], 20, 0, hits=[1,3,10])

  metrics["Loss"]   = loss.item()
  metrics["MRR"]    = mrr
  metrics["Hits@"]  = hits
  metrics["Auroc"]  = binary_auroc(scores, target_labels).item()
  metrics["Auprc"]  = binary_auprc(scores, target_labels).item()
  
  return metrics

def main(model_name, dataset_version, task, runs, epochs, patience, validation_size, test_size, quiet, evaluate_every, negative_rate, model_save_path):
  all_run_metrics = []
  for i in range(runs):
    run_model_save_path = model_save_path.replace(".pt", f"_run{i}.pt")
    random_seed = BASE_SEED + i
    # set_seed(random_seed)
    if not quiet: print(f'[i] Using random seed {random_seed}')
    
    # Get dataset
    if not quiet: print('[i] Getting dataset...', end='', flush=True)
    start_time_dataset = time.time()
    in_channels_dict, num_nodes_per_type, num_entities, num_relations, \
    train_triplets, train_index, flattened_features_per_type, val_triplets, \
    train_val_triplets, test_triplets, train_val_test_triplets = get_dataset(dataset_version, task, validation_size, test_size, quiet, random_seed)
    end_time_dataset = round(time.time() - start_time_dataset, 2)
    if not quiet: print(f'ok ({end_time_dataset} seconds)')
    
    # Model definition
    model, lr, regularization, grad_norm = get_model(model_name, dataset_version, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations)

    # Move to device
    model                       = model.to(device)
    train_index                 = train_index.to(device)
    flattened_features_per_type = {node_type:(features.to(device) if features!= None else None) for node_type,features in flattened_features_per_type.items()}
    
    change_points = None
    if model_name == 'rgat':
      rel_ids_sorted = train_index[:, 1]
      change_points = torch.cat([
          torch.tensor([0], device=device),
          (rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
          torch.tensor([rel_ids_sorted.size(0)], device=device) 
      ])
      
    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Training runs
    torch.autograd.set_detect_anomaly(True)
    val_metrics       = {"Auroc":0, "Auprc":0, "Loss":0, "MRR":0, "Hits@":0}
    patience_trigger  = 0
    best_mixed_metric = 0
    best_model_found  = False
    with trange(1, (epochs + 1), desc=f'Run {i} | Epochs', position=0) as epochs_tqdm:
      for epoch in epochs_tqdm:
        training_triplets, train_labels = negative_sampling(train_triplets, negative_rate)
        training_triplets, train_labels = training_triplets.to(device), train_labels.to(device)
        # Train
        train_metrics = train(
            model,
            optimizer, 
            grad_norm, 
            regularization,
            flattened_features_per_type,
            train_index,
            training_triplets,
            train_labels,
            change_points
          )
        # Validate
        if epoch%evaluate_every==0:
          validation_triplets, val_labels = negative_sampling(val_triplets, negative_rate)
          validation_triplets, val_labels = validation_triplets.to(device), val_labels.to(device)
          val_metrics = test(
            model,
            regularization,
            flattened_features_per_type,
            train_index,
            validation_triplets,
            val_labels,
            train_val_triplets,
            change_points
          )
        
        mixed_metric = 0.2*val_metrics["Auroc"] + 0.4*val_metrics["Auprc"] + 0.4*val_metrics["MRR"]
        if mixed_metric > best_mixed_metric:
          best_mixed_metric = mixed_metric
          patience_trigger = 0
          torch.save(model.state_dict(), run_model_save_path)
          best_model_found = True
        else:
          patience_trigger += 1
        if patience_trigger > patience:
          break
        
        epochs_tqdm.set_postfix( loss=train_metrics["Loss"], Tr_Auroc=train_metrics["Auroc"], Tr_Auprc=train_metrics["Auprc"],
                                     Val_Auroc=val_metrics["Auroc"], Val_Auprc=val_metrics["Auprc"], Val_mrr=val_metrics["MRR"], Val_hits=val_metrics["Hits@"])
      
      # Load best model
      if best_model_found:
        model.load_state_dict(torch.load(run_model_save_path))
      # Test best model
      model.eval()
      with torch.no_grad():    
          testing_triplets, test_labels = negative_sampling(test_triplets, negative_rate)
          testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)
          metrics = test(
            model,
            regularization,
            flattened_features_per_type,
            train_index,
            testing_triplets,
            test_labels,
            train_val_test_triplets,
            change_points
          )  
      print(f"Run {i} | Test Auroc: {metrics['Auroc']:.3f}, Test Auprc: {metrics['Auprc']:.3f}, Test MRR: {metrics['MRR']:.3f}, TEST HITS: {metrics['Hits@']}")
      all_run_metrics.append({
          "Auroc": metrics["Auroc"],
          "Auprc": metrics["Auprc"],
          "MRR": metrics["MRR"],
          "Hits@1": metrics["Hits@"][1],
          "Hits@3": metrics["Hits@"][3],
          "Hits@10": metrics["Hits@"][10]
      })
  
  metric_keys = ["Auroc", "Auprc", "MRR", "Hits@1", "Hits@3", "Hits@10"]
  avg_metrics = {key: float(np.mean([m[key] for m in all_run_metrics])) for key in metric_keys}
  std_metrics = {key: float(np.std([m[key] for m in all_run_metrics])) for key in metric_keys}

  # Salvataggio su JSON
  result_to_save = {
      "individual_runs": all_run_metrics,
      "average_metrics": avg_metrics,
      "std_metrics": std_metrics
  }
  json_save_path = model_save_path.replace(".pt", "_metrics.json")
  with open(json_save_path, "w") as f:
      json.dump(result_to_save, f, indent=4)
  # print(f"Saved metrics to {json_save_path}")

  return None

if __name__ == '__main__':
  print(f'[i] Running on {device}')

  parser = argparse.ArgumentParser(description='Ablation study on Vitagraph generation process.')
  parser.add_argument('-t', '--task', type=str, default='Compound-SideEffect', choices=['Compound-Gene,Gene-Compound', 'Compound-SideEffect', 'Gene-Gene'], help='Task to perform in the ablation study. SE: side effect prediction, GGI: gene-gene interaction prediction, DR: drug repurposing.')
  parser.add_argument('-d', '--dataset', type=str, default='vitagraph', choices=['drkg', 'vitagraph_no_features', 'vitagraph'], help='Dataset to use for the ablation study.')
  parser.add_argument('-m', '--model', type=str, default='rgcn', choices=['rgcn', 'rgat', 'compgcn'], help='Model to use for the ablation study.')
  parser.add_argument('-r', '--runs', type=int, default=10, help='Number of runs for the ablation study.')
  parser.add_argument('-e', '--epochs', type=int, default=400, help='Number of epochs for the ablation study.')
  parser.add_argument('-p', '--patience', type=int, default=200, help='Patience trigger.')
  parser.add_argument('--validation_size', type=float, default=0.1, help='Validation size for the ablation study.')
  parser.add_argument('--test_size', type=float, default=0.2, help='Test size for the ablation study.')
  parser.add_argument('--quiet', action='store_true', help='If set, the ablation study will print debug output.')
  parser.add_argument('--evaluate_every', type=int, default=5, help='Evaluate every n epochs.')
  parser.add_argument('--negative_rate', type=float, default=1, help='Negative sampling rate for the ablation study.')

  args            = parser.parse_args()
  model           = args.model
  dataset         = args.dataset
  task            = args.task
  runs            = args.runs
  epochs          = args.epochs
  patience        = args.patience
  validation_size = args.validation_size
  test_size       = args.test_size
  quiet           = args.quiet
  evaluate_every  = args.evaluate_every
  negative_rate   = args.negative_rate

  task_clean      = task.lower().replace('-', '_').replace(',', '_')
  model_save_path = os.path.join('ablation_models', dataset, task_clean, f'{model.lower()}.pt')
  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

  if not quiet: print(f'Running ablation study with model: {model}, dataset: {dataset}, task: {task}, runs: {runs} | device: {device}')
  main(model, dataset, task, runs, epochs, patience, validation_size, test_size, quiet, evaluate_every, negative_rate, model_save_path)
  