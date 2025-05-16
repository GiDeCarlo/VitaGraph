import gc
import os
import json
import pandas as pd
from os.path import isfile
from termcolor import colored
from time import perf_counter
from src.utils import extract_largest_connected_component, check_file, set_seeds, \
	read_yaml_config, read_drkg_df, stats, standardization, parse_args

set_seeds()

#*---------------------------------------------------------------------*
#|Restructure the interaction column in the DRKG 					 
#|The interaction column is divided in the columns: 				 
#|		source 	interaction type								     
#*---------------------------------------------------------------------*
def interaction_restructure(in_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found"

	drkg = pd.read_csv(in_fname, header = 0, names = ["head","interaction", "tail"], sep='\t',dtype=str)

	# drkg = drkg.loc[1:, :] # Remove header
	# Remove spaces from head and tail
	drkg["head"] = drkg["head"].astype(str).apply(standardization)
	drkg["tail"] = drkg["tail"].astype(str).apply(standardization)
	drkg["head"] = drkg["head"].astype(str).apply(lambda x: x.replace(" ",""))
	drkg["tail"] = drkg["tail"].astype(str).apply(lambda x: x.replace(" ",""))
	drkg["interaction"] = drkg["interaction"].astype(str)
	# Split interaction column into source, type and interaction columns
	drkg["source"] = drkg["interaction"].apply(lambda x: x.split("::")[0])

	drkg["type"] = drkg["head"].apply(lambda x: x.split("::")[0])+ "-" + drkg["tail"].apply(lambda x: x.split("::")[0])
	
	drkg["interaction"]   = drkg["interaction"].astype(str).apply(lambda x: x.split("::")[1] if len(x.split("::")[0]) == 3 else x.split("::")[1].split(":")[0])
	drkg["interaction"] = drkg["interaction"].astype(str).apply(lambda x: x.replace(" ","_"))
	drkg.to_csv(out_fname, sep='\t', index=False)

#*---------------------------------------------------------------------*
#|Removing duplicates 												 	 			  							 
#*---------------------------------------------------------------------*
#|Input: 															 												 			 
#| 			1. tsv filename of DRKG									 
# 			2. translation file path								 
#|			3. tsv filename where to save the cleaned version of DRKG
#*---------------------------------------------------------------------*
#|Output:															 												       
#|			1. number of rows translated			   				 
#*---------------------------------------------------------------------*
def translate_drugbank_genesid(in_fname, translation_file, out_fname):  # Add target_string argument
	assert isfile(in_fname), f"{in_fname} file not found"

	drkg = pd.read_csv(in_fname, sep='\t')
	translation = pd.read_csv(translation_file, names=["old_id", "new_id"], dtype=str)

	# 1. Create boolean masks to identify rows that need translation
	head_mask = drkg["head"].str.split(':').str[-1].isin(translation["old_id"])
	tail_mask = drkg["tail"].str.split(':').str[-1].isin(translation["old_id"])

	# 2. Extract IDs only for rows that need translation
	drkg.loc[head_mask, "head_id"] = drkg.loc[head_mask, "head"].str.split(':').str[-1]
	drkg.loc[tail_mask, "tail_id"] = drkg.loc[tail_mask, "tail"].str.split(':').str[-1]

	# 3. Merge for head translation (only on masked rows)
	drkg_head_masked = drkg.loc[head_mask].merge(translation, left_on="head_id", right_on="old_id", how="left")
	drkg.loc[head_mask, "new_head_id"] = drkg_head_masked["new_id"].values

	# 4. Merge for tail translation (only on masked rows)
	drkg_tail_masked = drkg.loc[tail_mask].merge(translation, left_on="tail_id", right_on="old_id", how="left")
	drkg.loc[tail_mask, "new_tail_id"] = drkg_tail_masked["new_id"].values

	# 5. Fill NaN with original IDs (only on masked rows)
	drkg.loc[head_mask, "head_id"] = drkg.loc[head_mask, "new_head_id"].fillna(drkg.loc[head_mask, "head"].str.split(':').str[-1])
	drkg.loc[tail_mask, "tail_id"] = drkg.loc[tail_mask, "new_tail_id"].fillna(drkg.loc[tail_mask, "tail"].str.split(':').str[-1])

	# 6. Reconstruct head and tail (only on masked rows)
	drkg.loc[head_mask, "head"] = drkg.loc[head_mask, "head"].str.split('::').str[0] + "::NCBI:" + drkg.loc[head_mask, "head_id"]
	drkg.loc[tail_mask, "tail"] = drkg.loc[tail_mask, "tail"].str.split('::').str[0] + "::NCBI:" + drkg.loc[tail_mask, "tail_id"]

	drkg.drop(columns=["head_id", "tail_id", "new_head_id", "new_tail_id"], inplace=True, errors='ignore')  # Drop temporary columns

	drkg.to_csv(out_fname, sep='\t', index=False)

	return sum(head_mask) + sum(tail_mask)

def map_disease_id(in_fname, mapping_fname, out_fname, map_to_db):

	drkg = pd.read_csv(in_fname, sep='\t')
	translation = pd.read_csv(mapping_fname, dtype=str)

	original_num_diseases = len([x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Disease' in x])

	# 1. Create boolean masks to identify rows that need translation
	drkg['head_source'] =  drkg["head"].str.split('::').str[-1].str.split(":").str[0]
	drkg['tail_source'] =  drkg["tail"].str.split('::').str[-1].str.split(":").str[0]

	translation_datasets = [x for x in translation.columns if x!=map_to_db]

	head_mask = (drkg["head"].str.contains('Disease')) & (drkg["head"].str.split('::').str[-1].isin(translation.values.reshape(-1)))
	tail_mask = (drkg["tail"].str.contains('Disease')) & (drkg["tail"].str.split('::').str[-1].isin(translation.values.reshape(-1)))

	# 2. Extract IDs only for rows that need translation
	drkg.loc[head_mask, "head_id"] = drkg.loc[head_mask, "head"].str.split('::').str[-1]
	drkg.loc[tail_mask, "tail_id"] = drkg.loc[tail_mask, "tail"].str.split('::').str[-1]

	for source in translation_datasets:
	# 3. Merge for head translation (only on masked rows)
		translation_no_dup = translation.drop_duplicates(subset=[source],keep='first') # assuming the dataset is correctly oredered
		drkg_head_masked = drkg.loc[(head_mask) & (drkg['head_source']==source)].merge(translation_no_dup, left_on="head_id", right_on=source, how="left")
		
		drkg.loc[(head_mask) & (drkg['head_source']==source), "new_head_id"] = drkg_head_masked[map_to_db].values

		# # 4. Merge for tail translation (only on masked rows)
		drkg_tail_masked = drkg.loc[(tail_mask) & (drkg['tail_source'] == source)].merge(translation_no_dup, left_on="tail_id", right_on=source, how="left")

		drkg.loc[(tail_mask) & (drkg['tail_source'] == source), "new_tail_id"] = drkg_tail_masked[map_to_db].values

	# # 5. Fill NaN with original IDs (only on masked rows)
	head_mask &= ~drkg.loc[head_mask, "new_head_id"].isna()
	tail_mask &= ~drkg.loc[tail_mask, "new_tail_id"].isna()

	num_mappable_diseases = len(set(drkg.loc[head_mask, "head"].unique()).union(set(drkg.loc[tail_mask, "tail"].unique())))
	# # 6. Reconstruct head and tail (only on masked rows)
	drkg.loc[head_mask, "head"] = drkg.loc[head_mask, "head"].str.split('::').str[0] + "::" + drkg.loc[head_mask, "new_head_id"]
	drkg.loc[tail_mask, "tail"] = drkg.loc[tail_mask, "tail"].str.split('::').str[0] + "::" + drkg.loc[tail_mask, "new_tail_id"]

	drkg.drop(columns=["head_id", "tail_id", "new_head_id", "new_tail_id",'head_source','tail_source'], inplace=True, errors='ignore')  # Drop temporary columns

	drkg.to_csv(out_fname, sep='\t',index=False)

	modified_num_diseases = len([x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Disease' in x])

	redundant_ids = original_num_diseases - modified_num_diseases

	affected_rows = sum((head_mask | tail_mask))

	return num_mappable_diseases, affected_rows, redundant_ids

def compounds_id_mapping(in_fname, mapping_fname, out_fname, map_to_db):
    # drkg = pd.read_csv('./drkg.tsv',sep='\t',names=["head",'interaction','tail'],header=0,dtype=str)
	drkg = pd.read_csv(in_fname, sep='\t',dtype=str)
	translation = pd.read_csv(mapping_fname, dtype=str).loc[:,['dataset_id', map_to_db]]

	original_num_compounds = len([x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Compound::' in x])
	# 1. Create boolean masks to identify rows that need translation
	head_mask = drkg["head"].str.split('::').str[-1].isin(translation["dataset_id"]) & drkg["head"].str.contains("Compound::")
	tail_mask = drkg["tail"].str.split('::').str[-1].isin(translation["dataset_id"]) & drkg["tail"].str.contains("Compound::")
	
	# 2. Extract IDs only for rows that need translation
	drkg.loc[head_mask, "head_id"] = drkg.loc[head_mask, "head"].str.split('::').str[-1]
	drkg.loc[tail_mask, "tail_id"] = drkg.loc[tail_mask, "tail"].str.split('::').str[-1]

	# 3. Merge for head translation (only on masked rows)
	drkg_head_masked = drkg.loc[head_mask].merge(translation, left_on="head_id", right_on="dataset_id", how="left")
	drkg.loc[head_mask, "new_head_id"] = drkg_head_masked[map_to_db].values

	# # 4. Merge for tail translation (only on masked rows)
	drkg_tail_masked = drkg.loc[tail_mask].merge(translation, left_on="tail_id", right_on="dataset_id", how="left")

	drkg.loc[tail_mask, "new_tail_id"] = drkg_tail_masked[map_to_db].values

	# # 5. Fill NaN with original IDs (only on masked rows)
	head_mask &= ~drkg.loc[head_mask, "new_head_id"].isna()
	tail_mask &= ~drkg.loc[tail_mask, "new_tail_id"].isna()

	drkg.loc[head_mask, "head_id"] = drkg.loc[head_mask, "new_head_id"].fillna(drkg.loc[head_mask, "head"].str.split('::').str[-1])
	drkg.loc[tail_mask, "tail_id"] = drkg.loc[tail_mask, "new_tail_id"].fillna(drkg.loc[tail_mask, "tail"].str.split('::').str[-1])

	num_mappable_cmps = len(set(drkg.loc[head_mask, "head"].unique()).union(set(drkg.loc[tail_mask, "tail"].unique())))
	# # 6. Reconstruct head and tail (only on masked rows)
	drkg.loc[head_mask, "head"] = drkg.loc[head_mask, "head"].str.split('::').str[0] + f"::{map_to_db.replace(' ','_')}:" + drkg.loc[head_mask, "head_id"]
	drkg.loc[tail_mask, "tail"] = drkg.loc[tail_mask, "tail"].str.split('::').str[0] + f"::{map_to_db.replace(' ','_')}:" + drkg.loc[tail_mask, "tail_id"]

	drkg.drop(columns=["head_id", "tail_id", "new_head_id", "new_tail_id"], inplace=True, errors='ignore')  # Drop temporary columns

	drkg.to_csv(out_fname,sep='\t',index=False)
	affected_rows = sum((head_mask | tail_mask))
	

	modified_num_compounds = len([x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Compound::' in x])

	print(f"Original number of compounds: {original_num_compounds}, Modified number of compounds: {modified_num_compounds}")
	duplicates_id = original_num_compounds - modified_num_compounds

	return num_mappable_cmps, affected_rows, duplicates_id 

#*---------------------------------------------------------------------*
#|Removing duplicates 													 			  						
#*---------------------------------------------------------------------*
#|Input: 																												 		
#| 			1. tsv filename of DRKG																			
#|			2. tsv filename where to save the cleaned version of DRKG		
#*---------------------------------------------------------------------*
#|Output:																												       
#|			1. Number of lines dropped																	
#*---------------------------------------------------------------------*
def drop_duplicates(in_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found"

	drkg = read_drkg_df(in_fname)
	n = len(drkg)
	drkg = drkg.drop_duplicates()
	# For each cell create a new column with lists as values where the list is the sorted head and tail
	# Rows with the same head and tail will have the same sorted list
	drkg['sorted_nodes'] = drkg[['head', 'tail']].apply(sorted, axis=1)       #create support column
	drkg['sorted_nodes'] = drkg['sorted_nodes'].apply(lambda x: x[0]+x[1])    #compress list into string

	# Drop duplicates based on the sorted nodes column and keep the first occurrence
	drkg = drkg.drop_duplicates(subset=['interaction', 'sorted_nodes'])

	# Drop the temporary column
	drkg = drkg.drop(columns=['sorted_nodes'])
	m = len(drkg)
	drkg.to_csv(out_fname, sep='\t', index=False)

	del drkg
	gc.collect()

	return n - m

#*---------------------------------------------------------------------*
#|Removing semicolons 												
#*---------------------------------------------------------------------*
#|Input: 															
#| 			1. tsv filename of DRKG									
#|			2. tsv filename where to save the cleaned version of DRKG  |
#*---------------------------------------------------------------------*
#|Output:															
#|			1. Number of lines removed							    
#*---------------------------------------------------------------------*
def remove_triplets_with_semicolon(in_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found"


	drkg = read_drkg_df(in_fname)
	filtered_drkg = drkg[~(drkg['head'].str.contains(';') | drkg['tail'].str.contains(';'))]
	removed = len(drkg) - len(filtered_drkg)
	filtered_drkg.to_csv(out_fname, sep='\t', index=False)

	return removed

#*---------------------------------------------------------------------*
#|Removing pipes 													
#*---------------------------------------------------------------------*
#|Input: 															
#| 			1. tsv filename of DRKG									
#|			2. tsv filename where to save the cleaned version of DRKG  |
#*---------------------------------------------------------------------*
#|Output:															
#|			1. Number of lines removed							    
#*---------------------------------------------------------------------*
def remove_triplets_with_pipe(in_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found"

	drkg = read_drkg_df(in_fname)
	filtered_drkg = drkg[~(drkg['head'].str.contains(r'\|MESH') | drkg['tail'].str.contains(r'\|MESH'))]
	removed = len(drkg) - len(filtered_drkg)
	filtered_drkg.to_csv(out_fname, sep='\t', index=False)

	return removed

#*---------------------------------------------------------------------*
#|Relations standardizations							 			
#*---------------------------------------------------------------------*
#|Input: 					                                        
#| 			1. tsv filename of DRKG									
#|			2. tsv filename where to save the cleaned version of DRKG  |
#*---------------------------------------------------------------------*
#|Output:															
#|			1. Number of lines cleaned								
#*---------------------------------------------------------------------*
def relation_standardization(in_fname, out_fname):
	'''
	All the relations that are joined:

	'TREATMENT':['T', 'treats', 'CtD', 'Treatment'],
	'ACTIVATOR':['A+', 'ACTIVATOR', 'AGONIST'],
	'BLOCKER':['ANTAGONIST', 'A-', 'BLOCKER', 'CHANNEL BLOCKER'],
	'CMP_BIND':['BINDER', 'CbG', 'target', 'DrugHumGen', 'B', 'DIRECT INTERACTION','ASSOCIATION','Interaction','PHYSICAL ASSOCIATION' ],
	'ENZYME':['enzyme','Z'],
	'EXPRESSION':['E','EXPRESSION'],
	'REGULATION':['Rg','Gr>G'],
	'UPREGULATION':['E+','CuG'],
	'DOWNREGULATION':['INHIBITOR', 'E-', 'N', 'CdG'],
	'GENE_BIND': ['BINDING', 'HumGenHumGen', 'GiG','B','PHYSICAL ASSOCIATION','ASSOCIATION','DIRECT INTERACTION','Interaction'],
	'J_c':['J'],
	'J_g':['J'],
	'gene_OTHER_cmp': [OTHER]
	'gene_OTHER_gene': [OTHER]
	'''
	assert isfile(in_fname), f"{in_fname} file not found."

	drkg = read_drkg_df(in_fname)
	n = len(drkg)

	print(f"Shape before cleaning: {drkg.shape}, tot relations: {drkg['interaction'].unique().shape}")

	######join: T, treats, CtD:
	for rel in ['T', 'treats', 'CtD', 'Treatment']:
		drkg.loc[drkg['interaction'] == rel, ('interaction')] = 'TREATMENT'

	######join: A+, ACTIVATOR, AGONIST:
	for rel in ['A+', 'ACTIVATOR', 'AGONIST']:
		drkg.loc[drkg['interaction'] == rel, ('interaction')] = 'ACTIVATOR'

	######join: ANTAGONIST, A-, BLOCKER, CHANNEL BLOCKER:
	for rel in ['ANTAGONIST', 'A-', 'BLOCKER', 'CHANNEL_BLOCKER']:
		drkg.loc[drkg['interaction'] == rel, ('interaction')] = 'BLOCKER'

	######join: BINDER, B, CbG, target:
	for rel in ['BINDER', 'CbG', 'target', 'DrugHumGen', ]:
		drkg.loc[drkg['interaction'] == rel, ('interaction')]= 'CMP_BIND'

	drkg.loc[(drkg['interaction'] == 'B') & ~((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'CMP_BIND'
	drkg.loc[(drkg['interaction'] == 'DIRECT_INTERACTION') & ~((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'CMP_BIND'
	drkg.loc[(drkg['interaction'] == 'ASSOCIATION') & ~((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'CMP_BIND'
	drkg.loc[(drkg['interaction'] == 'PHYSICAL_ASSOCIATION') & ~((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'CMP_BIND'
	drkg.loc[(drkg['interaction'] == 'Interaction') & ~((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'CMP_BIND'

	######join: enzyme, Z:
	drkg.loc[drkg['interaction'] == 'enzyme', ('interaction')] = 'ENZYME'
	drkg.loc[drkg['interaction'] == 'Z', ('interaction')] = 'ENZYME'

	######join: E, EXPRESSION:
	drkg.loc[(drkg['interaction'] == 'E') & ((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'EXPRESSION'
	drkg.loc[drkg['interaction'] == 'EXPRESSION', ('interaction')] = 'EXPRESSION'

	######join: Rg, Gr>G:
	drkg.loc[drkg['interaction'] == 'Rg', ('interaction')] = 'REGULATION'
	drkg.loc[drkg['interaction'] == 'Gr>G', ('interaction')] = 'REGULATION'

	######join: E+, CuG:
	drkg.loc[(drkg['interaction'] == 'E+') & ~((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'UPREGULATION'
	drkg.loc[drkg['interaction'] == 'CuG', ('interaction')] = 'UPREGULATION'

	#####join: INHIBITOR, E-, N, CdG:
	for rel in ['INHIBITOR', 'E-', 'N', 'CdG']:
		drkg.loc[drkg['interaction'] == rel, ('interaction')] = 'DOWNREGULATION'

	######join: B, BINDING, HumGenHumGen, GiG, PHYSICAL ASSOCIATION, DIRECT INTERACTION:
	for rel in ['BINDING', 'HumGenHumGen', 'GiG']:
		drkg.loc[drkg['interaction'] == rel, ('interaction')] = 'GENE_BIND'
	drkg.loc[(drkg['interaction'] == 'B') & ((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'GENE_BIND'
	drkg.loc[(drkg['interaction'] == 'PHYSICAL_ASSOCIATION') & ((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'GENE_BIND'
	drkg.loc[(drkg['interaction'] == 'ASSOCIATION') & ((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'GENE_BIND'
	drkg.loc[(drkg['interaction'] == 'DIRECT_INTERACTION') & ((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'GENE_BIND'
	drkg.loc[(drkg['interaction'] == 'Interaction') & ((drkg['head'].str.contains('Gene')) & (drkg['head'].str.contains('Gene'))), ('interaction')] = 'GENE_BIND'

	# GNBR::J::Compound:Disease	GNBR	Compound:Disease	role in disease pathogenesis
	# GNBR::J::Gene:Disease	GNBR	Disease:Gene	role in pathogenesis

	drkg.loc[(drkg['interaction'] =='J') & (drkg['head'].str.contains("Compound")), ('interaction')] = 'J_c'
	drkg.loc[(drkg['interaction'] =='J') & (drkg['head'].str.contains("Gene")), ('interaction')] = 'J_g'
	
	# DGIDB::OTHER:: Gene:Compound	DGIDB	Gene:Compound	
	# STRING::OTHER::Gene:Gene	STRING	Gene:Gene	
	drkg.loc[(drkg['interaction'] =='OTHER') & (drkg['tail'].str.contains("Compound")), ('interaction')] = 'gene_OTHER_cmp'
	drkg.loc[(drkg['interaction'] =='OTHER') & (drkg['tail'].str.contains("Gene")), ('interaction')] = 'gene_OTHER_gene'
	
	
	#--------------------------------------

	# # dropping relations with viruses, and therefore removeing any virus genes
	drkg = drkg.loc[(drkg['interaction'] != "VirGenHumGen") & (drkg['interaction'] != "DrugVirGen")]

	drkg = drkg.drop_duplicates(['head', 'interaction', 'tail'])
	m = len(drkg)

	print(f"Shape after cleaning: {drkg.shape}, tot relations: {drkg['interaction'].unique().shape}")

	drkg.to_csv(out_fname, sep='\t', index=False)  # Ensure no index is written and delimiter is tab

	del drkg
	gc.collect()

	return n - m

def to_remove(g, genes_to_remove):
	g = g.split(';')
	for e in g:
		if e in genes_to_remove:
			return True
	return False

#*---------------------------------------------------------------------*
#|Removing non-human genes 											
#*---------------------------------------------------------------------*
#|Input: 														    
#| 			1. drkg tsv 											
#|			2. csv with non human genese to remove from drkg		
#|			3. file name where to save the cleaned drkg				
#*---------------------------------------------------------------------*
#|Output:															
#|			1. number of lines and genes removed                    
#*---------------------------------------------------------------------*
def remove_non_human_genes(in_fname, non_human_genes_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found."
	assert isfile(non_human_genes_fname), f"{non_human_genes_fname} file not found."

	drkg = read_drkg_df(in_fname)

	orignal_len = len(drkg)
	drkg_genes = [x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Gene::' in x]
	original_num_genes = len(drkg_genes)
	
	non_human_genes = list(pd.read_csv(non_human_genes_fname)["GeneID"].unique())
	

	head_gene = (drkg['head'].str.split("::").str[0] != "Gene") | ~(drkg["head"].str.split(":").str[-1].isin(non_human_genes))
	tail_gene = (drkg['tail'].str.split("::").str[0] != "Gene") | ~(drkg["tail"].str.split(":").str[-1].isin(non_human_genes))

	
	drkg = drkg[head_gene & tail_gene]
	
	mod_len = len(drkg)
	drkg_genes = [x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Gene::' in x]
	mod_num_genes = len(drkg_genes)
	
	print(f"Original length: {orignal_len}, Modified length: {mod_len}")
	print(f"Original number of genes: {original_num_genes}, Modified number of genes: {mod_num_genes}")

	drkg.to_csv(out_fname, sep='\t', index=False)

	return orignal_len - mod_len, original_num_genes - mod_num_genes 

#*---------------------------------------------------------------------*
#|Removing compounds without SMILES 								
#|This can be due to the fact that the compounds are deprecated, etc.. |
#*---------------------------------------------------------------------*
#|Input:															
#|			1. drkg tsv												
#|			2. csv with compound_id, SMILES							
#|			3. file name where to save the cleaned drkg 			
#*---------------------------------------------------------------------*
#|Output:															
#|			1. number of lines and compounds removed                
#*---------------------------------------------------------------------*
def remove_compunds_without_smiles(in_fname, compound_smiles_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found."
	assert isfile(compound_smiles_fname), f"{compound_smiles_fname} file not found."

	drkg = read_drkg_df(in_fname)

	orignal_len = len(drkg)
	drkg_compounds = [x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Compound::' in x]
	original_num_compounds = len(drkg_compounds)
	
	compounds = pd.read_csv(compound_smiles_fname, dtype=str)
	compounds = compounds[(compounds["smiles"].notna()) & (compounds["morgan_fingerprints"] != 'ERROR')]
	compounds = compounds["cid"].values

	head_cmpunds = (drkg['head'].str.split("::").str[0] != "Compound") | (drkg["head"].str.split("::").str[-1].isin(compounds))
	tail_cmpunds = (drkg['tail'].str.split("::").str[0] != "Compound") | (drkg["tail"].str.split("::").str[-1].isin(compounds))

	
	drkg = drkg[head_cmpunds & tail_cmpunds]
	
	mod_len = len(drkg)
	drkg_compounds = [x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Compound::' in x]
	mod_num_compounds = len(drkg_compounds)
	print(f"Original length: {orignal_len}, Modified length: {mod_len}")
	print(f"Original number of compounds: {original_num_compounds}, Modified number of compounds: {mod_num_compounds}")

	drkg.to_csv(out_fname, sep='\t', index=False)

	return orignal_len - mod_len, original_num_compounds - mod_num_compounds 

def nodes_as_gene_features(in_fname, feature_node, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found."
	assert isfile(out_fname), f"{out_fname} file not found."

	# desc = f'Adding features from {entity_type}'

	drkg = read_drkg_df(in_fname)
	drkg_genes = [x for x in set(drkg["head"].values).union(set(drkg["tail"].values)) if 'Gene::' in x]

#*---------------------------------------------------------------------*
#|		 Incorporating nodes as features		  												
#*---------------------------------------------------------------------*
#| In the next section, nodes belonging to: 												
#| 	1. Pathways																											
#| 	2. Molecular Functions                  												
#| 	3. Cellular Components 																					
#| 	4. Biological Processes 																				
#| are added as features 																							
#*---------------------------------------------------------------------*
def add_genes_features_from_entity(in_fname, entity_type, in_features_file, out_features_file):
	assert isfile(in_fname), "Input file not found"
	assert isinstance(entity_type, str), "Entity type must be a string"
	# assert isfile(in_features_file), f'Features file {in_features_file} not found'

	# desc = f'Adding features from {entity_type}'

	entity_to_id = {}
	genes_entity_d = {}

	# Mapping entity to id
	with open(in_fname, 'r') as fin:
		_ = fin.readline() # Skip header
		for line in fin.readlines():
			parts = line.strip().split('\t')
			h = parts[0].split('::')
			t = parts[2].split('::')

			# if head is pathway
			if entity_type == h[0]:
				if h[1] not in entity_to_id:
					entity_to_id[h[1]] = len(entity_to_id)
			if entity_type == t[0]:
				if t[1] not in entity_to_id:
					entity_to_id[t[1]] = len(entity_to_id)
	
	n_entities = len(entity_to_id)

	print(colored(f'[i] {n_entities} different {entity_type}s', 'light_yellow' ))

	#################################################
	# Add n_entities sized binary vector as feature #
	#################################################

	with open(in_fname, 'r') as fin:
		_ = fin.readline() # skip header
		for line in fin.readlines():
			line = line.strip().split('\t')
			h = line[0]
			h_id = h.split('::')[1]
			t = line[2]
			t_id = t.split('::')[1]

			# Inizialize feature fector for each gene
			if 'Gene' in h:
				if h not in genes_entity_d:
					feature_vector = [0]*n_entities
					genes_entity_d[h] = {'id': h, entity_type: feature_vector}
					# genes_entity_d[h] = {}
					# genes_entity_d[h][entity_type] = feature_vector
				if entity_type in t:
					pathway_idx = entity_to_id[t_id]
					genes_entity_d[h][entity_type][pathway_idx] = 1
			if 'Gene' in t:
				if t not in genes_entity_d:
					feature_vector = [0]*n_entities
					genes_entity_d[t] = {'id': t, entity_type: feature_vector}
					# genes_entity_d[t] = {}
					# genes_entity_d[t][entity_type] = feature_vector
				if entity_type in h:
					pathway_idx = entity_to_id[h_id]
					genes_entity_d[t][entity_type][pathway_idx] = 1

	# print(genes_entity_d)
	new_features = pd.DataFrame.from_dict(genes_entity_d, orient='index')
	new_features = new_features.reset_index(drop=True)
	new_features[entity_type] = new_features[entity_type].apply(lambda x: ''.join(str(y) for y in x))

	# print(new_features)
	if not os.path.exists(in_features_file):
		new_features.to_csv(out_features_file, sep=',', index=False)
		return 0
	
	feature_file = pd.read_csv(in_features_file, sep=',',dtype=str)

	result = pd.merge(feature_file, new_features, how="inner", on=["id", "id"])
	# result = pd.concat([feature_file, new_features], axis=1, join="inner")

	result.to_csv(out_features_file, sep=',', index=False)

	##################################################################################################
	# Save Entity to id dictionary to save the position in the feature vector relative to the entity #
	##################################################################################################

	fname = f'Dataset/generated/{entity_type}_feature_vector_position_dict.json'

	with open(fname, 'w') as fout:
		json.dump(entity_to_id, fout)

	return 0

def add_genes_features_from_pathways(in_fname, in_features_file, out_features_file):
	entity_name = 'Pathway'
	return add_genes_features_from_entity(in_fname, entity_name, in_features_file, out_features_file)

def add_genes_features_from_molecular_functions(in_fname, in_features_file, out_features_file):
	entity_name = 'MolecularFunction'
	return add_genes_features_from_entity(in_fname, entity_name, in_features_file, out_features_file)

def add_genes_features_from_cellular_components(in_fname, in_features_file, out_features_file):
	entity_name = 'CellularComponent'
	return add_genes_features_from_entity(in_fname, entity_name, in_features_file, out_features_file)

def add_genes_features_from_biological_processes(in_fname, in_features_file, out_features_file):
	entity_name = 'BiologicalProcess'
	return add_genes_features_from_entity(in_fname, entity_name, in_features_file, out_features_file)

def remove_entity_nodes(in_fname, out_fname, entity_types):
	assert isfile(in_fname), "Input file not found"
	assert isinstance(entity_types, list), "Entity type must be a list"
	
	drkg = pd.read_csv(in_fname, sep='\t')
	starting_lenght = len(drkg)
	
	drkg = drkg[~drkg['head'].apply(lambda x: x.split("::")[0]).isin(entity_types) & ~drkg['tail'].apply(lambda x: x.split("::")[0]).isin(entity_types)]
	final_lenght = len(drkg)
	drkg.to_csv(out_fname, sep='\t', index=False)
	return starting_lenght - final_lenght  

def remove_hetionet_pathways(in_fname, out_fname):
	assert isfile(in_fname), f"{in_fname} file not found."
	

	drkg = read_drkg_df(in_fname)

	orignal_len = len(drkg)

	hetionet_pathways_mask = ~((drkg['source'] == 'Hetionet') & (drkg['head'].str.contains('Pathway') | drkg['tail'].str.contains('Pathway')))
	no_pathway_drkg = drkg.loc[hetionet_pathways_mask,: ]

	filtered_len = len(no_pathway_drkg)

	no_pathway_drkg.to_csv(out_fname, sep='\t', index=False)

	return orignal_len - filtered_len

def merge_dataset(dataset1, datasets, out_fname):
	df = pd.read_csv(dataset1, sep='\t')
	original_len = len(df)
	for dataset in datasets:
		df2 = pd.read_csv(dataset)
		df = pd.concat([df, df2])
	
	modified_len = len(df)

	df.to_csv(out_fname, sep='\t', index=False)

	return modified_len - original_len

if __name__ == '__main__':
	args = parse_args()
	params	= read_yaml_config(args.config_file)
	total_time	= 0
	overwrite		= params['overwrite']

	if params['interaction_restructure']['do']:
		in_fname	= params['interaction_restructure']['in_fname']
		out_fname = params['interaction_restructure']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[interaction_restructure] Started', 'light_green'))
			start = perf_counter()
			interaction_restructure(in_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[interaction_restructure] Finished in {end} seconds', 'light_green'))

	if params['remove_hetionet_pathways']['do']:
		dataset		= params['remove_hetionet_pathways']['in_fname']
		out_fname	= params['remove_hetionet_pathways']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[remove_hetionet_pathways] Started', 'light_green'))
			start = perf_counter()
			removed_lines = remove_hetionet_pathways(dataset, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[remove_hetionet_pathways] Finished in {end} seconds, removal of {removed_lines} lines', 'light_green'))

	if params['merge_datasets']['do']:
		dataset1	= params['merge_datasets']['dataset']
		datasets 	= params['merge_datasets']['to_add']
		# dataset2	= params['merge_dataset']['dataset2']
		out_fname   = params['merge_datasets']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[merge_dataset] Started', 'light_green'))
			start = perf_counter()
			added_lines = merge_dataset(dataset1, datasets, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[merge_dataset] Finished in {end} seconds, addition of {added_lines} lines', 'light_green'))

	if params['map_compounds_ids']['do']:
		in_fname	= params['map_compounds_ids']['in_fname']
		mapping_fname	= params['map_compounds_ids']['mapping_fname']
		map_to_db = params['map_compounds_ids']['map_to_db']
		out_fname   = params['map_compounds_ids']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[map_compounds_ids] Started', 'light_green'))
			start = perf_counter()
			compounds_translated, translated_rows, duplicated_compounds =  compounds_id_mapping(in_fname, mapping_fname, out_fname, map_to_db)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[map_compounds_ids] mapped {compounds_translated} compounds with {duplicated_compounds} redundant ids , {translated_rows} rows modified in {end} seconds', 'light_green'))
	
	if params['map_diseases_ids']['do']:
		in_fname	= params['map_diseases_ids']['in_fname']
		mapping_fname	= params['map_diseases_ids']['mapping_fname']
		map_to_db = params['map_diseases_ids']['map_to_db']
		out_fname   = params['map_diseases_ids']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[map_diseases_ids] Started', 'light_green'))
			start = perf_counter()
			diseases_translated, translated_rows, duplicated_diseases =  map_disease_id(in_fname, mapping_fname, out_fname, map_to_db)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[map_diseases_ids] mapped {diseases_translated} disease with {duplicated_diseases} redundant ids , {translated_rows} rows modified in {end} seconds', 'light_green'))
	
	if params['translate_ids']['do']:
		in_fname	= params['translate_ids']['in_fname']
		translation_dataset	= params['translate_ids']['translation_dataset']
		out_fname   = params['translate_ids']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[translate_ids] Started', 'light_green'))
			start = perf_counter()
			translated_rows =translate_drugbank_genesid(in_fname, translation_dataset, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[translate_ids] translated {translated_rows} rows in {end} seconds', 'light_green'))
	
	if params['remove_triplets_with_semicolon']['do']:
		in_fname  = params['remove_triplets_with_semicolon']['in_fname']
		out_fname = params['remove_triplets_with_semicolon']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[remove_triplets_with_semicolon] Started', 'light_green'))
			start = perf_counter()
			n = remove_triplets_with_semicolon(in_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[remove_triplets_with_semicolon] Finished | Removed {n} lines in {end} seconds', 'light_green'))

	if params['remove_triplets_with_pipe']['do']:
		in_fname  = params['remove_triplets_with_pipe']['in_fname']
		out_fname = params['remove_triplets_with_pipe']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[remove_triplets_with_pipe] Started', 'light_green'))
			start = perf_counter()
			n = remove_triplets_with_pipe(in_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[remove_triplets_with_pipe] Finished | Removed {n} lines in {end} seconds', 'light_green'))

	if params['relation_standardization']['do']:
		in_fname	= params['relation_standardization']['in_fname']
		out_fname = params['relation_standardization']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[relation_standardization] Started', 'light_green'))
			start = perf_counter()
			n = relation_standardization(in_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[relation_standardization] Finished | Cleaned {n} lines in {end} seconds', 'light_green'))

	if params['remove_non_human_genes']['do']:
		in_fname	= params['remove_non_human_genes']['in_fname']
		non_human_genes_fname = params['remove_non_human_genes']['non_human_genes_fname']
		out_fname = params['remove_non_human_genes']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[remove_non_human_genes] Started', 'light_green'))
			start = perf_counter()
			removed_lines, removed_genes = remove_non_human_genes(in_fname, non_human_genes_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[remove_non_human_genes] Finished | Removed {removed_lines} lines and {removed_genes} genes in {end} seconds', 'light_green'))

	if params['remove_compunds_without_smiles']['do']:
		in_fname	= params['remove_compunds_without_smiles']['in_fname']
		compound_smiles_fname = params['remove_compunds_without_smiles']['compound_smiles_fname']
		out_fname = params['remove_compunds_without_smiles']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[remove_compunds_without_smiles] Started', 'light_green'))
			start = perf_counter()
			removed_lines, removed_cmps = remove_compunds_without_smiles(in_fname, compound_smiles_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[remove_compunds_without_smiles] Finished | Removed {removed_lines} lines and {removed_cmps} compounds in {end} seconds', 'light_green'))

	if params['add_genes_features_from_pathways']['do']:
		in_fname	= params['add_genes_features_from_pathways']['in_fname']
		in_features_file = params['add_genes_features_from_pathways']['in_features_file']
		out_features_file = params['add_genes_features_from_pathways']['out_features_file']

		if overwrite or check_file(out_features_file):
			print(colored('[add_genes_features_from_pathways] Started', 'light_green'))
			start = perf_counter()
			n = add_genes_features_from_pathways(in_fname, in_features_file, out_features_file)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[add_genes_features_from_pathways] Finished | Added features in {end} seconds', 'light_green'))

	if params['add_genes_features_from_molecular_functions']['do']:
		in_fname	= params['add_genes_features_from_molecular_functions']['in_fname']
		in_features_file = params['add_genes_features_from_molecular_functions']['in_features_file']
		out_features_file = params['add_genes_features_from_molecular_functions']['out_features_file']

		if overwrite or check_file(out_features_file):
			print(colored('[add_genes_features_from_molecular_functions] Started', 'light_green'))
			start = perf_counter()
			n = add_genes_features_from_molecular_functions(in_fname, in_features_file, out_features_file)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[add_genes_features_from_molecular_functions] Finished | Added features in {end} seconds', 'light_green'))

	if params['add_genes_features_from_cellular_components']['do']:
		in_fname	= params['add_genes_features_from_cellular_components']['in_fname']
		in_features_file = params['add_genes_features_from_cellular_components']['in_features_file']
		out_features_file = params['add_genes_features_from_cellular_components']['out_features_file']

		if overwrite or check_file(out_features_file):
			print(colored('[add_genes_features_from_cellular_components] Started', 'light_green'))
			start = perf_counter()
			n = add_genes_features_from_cellular_components(in_fname, in_features_file, out_features_file)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[add_genes_features_from_cellular_components] Finished | Added features in {end} seconds', 'light_green'))
	
	if params['add_genes_features_from_biological_processes']['do']:
		in_fname	= params['add_genes_features_from_biological_processes']['in_fname']
		in_features_file = params['add_genes_features_from_biological_processes']['in_features_file']
		out_features_file = params['add_genes_features_from_biological_processes']['out_features_file']

		if overwrite or check_file(out_features_file):
			print(colored('[add_genes_features_from_biological_processes] Started', 'light_green'))
			start = perf_counter()
			n = add_genes_features_from_biological_processes(in_fname, in_features_file, out_features_file)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[add_genes_features_from_biological_processes] Finished | Added features in {end} seconds', 'light_green'))
	
	if params['remove_entity_nodes']['do']:
		in_fname  = params['remove_entity_nodes']['in_fname']
		to_remove = params['remove_entity_nodes']['to_remove']["entity"]
		out_fname = params['remove_entity_nodes']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[remove_entity_nodes] Started', 'light_green'))
			start = perf_counter()
			total = 0
			total = remove_entity_nodes(in_fname, out_fname, to_remove)

			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[remove_entity_nodes] Finished | Removed {total} lines in {end} seconds', 'light_green'))
	
	if params['drop_duplicates']['do']:
		in_fname	= params['drop_duplicates']['in_fname']
		out_fname = params['drop_duplicates']['out_fname']

		if overwrite or check_file(out_fname):
			print(colored('[drop_duplicates] Started', 'light_green'))
			start = perf_counter()
			n = drop_duplicates(in_fname, out_fname)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[drop_duplicates] Finished | Dropped {n} lines in {end} seconds', 'light_green'))
	
	if params['extract_lcc']['do']:
		in_fname	= params['extract_lcc']['in_fname']
		out_fname = params['extract_lcc']['out_fname']
		debug = params['extract_lcc']['debug']
		if overwrite or check_file(out_fname):
			print(colored('[extract_lcc] Started', 'light_green'))
			start = perf_counter()
			n = extract_largest_connected_component(in_fname, out_fname, debug)
			end = round(perf_counter()-start, 2)
			total_time += end
			print(colored(f'[extract_lcc] Finished | Dropped {n} lines in {end} seconds', 'light_green'))

	print(colored(f'\n[i] Total time required to setup the DRKG: {round(total_time, 2)} second(s)', 'light_magenta'))
	
	if params['stats']['do']:
		k = 8
		print()
		print('-'*k+' STATS '+'-'*k)
		start_file = params['stats']['start_file']
		end_file = params['stats']['end_file']
		stats(start_file, end_file)