import os
import csv
import yaml
import time
import random
import argparse
import numpy as np
import pandas as pd
from typing import *
from bidict import bidict
import rdkit.Chem as Chem
from termcolor import colored
import scipy.sparse as sparse
from collections import defaultdict
import rdkit.Chem.AllChem as AllChem
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from os.path import isfile, getsize, basename
from sklearn.model_selection import train_test_split
import torch

PARAMS_FILE = 'configurations/vitagraph.yml'

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

def check_file(file_name):
	"""
	Checks if a file exists, and if it does, asks the user if they want to override it.

	Parameters:
	file_name (str): The name of the file to check.

	Returns:
	bool: True if the user decides to override the file, False otherwise.
	"""
	# Check if the file exists
	if isfile(file_name):
		# File exists, ask the user if they want to override it
		response = input(colored(f"The file '{file_name}' already exists. Do you want to overwrite it? (yes/no): ", 'red'))
		if response.lower() in ['yes', 'y']:
			return True
		else:
			return False
	else:
		return True

def parse_args():
    parser = argparse.ArgumentParser(
        description='VitaGraph Cleaning and Preprocessing Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--config_file",
        type=str,
        default=PARAMS_FILE,
        metavar="PATH",
        help="Path to the YAML configuration file used for the cleaning pipeline (default: configurations/vitagraph.yml)"
    )
    
    return parser.parse_args()

def read_yaml_config(file_path):
	with open(file_path, 'r') as stream:
		try:
			return yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

def read_drkg_df(in_fname):
	df = pd.read_csv(in_fname, sep='\t', header = 0, names = ["head","interaction", "tail", "source", "type"], dtype=str)
	# Ensure all entries in the 'head' and 'tail' columns are strings
	df['head'] = df['head'].astype(str)
	df['tail'] = df['tail'].astype(str)
	return df

def is_compound(node):
	return 'Compound::' in node

def read_compound_smiles_file(in_fname):
	# desc = f'Reading SMILES from file {in_fname}'
	compound_features_d = {}

	with open(in_fname) as fin:
		_ = fin.readline()
		# for line in tqdm(fin.readlines(), desc=desc):
		for line in fin.readlines():
			line = line.strip().split(',')
			compound = line[0]
			features = line[2]
			compound_features_d[compound] = features
	
	return compound_features_d

def file_size_mb(file_path):
	size_in_bytes = getsize(file_path)
	size_in_mb = size_in_bytes / (1024 * 1024)
	return round(size_in_mb, 2)

def count_lines(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		return sum(1 for line in file)

def stats(file1_path, file2_path):
	"""Compares two files in terms of size and line count, outputting an ASCII table."""
	# Get file sizes in MB
	size1_mb = file_size_mb(file1_path)
	size2_mb = file_size_mb(file2_path)

	# Get line counts
	lines1 = count_lines(file1_path)
	lines2 = count_lines(file2_path)

	# Calculate differences
	size_diff_mb = size1_mb - size2_mb
	lines_diff = lines1 - lines2

	# Calculate percentages
	size_percentage = (size2_mb / size1_mb) * 100 if size1_mb != 0 else 0
	size_percentage_diff = 100-size_percentage
	lines_percentage = (lines2 / lines1) * 100 if lines1 != 0 else 0
	lines_percentage_diff = 100-lines_percentage

	# Determine individual column widths based on each filename length
	file1_name = basename(file1_path)
	file2_name = basename(file2_path)

	file1_col_width = max(len(file1_name), 16)
	file2_col_width = max(len(file2_name), 16)

	header_width = 16

	# Define the format string dynamically for each file column
	header_format = f"| {{:<{header_width}}} | {{:<{file1_col_width}}} | {{:<{file2_col_width}}} | {{:<{header_width}}} |"
	separator = f"+{'-' * (header_width+2)}+{'-' * (file1_col_width+2)}+{'-' * (file2_col_width+2)}+{'-' * (header_width+2)}+"

	# Build and print the ASCII table
	table = f"""
	{separator}
	{header_format.format('Metric', file1_name, file2_name, 'Difference')}
	{separator}
	{header_format.format('Size (MB)', f'{size1_mb:.2f}', f'{size2_mb:.2f}', f'{size_diff_mb:.2f}')}
	{header_format.format('Row Count', lines1, lines2, lines_diff)}
	{separator}
	{header_format.format('Size Percentage', '100.00%', f'{size_percentage:.2f}%', f'{size_percentage_diff:.2f}%')}
	{header_format.format('Row Percentage', '100.00%', f'{lines_percentage:.2f}%', f'{lines_percentage_diff:.2f}%')}
	{separator}
	"""

	print(table)

def standardization(entity):
    entity_type, entity_id = entity.split('::')
    if "|" in entity or ";" in entity:
        return entity
    if ':' in entity_id:
        if entity_type == "Compound":
            source, entity_id = entity_id.split(':')
            if source.startswith("chebi"):
                return f"Compound::CHEBI:{entity_id}"
            elif source.startswith("pubchem"):
                return f"Compound::PubChem_Compounds:{entity_id}"
        return entity
    else:
        
        if entity_type == "Gene":
            return f"Gene::NCBI:{entity_id}"
        if entity_type == "Tax":
            return f"Tax::MESH:{entity_id}"
        if entity_type == "Atc":
            return f"Atc::WHO:{entity_id}"
        if entity_type == "Symptom":
            return f"Symptom::MESH:{entity_id}"
        if entity_type == "Molecular Function" or entity_type == "Biological Process" or entity_type == "Cellular Component":
            return f"{entity_type}::GO:{entity_id}"
        if entity_type == "Side Effect":
            return f"Side Effect::umls:{entity_id}"
        if entity_type == "Pharmacologic Class":
            return f"Pharmacologic Class::bioontology:{entity_id}"
        if entity_type == "Pathway":
            if entity_id.startswith("WP"):
                return f"Pathway::WikiPathway:{entity_id}"
            elif entity_id.startswith("PC"):
                return f"Pathway::PathwayCommons:{entity_id}"
        if entity_type == "Compound":
            if entity_id.startswith("CHEMBL"):
                return f"Compound::CHEMBL:{entity_id}"
            elif entity_id.startswith("DB"):
                return f"Compound::drugbank:{entity_id}"
            
        if entity_type == "Disease":
            if entity_id.startswith("SARS-CoV"):
                return f"Disease::bioarx:{entity_id}"
                

        return f"{entity_type}::MISSING:{entity_id}"

# Utils for the definition of the dataset
def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list

class AtomEnvironment:
    """"A Class to store environment-information for fingerprint features"""
    def __init__(self, environment_atoms: Set[int]):
        self.environment_atoms = environment_atoms  # set of all atoms within radius

class CircularAtomEnvironment(AtomEnvironment):
    """"A Class to store environment-information for morgan-fingerprint features"""

    def __init__(self, central_atom: int, radius: int, environment_atoms: Set[int]):
        super().__init__(environment_atoms)
        self.central_atom = central_atom
        self.radius = radius

class UnfoldedMorganFingerprint:
    """Transforms smiles-strings or molecular objects into unfolded bit-vectors based on Morgan-fingerprints [1].
    Features are mapped to bits based on the amount of molecules they occur in.

    Long version:
        Circular fingerprints do not have a unique mapping to a bit-vector, therefore the features are mapped to the
        vector according to the number of molecules they occur in. The most occurring feature is mapped to bit 0, the
        second most feature to bit 1 and so on...

        Weak-point: features not seen in the fit method are not mappable to the bit-vector and therefore cause an error.
            This behaviour can be deactivated using ignore_unknown=True where these are simply ignored.

    References:
            [1] http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """

    def __init__(self, counted: bool = False, radius: int = 2, use_features: bool = False, ignore_unknown=False):
        """ Initializes the class

        Parameters
        ----------
        counted: bool
            if False, bits are binary: on if present in molecule, off if not present
            if True, bits are positive integers and give the occurrence of their respective features in the molecule
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            instead of atoms, features are encoded in the fingerprint. [2]

        ignore_unknown: bool
            if true features not occurring in fitting are ignored for transformation. Otherwise, an error is raised.

        References
        ----------
        [1] http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
        [2] http://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """

        self._n_bits = None
        self._use_features = use_features
        self._bit_mapping: Optional[bidict] = None

        if not isinstance(counted, bool):
            raise TypeError("The argument 'counted' must be a bool!")
        self._counted = counted

        if not isinstance(ignore_unknown, bool):
            raise TypeError("The argument 'ignore_unknown' must be a bool!")
        self.ignore_unknown = ignore_unknown

        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {radius})")

    def __len__(self):
        return self.n_bits

    @property
    def n_bits(self) -> int:
        """Number of unique features determined after fitting."""
        if self._n_bits is None:
            raise ValueError("Number of bits is undetermined!")
        return self._n_bits

    @property
    def radius(self):
        """Radius for the Morgan algorithm"""
        return self._radius

    @property
    def use_features(self) -> bool:
        """Returns if atoms are hashed according to more abstract features.[2]"""
        return self._use_features

    @property
    def counted(self) -> bool:
        """Returns the bool value for enabling counted fingerprint."""
        return self._counted

    @property
    def bit_mapping(self) -> bidict:
        return self._bit_mapping.copy()

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        self._create_mapping(mol_iterator)

    def _gen_features(self, mol_obj: Chem.Mol) -> Dict[int, int]:
        """returns the a dict, where the key is the feature-hash and the value is the count."""
        return AllChem.GetMorganFingerprint(mol_obj, self.radius, useFeatures=self.use_features).GetNonzeroElements()

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict:
        bi = {}
        _ = AllChem.GetMorganFingerprint(mol_obj, self.radius, useFeatures=self.use_features, bitInfo=bi)
        bit_info = {self.bit_mapping[k]: v for k, v in bi.items()}
        return bit_info

    def explain_smiles(self, smiles: str) -> dict:
        return self.explain_rdmol(Chem.MolFromSmiles(smiles))

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_fp_list = [self._gen_features(mol_obj) for mol_obj in mol_obj_list]
        self._create_mapping(mol_fp_list)
        return self._transform(mol_fp_list)

    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_iterator)

    def _map_features(self, mol_fp) -> List[int]:
        if self.ignore_unknown:
            return [self._bit_mapping[feature] for feature in mol_fp.keys() if feature in self._bit_mapping[feature]]
        else:
            return [self._bit_mapping[feature] for feature in mol_fp.keys()]
            
    def fit_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)

    def fit_transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)

    def _transform(self, mol_fp_list: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]) -> sparse.csr_matrix:
        data = []
        rows = []
        cols = []
        n_col = 0
        if self._counted:
            for i, mol_fp in enumerate(mol_fp_list):
                features, counts = zip(*mol_fp.items())
                data.append(counts)
                rows.append(self._map_features(features))
                cols.append(i)
                n_col += 1
        else:
            for i, mol_fp in enumerate(mol_fp_list):
                data.extend([1] * len(mol_fp))
                rows.extend(self._map_features(mol_fp))
                cols.extend([i] * len(mol_fp))
                n_col += 1
        return sparse.csr_matrix((data, (cols, rows)), shape=(n_col, self.n_bits))

    def _create_mapping(self, molecule_features: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]):
        unraveled_features = [f for f_list in molecule_features for f in f_list.keys()]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(unique_features, key=lambda f: (feature_hash_dict[f], f), reverse=True)
        self._bit_mapping = bidict(zip(feature_order, range(len(feature_order))))
        self._n_bits = len(self._bit_mapping)

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> Dict[int, List[CircularAtomEnvironment]]:
        bit2atom_dict = self.explain_rdmol(mol_obj)
        result_dict = defaultdict(list)

        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():  # type: int, tuple
            for central_atom, radius in matches:  # type: int, int
                if radius == 0:
                    result_dict[bit].append(CircularAtomEnvironment(central_atom, radius, {central_atom}))
                    continue
                env = Chem.FindAtomEnvironmentOfRadiusN(mol_obj, radius, central_atom)
                amap = {}
                _ = Chem.PathToSubmol(mol_obj, env, atomMap=amap)
                env_atoms = amap.keys()
                assert central_atom in env_atoms
                result_dict[bit].append(CircularAtomEnvironment(central_atom, radius, set(env_atoms)))

        # Transforming defaultdict to dict
        return {k: v for k, v in result_dict.items()}

FeatureMatrix = Union[np.ndarray, sparse.csr.csr_matrix]

class DataSet:
    """ Object to contain paired data such das features and label. Supports adding other attributes such as groups.
    """
    def __init__(self, label: np.ndarray, feature_matrix: FeatureMatrix):

        if not isinstance(label, np.ndarray):
            label = np.array(label).reshape(-1)

        if label.shape[0] != feature_matrix.shape[0]:
            raise IndexError

        self.label = label
        self.feature_matrix = feature_matrix
        self._additional_attributes = set()

    def add_attribute(self, attribute_name, attribute_values: np.ndarray):
        if not isinstance(attribute_values, np.ndarray):
            attribute_values = np.array(attribute_values).reshape(-1)

        if attribute_values.shape[0] != len(self):
            raise IndexError("Size does not match!")

        self._additional_attributes.add(attribute_name)
        self.__dict__[attribute_name] = attribute_values

    @property
    def columns(self) -> dict:
        r_dict = {k: v for k, v in self.__dict__.items() if k in self._additional_attributes}
        r_dict["label"] = self.label
        r_dict["feature_matrix"] = self.feature_matrix
        return r_dict

    def __len__(self):
        return self.label.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx) -> Union[dict, 'DataSet']:
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}

        data_slice = DataSet(self.label[idx], self.feature_matrix[idx])
        for additional_attribute in self._additional_attributes:
            data_slice.add_attribute(additional_attribute, self.__dict__[additional_attribute][idx])

        return data_slice
    
# Utility Functions to exrtact the LCC from the DRKG
class UnionFind:
    def __init__(self):
        self.parent = dict()
        self.rank = defaultdict(int)  # Optimization with rank compression

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
            
        # Union by rank to optimize operations
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def debug_print(message, debug=True):
    """Utility function to print messages only if debug is active"""
    if debug:
        print(message)

def count_connected_components(tsv_file, output_file="output", debug=True):
    """
    Analyzes the connected components in a knowledge graph in TSV format.
    
    Args:
        tsv_file: Path to the TSV file with 5 columns
        output_file: Output file
        debug: If True, prints debug messages
    
    Returns:
        Dict with statistics on the connected components
    """
    debug_print(f"Analyzing file: {tsv_file}", debug)
    start_time = time.time()
    
    uf = UnionFind()
    edges_count = 0
    
    # First pass: build connected components
    debug_print("Building connected components...", debug)
    with open(tsv_file, 'r', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        header = next(tsv_reader, None)  # Save header
        
        for i, row in enumerate(tsv_reader):
            if debug and i % 1000000 == 0 and i > 0:
                debug_print(f"  Processed {i:,} rows...", debug)
            try:
                if len(row) < 5:
                    debug_print(f"Warning: row {i+2} does not have 5 columns: {row}", debug)
                    continue
                head, _, tail, _, _ = row
                uf.union(head, tail)
                edges_count += 1
            except ValueError:
                debug_print(f"Warning: error at row {i+2}: {row}", debug)
                continue
    
    debug_print("Computing component statistics...", debug)
    
    # Collect component statistics
    component_sizes = defaultdict(int)
    node_to_component = {}
    
    for node in uf.parent:
        root = uf.find(node)
        component_sizes[root] += 1
        node_to_component[node] = root
    
    total_nodes = len(uf.parent)
    total_from_components = sum(component_sizes.values())
    num_components = len(component_sizes)
    
    # Verify data consistency
    assert total_nodes == total_from_components, "Error: inconsistent count!"
    
    # Sort components by size
    components_by_size = sorted(component_sizes.items(), key=lambda x: x[1], reverse=True)
    sizes = [size for _, size in components_by_size]
    
    # Identify the largest connected component (LCC)
    largest_component_id = components_by_size[0][0] if components_by_size else None
    largest_component_size = sizes[0] if sizes else 0
    
    # Extract LCC if requested
    if largest_component_id is not None:
        debug_print(f"Extracting LCC to file: {output_file}", debug)
        
        lcc_edges_count = 0
        lcc_nodes = set()

        # Check if the input and output files are the same
        same_file = os.path.abspath(tsv_file) == os.path.abspath(output_file)
        temp_output_file = output_file + ".temp" if same_file else output_file
        
        with open(tsv_file, 'r', encoding='utf-8') as fin, \
             open(temp_output_file, 'w', encoding='utf-8', newline='') as fout:
            
            tsv_reader = csv.reader(fin, delimiter='\t')
            tsv_writer = csv.writer(fout, delimiter='\t')
            
            # Write header
            header = next(tsv_reader, None)
            if header:
                tsv_writer.writerow(header)
            
            # Second pass: extract LCC rows
            for i, row in enumerate(tsv_reader):
                if debug and i % 1000000 == 0 and i > 0:
                    debug_print(f"  Filtered {i:,} rows for LCC...", debug)
                
                try:
                    if len(row) < 5:
                        continue
                    
                    head, _, tail, _, _ = row
                    
                    # Check if both nodes belong to the LCC
                    head_component = node_to_component.get(head)
                    tail_component = node_to_component.get(tail)
                    
                    if head_component == largest_component_id and tail_component == largest_component_id:
                        tsv_writer.writerow(row)
                        lcc_edges_count += 1
                        lcc_nodes.add(head)
                        lcc_nodes.add(tail)
                        
                except Exception as e:
                    debug_print(f"Error during LCC extraction at row {i+2}: {e}", debug)
                    continue
        
        # If we used a temporary file, replace the original with it
        if same_file:
            debug_print(f"Input and output files are the same. Replacing original with filtered data.", debug)
            os.replace(temp_output_file, output_file)
            
        debug_print(f"LCC extracted: {len(lcc_nodes):,} nodes, {lcc_edges_count:,} edges", debug)
    
    # STATISTICS
    if debug:
        debug_print(f"\n===== BASIC STATISTICS =====", debug)
        debug_print(f"Number of connected components: {num_components:,}", debug)
        debug_print(f"Total number of unique nodes: {total_nodes:,}", debug)
        debug_print(f"Total number of edges: {edges_count:,}", debug)
        debug_print(f"Sum of component sizes: {total_from_components:,}", debug)
        
        debug_print(f"\n===== ADVANCED STATISTICS =====", debug)
        debug_print(f"Average component size: {np.mean(sizes):.2f} nodes", debug)
        debug_print(f"Median component size: {np.median(sizes):.2f} nodes", debug)
        debug_print(f"Standard deviation: {np.std(sizes):.2f}", debug)
        
        debug_print("\n===== TOP 10 COMPONENTS BY SIZE =====", debug)
        for i, (_, size) in enumerate(components_by_size[:10], 1):
            percent = (size / total_nodes) * 100
            debug_print(f"#{i}: {size:,} nodes ({percent:.2f}% of the graph)", debug)
        
        debug_print("\n===== COMPONENT SIZE DISTRIBUTION =====", debug)
        size_distribution = defaultdict(int)
        for size in sizes:
            size_distribution[size] += 1
        
        # Function to group sizes into brackets
        def get_size_bracket(size):
            if size == 1:
                return "1 (isolated)"
            elif size == 2:
                return "2 (pairs)"
            elif size <= 5:
                return "3-5"
            elif size <= 10:
                return "6-10"
            elif size <= 100:
                return "11-100"
            elif size <= 1000:
                return "101-1000"
            elif size <= 10000:
                return "1001-10000"
            else:
                return ">10000"
        
        brackets = defaultdict(int)
        for size, count in size_distribution.items():
            brackets[get_size_bracket(size)] += count
        
        # Predefined bracket order
        bracket_order = ["1 (isolated)", "2 (pairs)", "3-5", "6-10", "11-100", "101-1000", "1001-10000", ">10000"]
        bracket_order = [b for b in bracket_order if b in brackets]
        
        debug_print("Component count by size bracket:", debug)
        for bracket in bracket_order:
            if bracket in brackets:
                debug_print(f"  {bracket}: {brackets[bracket]:,} components", debug)
        
        isolated_nodes = size_distribution.get(1, 0)
        isolated_percent = (isolated_nodes / num_components) * 100 if num_components > 0 else 0
        
        small_components = sum(count for size, count in size_distribution.items() if size <= 5)
        small_percent = (small_components / num_components) * 100 if num_components > 0 else 0
        
        debug_print(f"\nIsolated nodes: {isolated_nodes:,} ({isolated_percent:.2f}% of components)", debug)
        debug_print(f"Small components (≤5 nodes): {small_components:,} ({small_percent:.2f}% of components)", debug)
        
        max_possible_edges = total_nodes * (total_nodes - 1) / 2
        density = edges_count / max_possible_edges if max_possible_edges > 0 else 0
        debug_print(f"\nGraph density: {density:.8f}", debug)
        
        lcc_ratio = (largest_component_size / total_nodes) * 100 if total_nodes > 0 else 0
        debug_print(f"\nLCC contains {largest_component_size:,} nodes ({lcc_ratio:.2f}% of the graph)", debug)
        
        lcc_density = lcc_edges_count / (len(lcc_nodes) * (len(lcc_nodes) - 1) / 2) if len(lcc_nodes) > 1 else 0
        debug_print(f"LCC density: {lcc_density:.8f}", debug)
        
        execution_time = time.time() - start_time
        debug_print(f"\nAnalysis completed in {execution_time:.2f} seconds", debug)
    
    return edges_count-lcc_edges_count

def extract_largest_connected_component(tsv_file, output_file, debug=True):
    """
    Main function to analyze a graph from a TSV file.
    
    Args:
        tsv_file: Path to the TSV file
        output_file: output file
        debug: If True, show debug messages
    """
    if not os.path.isfile(tsv_file):
        print(f"Error: file '{tsv_file}' not found.")
        return None
    
    res = count_connected_components(
        tsv_file=tsv_file,
        output_file=output_file,
        debug=debug,
    )

    return res

# Utils for model training 
def load_data(edge_index_path, features_paths_per_type, verbose=True):

    edge_ind = pd.read_csv(edge_index_path, sep='\t', dtype=str)
   
    node_features = {}
    all_edge_ind_entities = set(edge_ind["head"]).union(set(edge_ind["tail"]))

    if features_paths_per_type != None: 
        node_features = {node_type: pd.read_csv(feature_path).drop_duplicates() for node_type, feature_path in features_paths_per_type.items()}
        
    entities_types = set(edge_ind["head"].apply(lambda x: x.split("::")[0])).union(set(edge_ind["tail"].apply(lambda x: x.split("::")[0]))) 
    for node_type in entities_types:
        if node_type not in node_features:
            node_features[node_type] = None
        else:
            node_features[node_type] = node_features[node_type][node_features[node_type]["id"].isin(all_edge_ind_entities)]  ## filter the entities not present in the edge index
            
    triplets_count = len(edge_ind)
    interaction_types_count = len(edge_ind.interaction.unique())

    if verbose:
        print(colored(f'[loaded edge index] triplets count: {triplets_count} interaction types count: {interaction_types_count}', 'green'))

    return edge_ind, node_features

def entities2id(edge_index, node_features_per_type):
    # create a dictionary that maps the entities to an integer id
    entities = set(edge_index["head"]).union(set(edge_index["tail"]))
    entities2id = {}
    all_nodes_per_type = {}
    for x in entities:
        if x.split("::")[0] not in all_nodes_per_type:
            all_nodes_per_type[x.split("::")[0]] = [x]
        else:
            all_nodes_per_type[x.split("::")[0]].append(x)
    for node_type, features in node_features_per_type.items():
        if features is None:
            for idx, node in enumerate(all_nodes_per_type[node_type]):
                entities2id[node] = idx #+ offset
            continue
        for idx, node in enumerate(features.id):
            entities2id[node] = idx #+ offset
    return entities2id, all_nodes_per_type

def entities2id_offset(edge_index, node_features_per_type, quiet=False):
    # create a dictionary that maps the entities to an integer id
    entities = set(edge_index["head"]).union(set(edge_index["tail"]))
    entities2id = {}
    all_nodes_per_type = {}
    
    for x in entities:
        if x.split("::")[0] not in all_nodes_per_type:
            all_nodes_per_type[x.split("::")[0]] = [x]
        else:
            all_nodes_per_type[x.split("::")[0]].append(x)
    
    if quiet:
        for node_type, nodes in all_nodes_per_type.items():
            print(colored(f'    [{node_type}] count: {len(nodes)}', 'green'))

    offset = 0
    for node_type, features in node_features_per_type.items():
        if features is None:
            for idx, node in enumerate(all_nodes_per_type[node_type]):
                entities2id[node] = idx + offset
            offset += len(all_nodes_per_type[node_type])
            continue
        
        all_edge_index_nodes = [ x for x in features.id.values if x in all_nodes_per_type[node_type]]

        for idx, node in enumerate(all_edge_index_nodes):
            entities2id[node] = idx + offset
        offset += len(all_edge_index_nodes)

    return entities2id, all_nodes_per_type

def rel2id(edge_index):
    # create a dictionary that maps the relations to an integer id
    rel2id = {rel.replace(" ","_"): idx for idx, rel in enumerate(edge_index.interaction.unique())}
    relations = list(rel2id.keys())
    for rel in relations:
        rel2id[f"rev_{rel}"] = rel2id[rel] 
    return rel2id

def rel2id_offset(edge_index):
    # create a dictionary that maps the relations to an integer id
    relation2id = {rel.replace(" ","_"): idx for idx, rel in enumerate(edge_index.interaction.unique())}
    rel_number = len(relation2id)
    relations = list(relation2id.keys())

    for rel in relations:
        relation2id[f"rev_{rel}"] = relation2id[rel] + rel_number

    relation2id["self"] = rel_number*2 
    # print(relation2id)
    return relation2id

def index_entities_edge_ind(edge_ind, entities2id):
    # create a new edge index where the entities are replaced by their integer id
    indexed_edge_ind = edge_ind.copy()
    indexed_edge_ind["head"] = indexed_edge_ind["head"].apply(lambda x: entities2id[x])
    indexed_edge_ind["tail"] = indexed_edge_ind["tail"].apply(lambda x: entities2id[x])
    return indexed_edge_ind

def edge_ind_to_id(edge_ind, entities2id, relation2id):
    # create a new edge index where the entities are replaced by their integer id
    indexed_edge_ind = edge_ind.copy()
    indexed_edge_ind["head"] = indexed_edge_ind["head"].apply(lambda x: entities2id[x])
    indexed_edge_ind["interaction"] = indexed_edge_ind["interaction"].apply(lambda x: relation2id[x.replace(" ","_")])
    indexed_edge_ind["tail"] = indexed_edge_ind["tail"].apply(lambda x: entities2id[x])
    return indexed_edge_ind

def graph_to_undirect(edge_index, rel_num):
    reverse_triplets = edge_index.copy()
    reverse_triplets[:,[0,2]] = reverse_triplets[:,[2,0]]
    reverse_triplets[:,1] += rel_num//2
    undirected_edge_index = np.concatenate([edge_index, reverse_triplets], axis=0)
    return torch.tensor(undirected_edge_index)

def add_self_loops(train_index, num_entities, num_relations):
    head = torch.tensor([x for x in range(num_entities)])
    interaction = torch.tensor([num_relations for _ in range(num_entities)])
    tail = torch.tensor([x for x in range(num_entities)])
    self_loops = torch.cat([head.view(1,-1), interaction.view(1,-1), tail.view(1,-1)], dim=0).T
    train_index_self_loops = torch.cat([train_index, self_loops], dim=0)
    return train_index_self_loops

def set_target_label(edge_ind, target_edges):
    edge_ind["label"] = edge_ind["type"].apply(lambda x: x in target_edges)
    return edge_ind

def select_target_triplets(edge_index):
    target_triplets = edge_index.loc[edge_index["label"]==1,:].copy()
    non_target_triplets = edge_index.loc[edge_index["label"]==0,:].copy()
    return non_target_triplets, target_triplets

def negative_sampling(target_triplets, negative_rate=1):
    target_triplets = np.array(target_triplets)
    src, _, dst = target_triplets.T
    uniq_entity = np.unique((src, dst))
    pos_num = target_triplets.shape[0]
    neg_num = pos_num * negative_rate
    neg_samples = np.tile(target_triplets, (negative_rate, 1))
    values = np.random.choice(uniq_entity, size=neg_num)
    choices = np.random.uniform(size=neg_num)
    # choice on who to perturb
    subj = choices > 0.5
    obj = choices <= 0.5
    # actual perturbation
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    labels = torch.zeros(target_triplets.shape[0]+neg_samples.shape[0])
    labels[:target_triplets.shape[0]] = 1
    neg_samples = torch.tensor(neg_samples)
    samples = torch.cat([torch.tensor(target_triplets), neg_samples], dim=0)
    return samples, labels

def triple_sampling(target_triplet, val_size, test_size, quiet=True):
    val_len = len(target_triplet) * val_size
    # split the data into training, testing, and validation 
    temp_data, test_data = train_test_split(target_triplet, test_size=test_size, random_state=42, shuffle=True)
    train_data, val_data = train_test_split(temp_data, test_size=(val_len / len(temp_data)), random_state=42, shuffle=True)
    # print the shapes of the resulting sets
    if not quiet:
        print(f"Total number of target edges: {len(target_triplet)}")
        print(f"\tTraining set shape: {len(train_data)}")
        print(f"\tValidation set shape: {len(val_data)}" )
        print(f"\tTesting set shape: {len(test_data)}\n")
    return train_data, val_data, test_data

def flat_index(triplets, num_nodes):
    fr, to = triplets[:, 0]*num_nodes, triplets[:, 2]
    offset = triplets[:, 1] * num_nodes*num_nodes 
    flat_indices = fr + to + offset
    return flat_indices

def entities_features_flattening(node_features_per_type, all_nodes_per_type):
    # flatten the features of the entities
    flattened_features_per_type = {}

    for node_type, features in node_features_per_type.items():
        if features is None:
            flattened_features_per_type[node_type] = None # torch.ones((len(all_nodes_per_type[node_type]), 1),dtype=torch.float)
            continue
        features = features.drop(columns=["id"])
        features = features.map(lambda x: np.array([int(v) for v in x ]))
        features_matrix = []
        for x in features.values:
            features_matrix.append(np.concatenate(x))
        flattened_features_per_type[node_type] = torch.tensor(np.array(features_matrix), dtype=torch.float)

    return flattened_features_per_type

def create_hetero_data(indexed_edge_ind, node_features_per_type, rel2id, verbose=True):
    data = HeteroData()
    total_nodes = 0
    for node_type, features in node_features_per_type.items():
        data[f"{node_type}"].x = torch.tensor(features, dtype=torch.float).contiguous()
        total_nodes += len(features)
    all_interaction_per_type= indexed_edge_ind[["interaction","type"]].drop_duplicates().values
    for interaction, entities in all_interaction_per_type:
        edge_interaction = indexed_edge_ind.loc[(indexed_edge_ind["interaction"] == interaction) & (indexed_edge_ind["type"]==entities)]
        entity_types = entities.split(" - ")  ######## "-" or " - " depending on the dataset
        edges = edge_interaction.loc[:,["head","tail"]].values
        data[entity_types[0].replace(" ",""),interaction.replace(" ","_"),entity_types[1].replace(" ","")].edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

    return data

def data_split_and_negative_sampling( data, target_edges, rev_target, val_ratio=0.2, test_ratio=0.3 ,neg_sampling_ratio=1.0):
    transform_split = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=True,
        is_undirected=True,
        edge_types=target_edges,
        rev_edge_types=rev_target
    )
    return transform_split(data)

def get_all_triplets(data, rel2id):    
    head_tail = torch.cat(list(data.edge_index_dict.values()), dim=1)
    # add the relation id to the triplets TO the last dimension
    rel_ids = []
    for edge_type in data.edge_types:
        rel_ids.append(torch.tensor([rel2id[edge_type[1]] for _ in range(data[edge_type].edge_index.shape[1])]))
    rel_ids = torch.cat(rel_ids, dim=0)
    triplets = torch.cat([head_tail, rel_ids.view(1,-1)], dim=0)
    triplets = triplets.T
    # Swap the order of tail and interaction to get the triplets in the form (head, interaction, tail)
    triplets[:,[1,2]] = triplets[:,[2,1]]
    return triplets

def get_target_triplets_and_labels(data, target_edges, relation2id):
    all_target_triplets = []
    all_labels = []

    for target_edge in target_edges:

        head_tail = data[target_edge].edge_label_index
        rel_ids = torch.tensor([relation2id[target_edge[1]] for _ in range(head_tail.shape[1])])
        # add the relation id to the triplets TO the last dimension

        triplets = torch.cat([head_tail, rel_ids.view(1,-1)], dim=0)
        triplets = triplets.T
        # Swap the order of tail and interaction to get the triplets in the form (head, interaction, tail)
        triplets[:,[1,2]] = triplets[:,[2,1]]
        all_target_triplets.append(triplets)
        all_labels.append(data[target_edge].edge_label)

    all_target_triplets = torch.cat(all_target_triplets, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_target_triplets, all_labels

def graph_transform(data):
    transformation = []
    transformation.append(T.ToUndirected())
    transformation.append(T.AddSelfLoops())
    transformation.append(T.RemoveDuplicatedEdges()) # Always remove duplicated edges
    transform = T.Compose(transformation)
    data = transform(data)
    return data
                      
def evaluation_metrics(model, embeddings, all_target_triplets, test_triplet, num_generate, device, hits=[1,3,10]):
    src, _, dst = all_target_triplets.T
    unique_nodes = torch.unique(torch.cat((src,dst), dim = 0))
    if num_generate > unique_nodes.size(0):
        print(f"[ERROR] requested more triplets than available nodes")
    with torch.no_grad():
        for head in [True, False]:
            generator = torch.Generator().manual_seed(42)
            random_indices = torch.randperm(unique_nodes.size(0), generator=generator)[:num_generate]
            selected_nodes = unique_nodes[random_indices]
            if head:
                head_rel = test_triplet[:, :2] #(test_triplet)  (all_target_triplets)
                head_rel = torch.repeat_interleave(head_rel, num_generate, dim=0) # shape (test_triplet.size(0)*100, 3)
                target_tails = torch.tile(selected_nodes, (1, test_triplet.size(0))).view(-1,1) #shape (test_triplet.size(0)*100, 1)
                mrr_triplets = torch.cat((head_rel, target_tails), dim=-1) #shape (test_triplet.size(0)*100, 3)
            else:
                rel_tail = test_triplet[:, 1:]
                rel_tail = torch.repeat_interleave(rel_tail, num_generate, dim=0) # shape (test_triplet.size(0)*100, 3)
                target_heads = torch.tile(selected_nodes, (1, test_triplet.size(0))).view(-1,1) #shape (test_triplet.size(0)*100, 1)
                mrr_triplets = torch.cat((target_heads, rel_tail), dim=-1) #shape (test_triplet.size(0)*100, 3)
            mrr_triplets = mrr_triplets.view(test_triplet.size(0), num_generate, 3)# shape(test triplets, mrr_triplets, 3)
            mrr_triplets = torch.cat((mrr_triplets, test_triplet.view(-1,1,3)), dim=1)# shape(test triplets, mrr_triplets+1, 3)
            scores = model.distmult(embeddings, mrr_triplets.view(-1,3)).view(test_triplet.size(0), num_generate+1)
            _, ranks = torch.sort(scores, descending=True)
            if head:
                ranks_s =  ranks[:, -1]
            else:
                ranks_o =  ranks[:, -1]
        # rank can't be zero since we then take the reciprocal of 0, so everyone is shifted by one position
        ranks = torch.cat([ranks_s, ranks_o]) + 1 # change to 1-indexed
        mrr = torch.mean(1.0 / ranks)
        hits = {at:0 for at in hits}
        for hit in hits:
            avg_count = torch.sum((ranks <= hit))/ranks.size(0)
            hits[hit] = avg_count.item()
    return mrr.item(), hits #, auroc, auprc