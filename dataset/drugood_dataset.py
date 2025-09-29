import numpy as np
import os.path as osp
import torch
import torch.utils
import torch.utils.data
from torch_geometric.data import InMemoryDataset, Data
import mmcv
import rdkit
from rdkit import Chem
from mmengine import fileio
import os

class SmileToGraph(object):
    # Adapt from https://github.com/tencent-ailab/DrugOOD/blob/main/drugood/datasets/pipelines/formating.py
    """Transform smile input to graph format"""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.smile2graph(results[key])
        return results

    # def get_atom_features(self, atom):
    #     # The usage of features is along with the Attentive FP.
    #     feature = np.zeros(4)
    #     # Symbol
    #     symbol = atom.GetSymbol()
    #     # 6: C, 7: N, 8: O, 9: F etc.
    #     symbol_list = [ 'C', 'N', 'O', 'F']
    #     if symbol in symbol_list:
    #         loc = symbol_list.index(symbol)
    #         feature[loc] = 1
        
    #     return feature

    def get_atom_features(self, atom):
        # The usage of features is along with the Attentive FP.
        feature = np.zeros(39)
        # Symbol
        symbol = atom.GetSymbol()
        symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
        if symbol in symbol_list:
            loc = symbol_list.index(symbol)
            feature[loc] = 1
        else:
            feature[15] = 1

        # Degree
        degree = atom.GetDegree()
        if degree > 5:
            print("atom degree larger than 5. Please check before featurizing.")
            raise RuntimeError

        feature[16 + degree] = 1

        # Formal Charge
        charge = atom.GetFormalCharge()
        feature[22] = charge

        # radical electrons
        radelc = atom.GetNumRadicalElectrons()
        feature[23] = radelc

        # Hybridization
        hyb = atom.GetHybridization()
        hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                              rdkit.Chem.rdchem.HybridizationType.SP2,
                              rdkit.Chem.rdchem.HybridizationType.SP3,
                              rdkit.Chem.rdchem.HybridizationType.SP3D,
                              rdkit.Chem.rdchem.HybridizationType.SP3D2]
        if hyb in hybridization_list:
            loc = hybridization_list.index(hyb)
            feature[loc + 24] = 1
        else:
            feature[29] = 1

        # aromaticity
        if atom.GetIsAromatic():
            feature[30] = 1

        # hydrogens
        hs = atom.GetNumImplicitHs()
        feature[31 + hs] = 1

        # chirality, chirality type
        if atom.HasProp('_ChiralityPossible'):
            feature[36] = 1

            try:
                chi = atom.GetProp('_CIPCode')
                chi_list = ['R', 'S']
                loc = chi_list.index(chi)
                feature[37 + loc] = 1
            except KeyError:
                feature[37] = 0
                feature[38] = 0

        return feature

    def get_bond_features(self, bond):
        feature = np.zeros(10)

        # bond type
        type = bond.GetBondType()
        bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                          rdkit.Chem.rdchem.BondType.DOUBLE,
                          rdkit.Chem.rdchem.BondType.TRIPLE,
                          rdkit.Chem.rdchem.BondType.AROMATIC]
        if type in bond_type_list:
            loc = bond_type_list.index(type)
            feature[0 + loc] = 1
        else:
            print("Wrong type of bond. Please check before feturization.")
            raise RuntimeError

        # conjugation
        conj = bond.GetIsConjugated()
        feature[4] = conj

        # ring
        ring = bond.IsInRing()
        feature[5] = ring

        # stereo
        stereo = bond.GetStereo()
        stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                       rdkit.Chem.rdchem.BondStereo.STEREOANY,
                       rdkit.Chem.rdchem.BondStereo.STEREOZ,
                       rdkit.Chem.rdchem.BondStereo.STEREOE]
        if stereo in stereo_list:
            loc = stereo_list.index(stereo)
            feature[6 + loc] = 1
        else:
            print("Wrong stereo type of bond. Please check before featurization.")
            raise RuntimeError

        return feature

    def smile2graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            return None
        src = []
        dst = []
        atom_feature = []
        bond_feature = []

        try:
            for atom in mol.GetAtoms():
                
                one_atom_feature = self.get_atom_features(atom)
                
                atom_feature.append(one_atom_feature)

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                one_bond_feature = self.get_bond_features(bond)
                src.append(i)
                dst.append(j)
                bond_feature.append(one_bond_feature)
                src.append(j)
                dst.append(i)
                bond_feature.append(one_bond_feature)

            src = torch.tensor(src).long()
            dst = torch.tensor(dst).long()
            edge_index = torch.vstack([src,dst])
            atom_feature = np.array(atom_feature)
            bond_feature = np.array(bond_feature)
            atom_feature = torch.tensor(atom_feature).float()
            bond_feature = torch.tensor(bond_feature).float()
            graph_cur_smile = Data(x=atom_feature, edge_index=edge_index, edge_attr=bond_feature)
            return graph_cur_smile

        except RuntimeError:
            return None

class DrugOOD(InMemoryDataset):
    splits = ['iid', 'ood', 'mixed']
    valid_datasets = [
        'lbap_general_ec50_assay',
        'lbap_general_ec50_scaffold',
        'lbap_general_ec50_size',
        'lbap_general_ic50_assay',
        'lbap_general_ic50_scaffold',
        'lbap_general_ic50_size'
    ]

    def __init__(self, root, dataset_name, mode='iid', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        assert dataset_name in self.valid_datasets, f"Dataset must be one of: {self.valid_datasets}"
        
        self.mode = mode
        self.dataset_name = dataset_name
        self.smile2graph = SmileToGraph(['smiles'])
        
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index(f'{self.dataset_name}_{mode}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return [f'{self.dataset_name}.json']

    @property
    def processed_file_names(self):
        return [
            f'{self.dataset_name}_iid.pt', 
            f'{self.dataset_name}_ood.pt', 
            f'{self.dataset_name}_mixed.pt'
        ]

    def download(self):
        if not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names[0])):
            print(f"Raw data {self.raw_file_names[0]} doesn't exist at {self.raw_dir}")
            print("Please make sure the data exists at the specified path.")
            raise FileNotFoundError

    def process(self):
        if self.mode == 'iid':
            raw_dataset = fileio.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['train']
            num = 6000
        elif self.mode == 'ood':
            raw_dataset = fileio.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_test']
            num = 5000
        elif self.mode == 'mixed':
            raw_dataset = fileio.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_val']
            num = 1000

        data_list = []
        for idx, case in enumerate(raw_dataset[:num]):
            try:
                case = self.smile2graph(case)
                if case is None or case['smiles'] is None:
                    continue
                    
                case['smiles'].y = torch.tensor(case['cls_label'], dtype=torch.long).unsqueeze(dim=0)
                case['smiles'].idx = idx
                data = case['smiles']
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            except Exception as e:
                print(f"Error processing case {idx}: {e}")
                continue

        idx = self.processed_file_names.index(f'{self.dataset_name}_{self.mode}.pt')
        
        if len(data_list) == 0:
            print(f"Warning: No valid data found for {self.mode} mode")
            empty_data = Data(x=torch.zeros(0, 1), edge_index=torch.zeros(2, 0, dtype=torch.long))
            data_list = [empty_data]

        torch.save(self.collate(data_list), self.processed_paths[idx])

