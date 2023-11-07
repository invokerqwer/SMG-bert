# from torch_geometric.nn import GINConv
from collections import OrderedDict
from typing import List
from typing import List

IUPAC_VOCAB = OrderedDict([
    ('<pad>', 0),
    ('H', 1),
    ('C', 2),
    ('N', 3),
    ('O', 4),
    ('F', 5),
    ('S', 6),
    ('Cl', 7),
    ('P', 8),
    ('Br', 9),
    ('B', 10),
    ('I', 11),
    ('Si', 12),
    ('Se', 13),
    ('<unk>', 14),
    ('<mask>', 15),
    ('<global>', 16)]
)

'''将原子换成对应的数'''
class Tokenizer():
    def __init__(self):

        self.vocab = IUPAC_VOCAB
        self.tokens = list(self.vocab.keys())
        assert self.start_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<global>"

    @property
    def mask_token(self) -> str:
        return "<mask>"

    @property
    def padding_token(self) -> str:
        return "<pad>"
    def mask_token_id(self):
        return self.convert_atom_to_id(self.mask_token)

    def convert_atom_to_id(self, atom: str) -> int:
        return self.vocab.get(atom, self.vocab['<unk>'])

    def convert_atoms_to_ids(self, mol: List[str]) -> List[int]:
        return [self.convert_atom_to_id(atom) for atom in mol]

    def convert_id_to_atom(self, index: int) -> str:
        return self.tokens[index]

    def convert_ids_to_atoms(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_atom(idx) for idx in indices]

    def add_special_atom(self, mol: List[str]) -> List[str]:
        return [self.start_token] + mol

    def padding_to_size(self,mol: List[int],size):
        return mol + [self.convert_atom_to_id(self.padding_token)] * (size - len(mol))



