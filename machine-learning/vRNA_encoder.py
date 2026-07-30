# Code purpose - to design encoder method that intake DNA with Watson-Crick base pairing secondary structure (duplex DNA, hairpins/stem loops) + labeled features -> dot bracket representation
# Uses:
# 1. PDB/SASBDB crystal/X-Ray/NMR/cryo-EM structures where a correct topology can be identified
# 2. Existing Literature with determined ORDER of loops (or basic information about their structure for SOME types, like for hybrid if 2 propeller + 1 lateral is stated)
import numpy as np
import ViennaRNA as vrna

def gen_residues(primary_sequence_list):
    residue_list = []
    for sequence in primary_sequence_list:
        for idx, nucleotide in enumerate(sequence):
            if idx == len(sequence) - 1:
                residue_list.append("S" + nucleotide)
            else:
                residue_list.append("S" + nucleotide + "P")
    return residue_list

def base_pair_calculator(sequence, alignment_score):
    # Condition: Sequence = list of nuceleotide strands
    # Output: List of base pairs
    seq_lengths = []
    for seq in sequence:
        seq_lengths.append(len(seq))
    total_length = sum(seq_lengths)
    num_strands = len(sequence)
    if len(sequence) > 1:
        sequence_combined = "&".join(sequence)
    else:
        sequence_combined = sequence
    # vrna.params_load_RNA_Turner2004() # RNA
    vrna.params_load_DNA_Mathews2004() # DNA, V9.6_DNA2
    fc = vrna.fold_compound(sequence_combined)
    dot_bracket, mfe = fc.pf()
    matrix = np.ones((total_length, total_length))
        # Stack to store the indices of the opening brackets
    stack = []
    # Iterate over the dot-bracket notation
    for i, char in enumerate(dot_bracket):
        #if char in '(|{,':
        if char in '(':
            # If the character is an opening bracket, push its index onto the stack
            stack.append(i)
        #elif char in ',)|}':
        elif char in ')':
            if stack: 
                # If the character is a closing bracket, pop the index of the matching opening bracket from the stack
                j = stack.pop()
                # Set the corresponding elements in the matrix to 1
                matrix[j][i] = alignment_score-1
                matrix[i][j] = alignment_score-1
    return matrix

def gen_distance_matrix(primary_sequence, # primary sequence - joined string, primary sequence - original list
                        primary_sequence_list):
    # secondary_matrix = np.ones((len(primary_sequence), len(primary_sequence)))
    alignment_score = 7
    secondary_matrix = base_pair_calculator(primary_sequence_list, alignment_score)
    # print(f"Details: Primary sequence: {primary_sequence}")
    residue_list = gen_residues(primary_sequence_list)
    # secondary_residue_matrix = np.zeros((len(''.join(residue_list)), len(''.join(residue_list))))
    secondary_residue_matrix = np.ones((len(''.join(residue_list)), len(''.join(residue_list))))

    idx_2 = 0
    for idx_y, residue_y in enumerate(residue_list):
        idx_1 = 0  # Reset idx_1 for each new residue_y
        for idx_x, residue_x in enumerate(residue_list):
            # print(f"Residue x: {residue_x}")
            # print(f"Residue y: {residue_y}")
            # print(f"idx_1: {idx_1}, idx_2: {idx_2}")
            # print(f"Alignment information: {idx_x} and {idx_y}, score = {secondary_matrix[idx_x, idx_y]}")
            if secondary_matrix[idx_x, idx_y] == int(alignment_score-1):
                alignment_score_matrix = np.full((len(residue_x), len(residue_y)), alignment_score-1)
                secondary_residue_matrix[idx_1:idx_1+len(residue_x), idx_2:idx_2+len(residue_y)] = alignment_score_matrix
            idx_1 += len(residue_x)  # Increment idx_1 for the next residue_x
        idx_2 += len(residue_y)  # Increment idx_2 for the next residue_y     
    return secondary_residue_matrix

class get_secondary_structure():
    def __init__(self, primary_sequence):
        self.primary_sequence = primary_sequence

    def gen_distance_matrix(self):
        return gen_distance_matrix("".join(self.primary_sequence), 
                                   self.primary_sequence)