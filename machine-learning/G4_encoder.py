# Code purpose - to design encoder method that intake G4/iM secondary structure + labeled features -> dot bracket representation
# Uses:
# 1. PDB/SASBDB crystal/X-Ray/NMR/cryo-EM structures where a correct topology can be identified
# 2. Existing Literature with determined ORDER of loops (or basic information about their structure for SOME types, like for hybrid if 2 propeller + 1 lateral is stated)
import numpy as np

def gen_residues(primary_sequence_list):
    residue_list = []
    for sequence in primary_sequence_list:
        for idx, nucleotide in enumerate(sequence):
            if idx == len(sequence) - 1:
                residue_list.append("S" + nucleotide)
            else:
                residue_list.append("S" + nucleotide + "P")
    return residue_list

def gen_distance_matrix(primary_sequence, # primary sequence - joined string
                        primary_sequence_list, # primary sequence - original list
                        secondary_structure, 
                        loop_order, 
                        multi_block_cond,
                        asymmetric_block):
    # Primary sequence - primary sequence of G4
    # Secondary structure - dot-cross bracket format of secondary structure (RNAFold method)
    # Loop order - order of loops in G4
    # Structure type - The type of G4 based on its topology (not as important)
    # Multi-block cond - Whether the G4 is one-block (~90% of the time) or contains multiple G4 stacks connected by a linker
    
    ##########################################
    ##########################################
    # Canonical G4's
    ##########################################
    ##########################################
    
    # info_dict = {}
    # for i in range(len(primary_sequence)):
    #     info_dict[i] = [primary_sequence[i], secondary_structure[i]]
    # debug step - remove unnecessary spaces
    
    # Make a numpy array for distance matrix, np zeros
    # secondary_matrix = np.zeros((len(primary_sequence), len(primary_sequence)))
    secondary_matrix = np.zeros((len(primary_sequence), len(primary_sequence)))

    loop_order = [loop.replace(' ','') for loop in loop_order]
    loop_order.remove('Bulge') if 'Bulge' in loop_order else None
    g4_stretches = [index for index, value in enumerate(secondary_structure) if value == '+']
    num_tetrads = len(g4_stretches) // 4

    pairwise_alignments = []
    if multi_block_cond >= 2 and asymmetric_block == False and g4_stretches:
        # multi-block structures - multiple separate G4 structures
        # For instance, if multi_blocK_cond == 2: 
        # Split the g4 list into twice the number of items
        # print(g4_stretches)
        g4_list = [g4_stretches[i:i+num_tetrads//multi_block_cond] for i in range(0, len(g4_stretches), num_tetrads//multi_block_cond)]
        # Block list - split the g4 list into groups of 4
        block_list = [g4_list[i:i+4] for i in range(0, len(g4_list), 4)]
        # print(g4_list)
        # Mapping G-tracts
        if loop_order == ["Propeller", "Propeller", "Propeller", "Propeller", "Propeller", "Propeller"]:
            for g4_block in block_list:
                # print(g4_block)
                # Parallel (largely only multi-block structure)
                for i in range(len(g4_block[0])):
                    pairwise_alignments.append((g4_block[0][i], g4_block[1][i]))
                    pairwise_alignments.append((g4_block[0][i], g4_block[3][i]))
                for i in range(len(g4_block[1])):
                    pairwise_alignments.append((g4_block[1][i], g4_block[2][i]))
                for i in range(len(g4_block[2])):
                    pairwise_alignments.append((g4_block[2][i], g4_block[3][i]))

    elif multi_block_cond >= 2 and asymmetric_block and g4_stretches:
        # Asymmetric block - multiple stacked G4's that do not have the same number of tetrads (rare)
        # This is a case-by-case distinction, but for this algorithm assume each of the first few tetrads are 3 in length, last is 2
        # print("check")
        # Asymmetric block - [3, 3, 2] = signifies 8 in total, 3 in first 2, 2 in last
        # Split g4 list so that every four splits, it iterates through the next value in asymetric block and changes the number of 
        # nucleotide locations in each split relavent to the next value in asymmetric block
        def split_list(input_list, lengths):
            result = []
            index = 0
            length_index = 0
            length_count = 0
            while index < len(input_list):
                current_length = lengths[length_index]
                sublist = input_list[index:index + current_length]
                result.append(sublist)
                index += current_length
                length_count += 1
                # Move to the next length after every four sublists
                if length_count == 4:
                    length_index = (length_index + 1) % len(lengths)
                    length_count = 0
            return result
        g4_list = split_list(g4_stretches, asymmetric_block)
        # print(g4_list)
        block_list = [g4_list[i:i+4] for i in range(0, len(g4_list), 4)]
        if loop_order == ["Propeller", "Propeller", "Propeller", "Propeller", "Propeller", "Propeller"]:
            for g4_block in block_list:
                # print(g4_block)
                # Parallel (largely only multi-block structure)
                for i in range(len(g4_block[0])):
                    pairwise_alignments.append((g4_block[0][i], g4_block[1][i]))
                    pairwise_alignments.append((g4_block[0][i], g4_block[3][i]))
                for i in range(len(g4_block[1])):
                    pairwise_alignments.append((g4_block[1][i], g4_block[2][i]))
                for i in range(len(g4_block[2])):
                    pairwise_alignments.append((g4_block[2][i], g4_block[3][i]))

    elif multi_block_cond == 1 and g4_stretches:
        g4_list = [g4_stretches[i:i+num_tetrads] for i in range(0, len(g4_stretches), num_tetrads)]
        ##########################################
        # Parallel
        """
            Propeller alignment:
            - Assuming continuous for all (no bulge):
            - Split the g4_streches into a 4 groups of 3 (list of len 4 with 3 elements each)
            APPEND ALL PAIRWISE RELATIONSHIPS TO LIST: 
            First row of G's - first, no, first; middle, no, middle; last, no, last
            Second row of G's - first, no; middle, no; last, no
            Third row of G's - first; middle; last
            Fourth - no need to add, as already done
            Go through each of the list of pairwise representation (i, j) or (j, i) -> add to the matrix
        """
        ##########################################

        # Parallel G4
        if loop_order == ["Propeller", "Propeller", "Propeller"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][i]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][i]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][i]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][i]))

        ##########################################
        # Antiparallel
        ##########################################

        # Chair G4
        elif loop_order == ["Lateral", "Lateral", "Lateral"]:
        
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][(-i)-1]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][(-i)-1]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][(-i)-1]))

        # Basket-Type 1 G4 
        elif loop_order == ["Lateral", "Diagonal", "Lateral"]:

            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[2][i]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[3][i]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][(-i)-1]))

        # Basket-Type 2 G4
        elif loop_order == ["Diagonal", "Propeller", "Diagonal"]:

            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[2][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][(-i)-1]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[3][(-i)-1]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][i]))

        # Basket-Type 3 G4
        elif loop_order == ["Lateral", "Propeller", "Lateral"]:
            
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][i]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][i]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][(-i)-1]))

        ##########################################
        # Hybrid
        # DISCLAIMER - Hybrid G4' are more complex as they're strand topology isn't symmetric, thus flipped
        # versions need to be coded
        ##########################################

        # Hybrid Group 1 - only propeller and lateral loops/no diagonal crossing over interactions
        elif loop_order == ["Propeller", "Lateral", "Lateral"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                # pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[1][i]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][i]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][(-i)-1]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][(-i)-1]))
        elif loop_order == ["Lateral", "Lateral", "Propeller"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][i]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][(-i)-1]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][i]))

        elif loop_order == ["Propeller", "Propeller", "Lateral"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][i]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][(-i)-1]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][i]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][(-i)-1]))
        elif loop_order == ["Lateral", "Propeller", "Propeller"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[3][(-i)-1]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[2][i]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][i]))
        
        # Hybrid Group 2 - diagonal crossing over interactions (Lateral, Diagonal, Propeller/Propeller, Diagonal, Lateral)
        elif loop_order == ["Propeller", "Diagonal", "Lateral"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][i]))
                pairwise_alignments.append((g4_list[0][i], g4_list[2][(-i)-1]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[3][i]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][(-i)-1]))
        elif loop_order == ["Lateral", "Diagonal", "Propeller"]:
            # Mapping G-tracts
            for i in range(len(g4_list[0])):
                pairwise_alignments.append((g4_list[0][i], g4_list[1][(-i)-1]))
                pairwise_alignments.append((g4_list[0][i], g4_list[2][i]))
            for i in range(len(g4_list[1])):
                pairwise_alignments.append((g4_list[1][i], g4_list[3][(-i)-1]))
            for i in range(len(g4_list[2])):
                pairwise_alignments.append((g4_list[2][i], g4_list[3][i]))
    # print(f"Pairwise alignments: {pairwise_alignments}")
    # print(f"Secondary matrix size: {secondary_matrix.shape}")
    alignment_score = 6
    if pairwise_alignments:
        for alignment in pairwise_alignments:
            try: 
                secondary_matrix[alignment[0], alignment[1]] = alignment_score - 1 
                secondary_matrix[alignment[1], alignment[0]] = alignment_score - 1 
            except IndexError:
                continue
    # Expand the matrix to the residue size instead of sequence size
    residue_list = gen_residues(primary_sequence_list)
    secondary_residue_matrix = np.zeros((len(''.join(residue_list)), len(''.join(residue_list))))
    idx_2 = 0
    for idx_y, residue_y in enumerate(residue_list):
        idx_1 = 0  # Reset idx_1 for each new residue_y
        for idx_x, residue_x in enumerate(residue_list):
            if secondary_matrix[idx_x, idx_y] == int(alignment_score-1):
                alignment_score_matrix = np.full((len(residue_x), len(residue_y)), alignment_score)
                secondary_residue_matrix[idx_1:idx_1+len(residue_x), idx_2:idx_2+len(residue_y)] = alignment_score_matrix
            idx_1 += len(residue_x)  # Increment idx_1 for the next residue_x
        idx_2 += len(residue_y)  # Increment idx_2 for the next residue_y     
    return secondary_residue_matrix

class get_G4_secondary_structure():
    def __init__(self, 
                 primary_sequence, 
                 secondary_structure, 
                 loop_order, 
                 multi_block_cond, 
                 asymmetric_block):
        self.primary_sequence = primary_sequence
        self.secondary_structure = secondary_structure
        self.loop_order = loop_order
        self.multi_block_cond = multi_block_cond
        self.asymmetric_block = asymmetric_block

    def gen_distance_matrix(self):
        return gen_distance_matrix("".join(self.primary_sequence), 
                                   self.primary_sequence,
                                   self.secondary_structure, 
                                   self.loop_order, 
                                   self.multi_block_cond, 
                                   self.asymmetric_block)
    
