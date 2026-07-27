# Converts original CG .pdb files into compatible format for Open3SPN2 MD simulation tool
import sys

def parse_pdb(filename):
    atoms = []
    conect = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms.append(line)
            elif line.startswith('CONECT') or line.startswith('END'):
                conect.append(line)
    return atoms, conect

def group_nucleotides(atoms):
    # Group atoms by residue number (column 22:26)
    from collections import defaultdict
    residues = defaultdict(list)
    for atom in atoms:
        resnum = int(atom[22:26])
        residues[resnum].append(atom)
    return [residues[k] for k in sorted(residues.keys())]

def reorder_to_ns_nps(nucleotides):
    """
    Output: NS NPS NPS NPS ...
    """
    new_atoms = []
    for i, nuc in enumerate(nucleotides):
        base = [a for a in nuc if a[12:16].strip() in {'A', 'T', 'G', 'C', 'U'}]
        sugar = [a for a in nuc if a[12:16].strip() == 'S']
        phosphate = [a for a in nuc if a[12:16].strip() == 'P']
        if i == 0:
            # First nucleotide: base, sugar
            new_atoms.extend(base + sugar)
        else:
            # All others: base, phosphate, sugar
            new_atoms.extend(base + phosphate + sugar)
    return new_atoms

def dna_resname(line):
    # Residue name is columns 17:20 (0-based index 17:20)
    resname = line[17:20].strip()
    mapping = {'A': 'DA', 'G': 'DG', 'C': 'DC', 'T': 'DT', 'U': 'DU'}
    if resname in mapping:
        new_res = mapping[resname]
        # Pad to 3 chars for PDB format
        new_res = new_res.ljust(3)
        return line[:17] + new_res + line[20:]
    else:
        return line

# def fix_element_column(line):
#     # Only modify ATOM lines
#     if not line.startswith("ATOM"):
#         return line
#     # Get the original element (last column, typically columns 76-78)
#     orig_elem = line[76:78].strip()
#     mapping = {'C': 'O', 'G': 'C', 'A': 'N', 'T': 'S', 'S': 'H', 'P': 'P'}
#     new_elem = mapping.get(orig_elem, orig_elem)
#     # Pad to 2 chars, right-aligned
#     new_elem = new_elem.rjust(2)
#     # Replace in line (columns 76-78)
#     return line[:76] + new_elem + line[78:]
def fix_element_column(line):
    # Only modify ATOM lines
    if not line.startswith("ATOM"):
        return line

    # Determine atom name (columns 12-16) and original element (76-78)
    atom_name = line[12:16].strip()
    orig_elem = line[76:78].strip()

    # Force all bases (A, C, G, T, U) to element C
    if atom_name in {"A", "C", "G", "T", "U"}:
        new_elem = "C"
    elif atom_name == "S":
        # Sugar bead
        new_elem = "H"
    elif atom_name == "P":
        # Phosphate bead
        new_elem = "P"
    else:
        # Leave anything else unchanged
        new_elem = orig_elem

    # Pad to 2 chars, right-aligned
    new_elem = new_elem.rjust(2)
    # Replace in line (columns 76-78)
    return line[:76] + new_elem + line[78:]

def convert_pdb(input_pdb, output_pdb, is_dna=True):
    atoms, conect = parse_pdb(input_pdb)
    nucleotides = group_nucleotides(atoms)
    new_atoms = reorder_to_ns_nps(nucleotides)
    with open(output_pdb, 'w') as f:
        for line in new_atoms:
            if is_dna:
                fixed = fix_element_column(dna_resname(line))
            else:
                fixed = fix_element_column(line)
            f.write(fixed)

# extract model + the conect sticks 'for better output visualization'
def extract_model_with_conect(input_pdb, model_number, output_basename, base_pdb=None,
                              include_conect=True, is_dna=True):
    """
    Extract MODEL <model_number> from a multi-model PDB (input_pdb) and write:
        f"{output_basename}_model{model_number}.pdb"
    - Reorders atoms to SNPSNPSNP...SN (reverse orientation): S+N for last residue, S+N+P otherwise.
    - If base_pdb is provided, CONECT records are taken from that file (original PDB),
      remapped to the model's atom serials using (resSeq, atomName), and written before END.
    """
    atoms_block = []
    in_target = False

    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                current = line[5:].strip()
                in_target = (current == str(model_number))
            elif line.startswith("ENDMDL"):
                if in_target:
                    break
            elif in_target and (line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER")):
                atoms_block.append(line)

    if not atoms_block:
        raise ValueError(f"Model {model_number} not found in {input_pdb}")

    # Build mapping for model atoms: key -> new serial
    # key = (resSeq, atomName)
    def parse_atom(line):
        serial = int(line[6:11])
        name = line[12:16].strip()
        resseq = int(line[22:26])
        return serial, name, resseq

    model_key_to_serial = {}
    for line in atoms_block:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            serial, name, resseq = parse_atom(line)
            model_key_to_serial[(resseq, name)] = serial

    # --- Reorder to SNPSNPSNP...SN ---
    def reorder_to_snp_sn_from_atoms(atom_lines):
        # Group by residue (based on ATOM lines only)
        nucleotides = group_nucleotides(atom_lines)
        new_atoms = []
        rev = list(reversed(nucleotides))
        n = len(rev)
        for j, nuc in enumerate(rev):
            base = [a for a in nuc if a[12:16].strip() in {'A', 'C', 'G', 'T', 'U'}]
            sugar = [a for a in nuc if a[12:16].strip() == 'S']
            phosphate = [a for a in nuc if a[12:16].strip() == 'P']
            if j == n - 1:
                # Last in output: S, N
                new_atoms.extend(sugar + base)
            else:
                # Others: S, N, P
                new_atoms.extend(sugar + base + phosphate)
        return new_atoms

    atom_only = [l for l in atoms_block if l.startswith("ATOM")]
    # Keep any non-ATOM (e.g., HETATM) if present; here we focus on nucleic acid coarse atoms.
    reordered_atoms = reorder_to_snp_sn_from_atoms(atom_only)
    # print(f"Reordered {reordered_atoms} atoms for model {model_number}")
    # Prepare CONECT remapping from base PDB (optional)
    remapped_conect = []
    if include_conect and base_pdb:
        base_atoms = {}
        base_conect_lines = []
        with open(base_pdb, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    serial, name, resseq = parse_atom(line)
                    base_atoms[serial] = (resseq, name)
                elif line.startswith("CONECT"):
                    base_conect_lines.append(line)

        for line in base_conect_lines:
            parts = line.strip().split()
            raw_serials = [int(p) for p in parts[1:]]
            new_serials = []
            valid = True
            for s in raw_serials:
                if s not in base_atoms:
                    valid = False
                    break
                key = base_atoms[s]  # (resseq, name)
                if key not in model_key_to_serial:
                    valid = False
                    break
                new_serials.append(model_key_to_serial[key])
            if valid and new_serials:
                remapped = "CONECT" + "".join(f"{s:5d}" for s in new_serials) + "\n"
                remapped_conect.append(remapped)

    output_pdb = f"{output_basename}_model{model_number}.pdb"
    with open(output_pdb, 'w') as out:
        # Write reordered atoms
        for line in reordered_atoms:
            fixed = fix_element_column(dna_resname(line)) if is_dna else fix_element_column(line)
            out.write(fixed)
        # Single TER, then CONECT, then END
        out.write("TER\n")
        for c in remapped_conect:
            out.write(c if c.endswith("\n") else c + "\n")
        out.write("END\n")
    return output_pdb

def scale_pdb_coordinates(input_path, output_path, factor=10.0):
    """
    Scale all ATOM/HETATM xyz coordinates in a PDB by a constant factor.
    Preserves formatting and all non-coordinate records (MODEL/ENDMDL/TER/CONECT/etc.).
    """
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if (line.startswith('ATOM') or line.startswith('HETATM')) and len(line) >= 54:
                try:
                    x = float(line[30:38]) * factor
                    y = float(line[38:46]) * factor
                    z = float(line[46:54]) * factor
                    new_line = (
                        f"{line[:30]}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{line[54:]}"
                    )
                    f_out.write(new_line)
                except ValueError:
                    # If parsing fails, write original line unchanged
                    f_out.write(line)
            else:
                f_out.write(line)

# min, max, 3D grid -> put K+ ions in the grid space + filter out any assigned K+
pdb_file_name = 'PDB_file_name'
# input_file = './open3spn2_files/dec15-25/' + pdb_file_name + '.pdb'
# output_file = './open3spn2_files/dec15-25/' + pdb_file_name + '_reordered.pdb'
input_file = './open3SPN2_final/!experiments/' + pdb_file_name + '.pdb'
output_file = './open3SPN2_final/!experiments/' + pdb_file_name + '_reordered.pdb'

if __name__ == "__main__":
    # Set is_dna to False for RNA, True for DNA
    is_dna = True  # Change to False for RNA
    convert_pdb(input_file, output_file, is_dna=is_dna)
    print(f"Converted {input_file} to {output_file}")