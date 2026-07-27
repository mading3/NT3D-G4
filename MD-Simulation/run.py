from pathlib import Path

import open3SPN2
import openmm
import openmm.app
from openmm import unit
import sys
import matplotlib.pyplot as plt
import pandas as pd
import heapq
import signal
import threading
import shutil
from typing import Optional, Any
try:
    from openmmplumed import PlumedForce  # OpenMM-PLUMED plugin
    HAS_PLUMED = True
except Exception:
    HAS_PLUMED = False


# --- Console Output Logging ---
class LogTee(object):
    """Splits stdout/stderr to both the terminal and a local file."""
    def __init__(self, filename):
        self.terminal = sys.__stdout__  # Use original stdout
        self.stderr = sys.__stderr__
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filename, "a", buffering=1, encoding='utf-8')  # Line buffered
        self.closed = False
        
    def write(self, message):
        try:
            if not self.closed:
                self.terminal.write(message)
                self.terminal.flush()
                if self.log and not self.log.closed:
                    self.log.write(message)
                    self.log.flush()
            else:
                self.terminal.write(message)
        except Exception:
            pass  # Avoid exceptions during exception handling

    def flush(self):
        try:
            if not self.closed:
                self.terminal.flush()
                if self.log and not self.log.closed:
                    self.log.flush()
        except Exception:
            pass

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                if self.log and not self.log.closed:
                    self.log.close()
            except Exception:
                pass
    
    def __del__(self):
        self.close()

##
# Configuration
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "INPUT"
OUTPUT_DIR = BASE_DIR / "OUTPUT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint configuration
CHECKPOINT_INTERVAL = 50000  # Save checkpoint every N steps
#THIS IS NORMAL STEPS
#TOTAL_STEPS = 10000000
#SHORTER STEPS FOR TESTING
TOTAL_STEPS = 1000000

STEPS_PER_REPORT = 1000

NANOMETER = getattr(unit, "nanometer")
KELVIN = getattr(unit, "kelvin")
KILOJOULE_PER_MOLE = getattr(unit, "kilojoule_per_mole")

K_VALUE = 100  # Restraint force constant in kJ/mol/nm^2
TOLERANCE_VALUE = 100

# Mesh restraints configuration (using PLUMED with native OpenMM backend)
ENABLE_PLUMED_PLUS_MESH = True  # Toggle to enable/disable mesh on '+' residues
# Strong default wall constants (kJ/mol/A^2). Applied via PLUMED harmonic walls.
PLUMED_KAPPA_PP = 1000.0
PLUMED_KAPPA_SS = 1000.0
PLUMED_KAPPA_BB = 1000.0

# Global for signal handling
CURRENT_SIMULATION = None
CURRENT_FILE_NAME = None
LOG_TEE = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    if CURRENT_SIMULATION and CURRENT_FILE_NAME:
        try:
            # Save checkpoint and progress before exiting
            output_file_prefix = OUTPUT_DIR / f"{CURRENT_FILE_NAME}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"
            checkpoint_file = output_file_prefix.with_suffix(".chk")
            temp_chk = output_file_prefix.with_suffix(".chk.tmp")
            
            print("Saving checkpoint...")
            with open(temp_chk, "wb") as f:
                f.write(CURRENT_SIMULATION.context.createCheckpoint())
            temp_chk.replace(checkpoint_file)
            
            current_step = CURRENT_SIMULATION.currentStep
            save_progress(CURRENT_FILE_NAME, current_step)
            print(f"Progress saved at step {current_step}")
        except Exception as e:
            print(f"Error during shutdown: {e}")
    
    if LOG_TEE:
        LOG_TEE.close()
    
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class LowestEnergyTracker:
    """Track the lowest N energy structures during simulation."""
    
    def __init__(self, n=5):
        self.n = n
        self.structures = []  # Max heap of (negative_energy, step, positions)
    
    def add_structure(self, energy, step, positions):
        """Add a structure if it's among the lowest N energies."""
        # Use negative energy for max heap (to track minimum)
        neg_energy = -energy
        
        if len(self.structures) < self.n:
            heapq.heappush(self.structures, (neg_energy, step, positions.copy()))
        elif neg_energy > self.structures[0][0]:
            heapq.heapreplace(self.structures, (neg_energy, step, positions.copy()))
    
    def get_sorted_structures(self):
        """Return structures sorted by energy (lowest first)."""
        # Sort by energy (remember to negate back)
        return sorted([(-neg_e, step, pos) for neg_e, step, pos in self.structures])


class LowestEnergyReporter:
    """Custom reporter to track lowest energy structures."""
    
    def __init__(self, tracker, reportInterval):
        self.tracker = tracker
        self.reportInterval = reportInterval
    
    def describeNextReport(self, simulation):
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, False, False, False, True, None)
    
    def report(self, simulation, state):
        energy = state.getPotentialEnergy().value_in_unit(KILOJOULE_PER_MOLE)
        step = simulation.currentStep
        positions = state.getPositions()
        self.tracker.add_structure(energy, step, positions)


class AppendablePDBReporter:
    """Custom PDB reporter that supports append mode for resuming simulations."""
    
    def __init__(self, file, reportInterval, append=False):
        self._reportInterval = reportInterval
        self._append = append
        self._out = None
        self._topology = None
        self._nextModel = 0
        
        if append:
            # Count existing models if appending
            if Path(file).exists():
                with open(file, 'r') as f:
                    for line in f:
                        if line.startswith('MODEL'):
                            self._nextModel += 1
            self._file = file
        else:
            self._file = file
    
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)
    
    def report(self, simulation, state):
        if self._topology is None:
            self._topology = simulation.topology
        
        # Open file in append mode if resuming, otherwise write mode
        mode = 'a' if self._append and self._nextModel > 0 else 'w' if self._nextModel == 0 else 'a'
        
        with open(self._file, mode) as out:
            openmm.app.PDBFile.writeHeader(self._topology, out)
            openmm.app.PDBFile.writeModel(self._topology, state.getPositions(), out, self._nextModel)
        
        self._nextModel += 1
        
        # After first write in append mode, keep appending
        if self._append:
            self._append = True


def add_backbone_restraints(system, file_name):
    """Add harmonic restraints to backbone particles."""
    k_restraint = K_VALUE * KILOJOULE_PER_MOLE / NANOMETER**2
    restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", k_restraint)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = list(system.coord.positions)
    #backbone_names = {"P", "S"}

    # Reading dot bracket file to get restrained atoms
    dotbracket_file = INPUT_DIR / f"{file_name}.txt"
    dotbracket_str = dotbracket_file.read_text().strip()

    i = 0
    flag = False
    nuc_base = ['A', 'C', 'G', 'T']
    backbone_names = ["P", "S"]
    dotbracket_interaction = ['+','(',')']
    for atom in system.top.atoms():
        print(atom.name, end="")
        #if atom.name in backbone_names:
        if atom.name in nuc_base:
            if dotbracket_str[i] in dotbracket_interaction:
                flag = True
                print("*", end="")
                pos = positions[atom.index]
                restraint.addParticle(atom.index, [pos[0], pos[1], pos[2]])
            i += 1
        if atom.name in backbone_names and flag:
            print("*", end="")
            pos = positions[atom.index]
            restraint.addParticle(atom.index, [pos[0], pos[1], pos[2]])
            flag = False
    print()

    system.addForce(restraint)


def add_neighbor_base_restraints(system):
    """Add harmonic bond force between neighboring bases to limit distance to 6 Angstroms."""
    print("Adding neighbor base restraints (6 Angstroms)...")
    k_bond = K_VALUE * KILOJOULE_PER_MOLE / NANOMETER**2
    r0 = 5.5 * unit.angstrom
    bond_force = openmm.HarmonicBondForce()
    
    nuc_base = ['A', 'C', 'G', 'T']
    
    # Iterate over chains and residues to find neighbors
    count = 0
    for chain in system.top.chains():
        prev_base_index = None
        for residue in chain.residues():
            base_atom = None
            for atom in residue.atoms():
                if atom.name in nuc_base:
                    base_atom = atom
                    break
            
            if base_atom:
                if prev_base_index is not None:
                    bond_force.addBond(prev_base_index, base_atom.index, r0, k_bond)
                    count += 1
                prev_base_index = base_atom.index
    
    print(f"Added {count} neighbor base restraints.")
    system.addForce(bond_force)


def _collect_plus_bead_groups(system, file_name):
    """Collect atom indices (1-based) for P/S/Base of residues whose base is marked '+' in dot-bracket.

    Returns:
        tuple(list[int], list[int], list[int]) -> (gP, gS, gB)
    """
    dotbracket_file = INPUT_DIR / f"{file_name}.txt"
    if not dotbracket_file.exists():
        raise FileNotFoundError(f"Dot-bracket file not found: {dotbracket_file}")

    dotbracket_str = dotbracket_file.read_text().strip()
    nuc_base = ['A', 'C', 'G', 'T']

    gP, gS, gB = [], [], []
    i = 0  # position along dot-bracket string (per base residue)

    for chain in system.top.chains():
        for residue in chain.residues():
            # find base atom in this residue
            base_atom = None
            for atom in residue.atoms():
                if atom.name in nuc_base:
                    base_atom = atom
                    break

            if base_atom is None:
                # This residue may be incomplete; skip from dot-bracket counting.
                continue

            if i >= len(dotbracket_str):
                # Safety: avoid index error if dot-bracket shorter than residues
                break

            if dotbracket_str[i] == '+':
                # include base
                gB.append(base_atom.index + 1)  # PLUMED is 1-based
                # include P and S in same residue
                for atom in residue.atoms():
                    if atom.name == 'P':
                        gP.append(atom.index + 1)
                    elif atom.name == 'S':
                        gS.append(atom.index + 1)

            # advance to next base position
            i += 1

    return gP, gS, gB




def add_plumed_plus_mesh_restraints(system, file_name):
    """Attach PLUMED mesh restraints on '+' residues with same-type pair distance walls.

    Enforces same-type pair distance walls among all '+' residue beads (P, S, Base).
    Distance ranges (Angstrom):
      - P–P:  lower 5.0,  upper 26.0
      - S–S:  lower 3.5,  upper 21.0
      - B–B:  lower 3.0,  upper 11.0
    Uses PLUMED RESTRAINT actions with harmonic walls.
    """
    if not ENABLE_PLUMED_PLUS_MESH:
        print("'+' mesh restraints disabled (ENABLE_PLUMED_PLUS_MESH=False).")
        return

    if not HAS_PLUMED:
        print("openmmplumed not available; skipping PLUMED '+' mesh restraints.\n"
              "Install: pip install openmm-plumed (and ensure PLUMED 2.7+ is available).")
        return

    gP, gS, gB = _collect_plus_bead_groups(system, file_name)
    print(f"PLUMED '+': counts -> P:{len(gP)} S:{len(gS)} B:{len(gB)}")

    # Convert 1-based PLUMED indices to 0-based for atom referencing (OpenMM will handle internally)
    # Keep as 1-based for PLUMED script
    
    # Distance limits in Angstrom and force constants
    limits = {
        'PP': (5.0, 26.0, PLUMED_KAPPA_PP, gP),
        'SS': (3.5, 21.0, PLUMED_KAPPA_SS, gS),
        'BB': (3.0, 11.0, PLUMED_KAPPA_BB, gB),
    }

    lines = []
    lines.append("UNITS LENGTH=A ENERGY=kj/mol")
    
    pair_count = 0
    
    for pair_type, (r_min, r_max, kappa, group) in limits.items():
        if len(group) < 2:
            continue
        
        pair_index = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                atom_i = group[i]  # Already 1-based
                atom_j = group[j]  # Already 1-based
                
                # Create distance variable for this pair
                d_var = f"d{pair_type}_{pair_index}"
                lines.append(f"{d_var}: DISTANCE ATOMS={atom_i},{atom_j} NOPBC")
                
                # Apply lower and upper wall restraints
                lower_label = f"w{pair_type}_low_{pair_index}"
                upper_label = f"w{pair_type}_up_{pair_index}"
                lines.append(f"{lower_label}: LOWER_WALLS ARG={d_var} AT={r_min} KAPPA={kappa} EXP=2 EPS=1")
                lines.append(f"{upper_label}: UPPER_WALLS ARG={d_var} AT={r_max} KAPPA={kappa} EXP=2 EPS=1")
                
                pair_index += 1
                pair_count += 1

    plumed_str = "\n".join(lines) + "\n"
    
    if pair_count == 0:
        print("No '+' residue pairs found; skipping PLUMED mesh restraints.")
        return

    # Write the PLUMED script to OUTPUT for inspection
    output_file_prefix = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"
    plumed_file = output_file_prefix.with_suffix('.plumed.dat')
    with open(plumed_file, 'w') as f:
        f.write(plumed_str)
    print(f"Wrote PLUMED script ({pair_count} pairs) to {plumed_file}")

    # Write summary
    summary_file = output_file_prefix.with_suffix('.mesh.txt')
    with open(summary_file, 'w') as f:
        f.write("PLUMED '+' mesh restraints:\n")
        for pair_type, (r_min, r_max, kappa, group) in limits.items():
            n_pairs = len(group) * (len(group) - 1) // 2 if len(group) >= 2 else 0
            f.write(f"{pair_type}: {n_pairs} pairs, r ∈ [{r_min}, {r_max}] Å, k={kappa} kJ/mol/Å²\n")
        f.write(f"Total pairs: {pair_count}\n")
    print(f"Mesh summary written to {summary_file}")

    # Attach force
    try:
        system.addForce(PlumedForce(plumed_str))
        print(f"PlumedForce attached with {pair_count} pair restraints.")
    except Exception as e:
        print(f"Error attaching PlumedForce: {e}")
        raise


def load_checkpoint(simulation, file_name):
    """Use OpenMM's native checkpoint to restore state if available."""
    # Construct path matching the reporter's output filename
    output_file_prefix = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"
    checkpoint_file = output_file_prefix.with_suffix(".chk")
    
    if checkpoint_file.exists():
        print(f"Found checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            data = f.read()
        simulation.context.loadCheckpoint(data)
        return checkpoint_file
    
    # Fallback to check for simple filename (backward compatibility or if naming changed)
    simple_checkpoint = OUTPUT_DIR / f"{file_name}.chk"
    if simple_checkpoint.exists():
        print(f"Found checkpoint file (simple name): {simple_checkpoint}")
        with open(simple_checkpoint, "rb") as f:
            data = f.read()
        simulation.context.loadCheckpoint(data)
        return simple_checkpoint
        
    return None


def get_progress_file(file_name):
    """Get path to progress tracking file."""
    return OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}_progress.txt"


def load_progress(file_name):
    """Load the current step from progress file."""
    progress_file = get_progress_file(file_name)
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return int(f.read().strip())
    return 0


def save_progress(file_name, current_step):
    """Save current step to progress file."""
    progress_file = get_progress_file(file_name)
    with open(progress_file, 'w') as f:
        f.write(str(current_step))


def count_pdb_frames(pdb_file):
    """Count the number of MODEL frames in a PDB file."""
    if not pdb_file.exists():
        return 0
    
    count = 0
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                count += 1
    return count


def save_lowest_energy_structures(tracker, topology, file_name):
    """Save the lowest 5 energy structures to a combined PDB file."""
    output_file = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}_lowest5.pdb"
    
    # Construct paths to full trajectory and energy file
    output_file_prefix = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"
    pdb_file = output_file_prefix.with_suffix(".pdb")
    energy_file = output_file_prefix.with_suffix(".csv")
    
    if not pdb_file.exists() or not energy_file.exists():
        print(f"Cannot extract lowest energy structures: {pdb_file} or {energy_file} missing.")
        return

    try:
        # Read energy file
        df = pd.read_csv(energy_file)
        
        # Identify columns
        step_col = '#"Step"'
        energy_col = 'Potential Energy (kJ/mole)'
        
        # Handle potential column name variations
        if step_col not in df.columns:
             # Try to find step column
             for c in df.columns:
                 if 'Step' in c:
                     step_col = c
                     break
        
        if energy_col not in df.columns:
             for c in df.columns:
                 if 'Potential Energy' in c:
                     energy_col = c
                     break
        
        if step_col not in df.columns or energy_col not in df.columns:
            print(f"Could not find required columns in {energy_file}. Columns: {df.columns}")
            return

        # Sort and get top 5
        sorted_df = df.sort_values(by=energy_col).head(5)
        target_indices = sorted_df.index.tolist() # 0-based indices corresponding to frames
        
        print(f"\nLowest 5 energy structures (extracted from trajectory):")
        
        # Extract frames from PDB
        extracted_frames = {} # index -> list of lines
        header_lines = []
        
        current_model_idx = -1
        in_model = False
        buffer = []
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('MODEL'):
                    current_model_idx += 1
                    in_model = True
                    buffer = []
                elif line.startswith('ENDMDL'):
                    in_model = False
                    if current_model_idx in target_indices:
                        extracted_frames[current_model_idx] = buffer[:]
                elif in_model:
                    if current_model_idx in target_indices:
                        buffer.append(line)
                else:
                    # Header lines (before first MODEL)
                    if current_model_idx == -1:
                        header_lines.append(line)
        
        # Write output
        with open(output_file, 'w') as f:
            # Write header
            for line in header_lines:
                f.write(line)
            
            for i, (idx, row) in enumerate(sorted_df.iterrows()):
                step = int(row[step_col])
                energy = row[energy_col]
                
                print(f"  Model {i+1}: Step {step}, Energy = {energy:.2f} kJ/mole")
                
                if idx in extracted_frames:
                    f.write(f"MODEL     {i+1:4d}\n")
                    f.write(f"REMARK Energy: {energy:.6f} kJ/mole, Step: {step}\n")
                    for line in extracted_frames[idx]:
                        f.write(line)
                    f.write("ENDMDL\n")
                else:
                    print(f"  Warning: Frame {idx} not found in PDB file.")
            
            f.write("END\n")
            
        print(f"Saved lowest 5 energy structures to {output_file}")

    except Exception as e:
        print(f"Error extracting lowest energy structures: {e}")
        import traceback
        traceback.print_exc()


def plot_energy(energy_file: Path, file_name: str):
    """Plot potential energy vs step from simulation data."""
    try:
        # Read energy data
        df = pd.read_csv(energy_file)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot potential energy
        ax1.plot(df['#"Step"'], df['Potential Energy (kJ/mole)'], 'b-', linewidth=0.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Potential Energy (kJ/mole)')
        ax1.set_title(f'{file_name} - Potential Energy vs Step')
        ax1.grid(True, alpha=0.3)
        
        # Plot temperature
        ax2.plot(df['#"Step"'], df['Temperature (K)'], 'r-', linewidth=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_title(f'{file_name} - Temperature vs Step')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_file = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}_energy_plot.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Energy plot saved to {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting energy: {e}")


def get_last_step_from_csv(file_name):
    """Get the last step recorded in the energy CSV file."""
    output_file_prefix = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"
    energy_file = output_file_prefix.with_suffix(".csv")
    
    if not energy_file.exists():
        return 0
        
    try:
        # Read only the last few lines to be efficient, or just read with pandas
        # Since files might be large, reading full file with pandas might be slow, 
        # but for now let's assume it fits in memory or use a more robust method if needed.
        # Using pandas is safest for parsing CSV correctly.
        df = pd.read_csv(energy_file)
        
        # Identify step column
        step_col = '#"Step"'
        if step_col not in df.columns:
             for c in df.columns:
                 if 'Step' in c:
                     step_col = c
                     break
        
        if step_col in df.columns and not df.empty:
            return int(df[step_col].iloc[-1])
            
    except Exception as e:
        print(f"Warning: Could not read last step from {energy_file}: {e}")
        
    return 0

def run_simulation(file_path: Path):
    file_name = file_path.stem
    print(f"\n=== Processing {file_path.name} ===")
    
    # Check if we're resuming based on CSV progress
    csv_steps = get_last_step_from_csv(file_name)
    progress_steps = load_progress(file_name)
    
    # Use the larger of the two, or prefer CSV if available as it's the actual data
    completed_steps = max(csv_steps, progress_steps)
    is_resuming = completed_steps > 0
    
    if is_resuming:
        print(f"Found existing progress: {completed_steps} steps completed (CSV: {csv_steps}, Progress: {progress_steps})")
        
        # Check if simulation is already complete
        if completed_steps >= TOTAL_STEPS:
            print(f"Simulation already completed! (Steps: {completed_steps} >= {TOTAL_STEPS})")
            # Still generate plots and lowest energy structures if needed
            output_file_prefix = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"
            energy_file = output_file_prefix.with_suffix(".csv")
            
            # Save lowest energy structures
            # We need a dummy topology or load it. Since we haven't created 'system' yet, 
            # we should probably create it to get topology, or refactor.
            # For now, let's proceed to create system as usual, but skip simulation loop.
            pass 
    
    dna = open3SPN2.DNA.fromCoarsePDB(str(file_path))
    dna.computeTopology(template_from_X3DNA=False)
    dna.periodic = False

    system = open3SPN2.System(dna, periodicBox=None)
    system.add3SPN2forces(verbose=True)
    add_backbone_restraints(system, file_name)
    add_neighbor_base_restraints(system)
    add_plumed_plus_mesh_restraints(system, file_name)

    simulation_platform = 'CUDA'
    temperature = 300 * KELVIN
    system.initializeMD(temperature=temperature, platform_name=simulation_platform)
    simulation = system.simulation
    simulation.context.setPositions(system.coord.getPositions())

    energy_unit = KILOJOULE_PER_MOLE
    state = simulation.context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy().value_in_unit(energy_unit)
    print('TotalEnergy', round(total_energy, 6), energy_unit.get_symbol())

    energies = {}
    for force_name, force in system.forces.items():
        group = force.getForceGroup()
        state = simulation.context.getState(getEnergy=True, groups=2**group)
        energies[force_name] = state.getPotentialEnergy().value_in_unit(energy_unit)

    for force_name in system.forces.keys():
        print(force_name, round(energies[force_name], 6), energy_unit.get_symbol())

    output_file_prefix = OUTPUT_DIR / f"{file_name}_sim_k{K_VALUE}_tol{TOLERANCE_VALUE}"

    # Create PDB reporter (append mode if resuming)
    pdb_file = output_file_prefix.with_suffix(".pdb")
    pdb_reporter = AppendablePDBReporter(str(pdb_file), 1000, append=is_resuming)
    
    # Energy reporter to stdout
    energy_reporter_stdout = openmm.app.StateDataReporter(
        sys.stdout,
        1000,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
    )
    
    # Energy reporter to file for plotting (append if resuming)
    energy_file = output_file_prefix.with_suffix(".csv")
    energy_reporter_file = openmm.app.StateDataReporter(
        str(energy_file),
        1000,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        append=is_resuming
    )
    
    # Lowest energy tracker
    lowest_energy_tracker = LowestEnergyTracker(n=5)
    lowest_energy_reporter = LowestEnergyReporter(lowest_energy_tracker, 1000)
    
    # Note: We'll handle checkpointing manually for atomic writes
    # checkpoint_file = output_file_prefix.with_suffix(".chk")
    # checkpoint_reporter = openmm.app.CheckpointReporter(
    #     str(checkpoint_file),
    #     CHECKPOINT_INTERVAL,
    # )
    
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(energy_reporter_stdout)
    simulation.reporters.append(energy_reporter_file)
    simulation.reporters.append(lowest_energy_reporter)

    system_xml = openmm.XmlSerializer.serialize(system)
    with open(output_file_prefix.with_suffix(".xml"), 'w+') as f:
        f.write(system_xml)

    # Check for existing checkpoint
    checkpoint_path = None
    if is_resuming:
        checkpoint_path = load_checkpoint(simulation, file_name)
        
        if checkpoint_path is None:
            # Mismatch: We think we should resume (steps > 0), but no checkpoint found.
            # Or maybe we are at step 0 but files exist?
            if completed_steps > 0:
                raise RuntimeError(f"Error: Simulation incomplete (step {completed_steps}) but checkpoint file missing. Cannot resume safely.")
            else:
                # Steps is 0, so we start new.
                print("Starting new simulation (no checkpoint found for step 0)")
                simulation.minimizeEnergy(tolerance=TOLERANCE_VALUE*unit.kilojoule_per_mole)
                simulation.context.setVelocitiesToTemperature(temperature)
        else:
             print(f"Resuming from step {completed_steps}")
    else:
        print("Starting new simulation")
        simulation.minimizeEnergy(tolerance=TOLERANCE_VALUE*unit.kilojoule_per_mole)
        simulation.context.setVelocitiesToTemperature(temperature)
        completed_steps = 0
    
    # Calculate remaining steps
    remaining_steps = TOTAL_STEPS - completed_steps
    
    # Set global references for signal handler
    global CURRENT_SIMULATION, CURRENT_FILE_NAME
    CURRENT_SIMULATION = simulation
    CURRENT_FILE_NAME = file_name
    
    if remaining_steps > 0:
        print(f"Running {remaining_steps} steps (total: {TOTAL_STEPS})")
        
        # Run simulation in chunks to save progress
        chunk_size = CHECKPOINT_INTERVAL
        steps_run = 0
        
        checkpoint_file = output_file_prefix.with_suffix(".chk")
        temp_chk = output_file_prefix.with_suffix(".chk.tmp")
        
        while steps_run < remaining_steps:
            current_chunk = min(chunk_size, remaining_steps - steps_run)
            simulation.step(current_chunk)
            steps_run += current_chunk
            
            # Atomic checkpoint save
            try:
                with open(temp_chk, "wb") as f:
                    f.write(simulation.context.createCheckpoint())
                temp_chk.replace(checkpoint_file)
                save_progress(file_name, completed_steps + steps_run)
                print(f"Checkpoint saved at step {completed_steps + steps_run}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
    else:
        print("Simulation already completed!")
    
    # Clear global references
    CURRENT_SIMULATION = None
    CURRENT_FILE_NAME = None
    
    # Save lowest energy structures
    save_lowest_energy_structures(lowest_energy_tracker, system.top, file_name)
    
    # Plot energy
    plot_energy(energy_file, file_name)
    
    # Validate output consistency
    expected_frames = (TOTAL_STEPS // 1000) + 1  # +1 for initial frame
    actual_frames = count_pdb_frames(pdb_file)
    print(f"\nPDB frames: {actual_frames} (expected: ~{expected_frames})")
    
    # Clean up checkpoint files after successful completion
    checkpoint_path = output_file_prefix.with_suffix(".chk")
    progress_file = get_progress_file(file_name)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint file: {checkpoint_path}")
    if progress_file.exists():
        progress_file.unlink()
        print(f"Removed progress file: {progress_file}")
    
    print(f"\nSimulation completed successfully!")
    print(f"Output PDB: {pdb_file}")
    print(f"Energy CSV: {energy_file}")

def main():
    # Setup console logging
    global LOG_TEE
    log_file = OUTPUT_DIR / "console.log"
    LOG_TEE = LogTee(log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        sys.stdout = LOG_TEE
        sys.stderr = LOG_TEE
        
        print(f"Simulation started at: {pd.Timestamp.now()}")
        print(f"Configuration: K={K_VALUE}, Tolerance={TOLERANCE_VALUE}, Total Steps={TOTAL_STEPS}")
        print(f"Log file: {log_file}\n")
        
        pdb_files = sorted(INPUT_DIR.glob("*.pdb"))
        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in {INPUT_DIR}")

        for pdb_file in pdb_files:  # Process only the first file
            run_simulation(pdb_file)
            
        print(f"\nSimulation session ended at: {pd.Timestamp.now()}")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Restore original stdout/stderr before closing LogTee
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if LOG_TEE:
            LOG_TEE.close()
            LOG_TEE = None

    
if __name__ == "__main__":
    main()


