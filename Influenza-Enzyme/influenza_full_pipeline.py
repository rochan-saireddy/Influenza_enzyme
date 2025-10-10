import os
import subprocess
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.docking import DockMCMProtocol
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import Interface
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIG ===
base_dir = "Matrix Protein 1/Results"
ligandmpnn_path = "LigandMPNN/protein_mpnn_run.py"

antigens = {
    "M1": "Matrix Protein 1/Results/cluster1_1.pdb",
    "HA": "HA/Results/clusterHA_1.pdb"  # replace with your HA pdb path
}

enzyme_chain = "A"
antigen_chain = "B"
num_designs = 20
max_rounds = 10
convergence_threshold = 1.0  # kcal/mol ΔG improvement
num_threads = 4
top_k = 2  # keep top 2 designs per round

# Initialize PyRosetta
init("-mute all")

# === FUNCTIONS ===
def run_ligandmpnn(input_pdb, out_folder, num_designs):
    os.makedirs(out_folder, exist_ok=True)
    cmd = [
        "python3", ligandmpnn_path,
        "--pdb_path", input_pdb,
        "--chain_id_json", json.dumps({"enzyme": enzyme_chain, "antigen": antigen_chain}),
        "--num_seq_per_target", str(num_designs),
        "--out_folder", out_folder,
        "--cuda"
    ]
    print(f"[+] Running LigandMPNN on {input_pdb}")
    subprocess.run(cmd, check=True)

def redock_and_score(pdb_file):
    try:
        pose = pose_from_pdb(pdb_file)
        relax = FastRelax()
        relax.apply(pose)

        dock = DockMCMProtocol()
        dock.set_scorefxn(get_fa_scorefxn())
        dock.apply(pose)

        interface = Interface(enzyme_chain, antigen_chain)
        interface.calculate(pose)
        dG = interface.get_binding_energy()

        out_file = pdb_file.replace(".pdb", "_relaxed_docked.pdb")
        pose.dump_pdb(out_file)
        return dG, out_file
    except Exception as e:
        print(f"[!] Error processing {pdb_file}: {e}")
        return float("inf"), pdb_file

def evaluate_designs_parallel(pdb_folder):
    pdbs = [os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_pdb = {executor.submit(redock_and_score, pdb): pdb for pdb in pdbs}
        for future in as_completed(future_to_pdb):
            dG, out_file = future.result()
            results.append((dG, out_file))
            print(f"[✓] {out_file} ΔG: {dG:.2f}")

    results.sort(key=lambda x: x[0])
    return results

def plot_and_save_results(antigen_name, history):
    rounds = []
    top1_dG = []
    top2_dG = []

    summary_rows = []

    for entry in history:
        round_num = entry["round"]
        rounds.append(round_num)

        t1 = entry["top_designs"][0]["ΔG"]
        top1_dG.append(t1)
        summary_rows.append({
            "Round": round_num,
            "Rank": 1,
            "ΔG": t1,
            "PDB": entry["top_designs"][0]["pdb"]
        })

        if len(entry["top_designs"]) > 1:
            t2 = entry["top_designs"][1]["ΔG"]
            top2_dG.append(t2)
            summary_rows.append({
                "Round": round_num,
                "Rank": 2,
                "ΔG": t2,
                "PDB": entry["top_designs"][1]["pdb"]
            })
        else:
            top2_dG.append(None)

    # Create plot folder
    plot_dir = os.path.join(base_dir, antigen_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Line plot of ΔG per round
    plt.figure()
    plt.plot(rounds, top1_dG, "-o", label="Top 1 ΔG")
    plt.plot(rounds, top2_dG, "-o", label="Top 2 ΔG")
    plt.xlabel("Round")
    plt.ylabel("Interface ΔG (kcal/mol)")
    plt.title(f"{antigen_name} ΔG per Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{antigen_name}_ΔG_plot.png"))
    plt.close()

    # Save summary table
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(plot_dir, f"{antigen_name}_summary.csv"), index=False)
    print(f"[✓] Saved plot and summary for {antigen_name} in {plot_dir}")

# === MAIN PIPELINE ===
if __name__ == "__main__":
    for antigen_name, input_pdb in antigens.items():
        print(f"\n=== STARTING DESIGN PIPELINE FOR {antigen_name} ===")

        current_inputs = [input_pdb]
        last_best_score = float("inf")
        history = []

        for round_num in range(1, max_rounds + 1):
            print(f"\n=== {antigen_name} ROUND {round_num} ===")
            round_dir = os.path.join(base_dir, antigen_name, f"Round_{round_num}")
            os.makedirs(round_dir, exist_ok=True)

            # Generate mutants for all current inputs
            mutant_dirs = []
            for idx, inp in enumerate(current_inputs):
                inp_dir = os.path.join(round_dir, f"input_{idx+1}")
                run_ligandmpnn(inp, inp_dir, num_designs)
                mutant_dirs.append(inp_dir)

            # Dock & score all mutants
            all_results = []
            for d in mutant_dirs:
                all_results.extend(evaluate_designs_parallel(d))

            # Save top_k designs
            top_designs = all_results[:top_k]
            current_inputs = []
            for rank, (score, pdb_file) in enumerate(top_designs, start=1):
                best_copy = os.path.join(round_dir, f"top{rank}_round{round_num}.pdb")
                shutil.copy(pdb_file, best_copy)
                current_inputs.append(best_copy)
                print(f"[✓] Saved top{rank} ΔG: {score:.2f} -> {best_copy}")

            history.append({
                "round": round_num,
                "top_designs": [{"ΔG": s[0], "pdb": s[1]} for s in top_designs]
            })

            # Save history
            history_file = os.path.join(base_dir, antigen_name, "design_history.json")
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

            # Convergence check
            improvement = last_best_score - top_designs[0][0]
            if improvement < convergence_threshold:
                print(f"[!] Convergence reached for {antigen_name} (ΔΔG={improvement:.2f}). Stopping.")
                break
            last_best_score = top_designs[0][0]

        # Generate plots & summary CSV
        plot_and_save_results(antigen_name, history)

        print(f"\n=== {antigen_name} DESIGN COMPLETE ===")
        print(f"Total rounds: {len(history)}")
        print(f"Final top design ΔG: {history[-1]['top_designs'][0]['ΔG']:.2f}")
        print(f"Final best structure: {history[-1]['top_designs'][0]['pdb']}")
