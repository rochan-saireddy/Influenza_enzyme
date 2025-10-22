# TEMP: fix for np.int removed in NumPy >=1.24
import numpy as np
if not hasattr(np, "int"):
    np.int = int

import os
import subprocess
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn, pose_from_sequence
from pyrosetta.rosetta.protocols.docking import DockMCMProtocol
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover as Interface
from pyrosetta.rosetta.core.pose import append_pose_to_pose
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===
ligandmpnn_path = "../../run.py"
base_dir = "Results"
num_designs = 20
max_rounds = 10
convergence_threshold = 1.0
num_threads = 4
top_k = 2

# Input PDBs
m1_input_pdb = "Matrix Protein 1/Results/cluster1_1.pdb"
ha_darpin_pdb = "Backbones/DARPin_E3_5_Cleaned1.pdb"
ha_antigen_pdb = "HA/Hemagglutinin H5.pdb"
subtilisin_native_pdb = "Backbones/Subtilisin_Cleaned.pdb"

# Initialize PyRosetta
init("-mute all")

# === HELPER FUNCTIONS ===
def run_ligandmpnn(input_pdb, out_folder, num_designs, enzyme_chain, antigen_chain):
    os.makedirs(out_folder, exist_ok=True)
    cmd = [
        "python3", "../../run.py",
        "--pdb_path", input_pdb,
        "--out_folder", out_folder,
        "--chains_to_design", enzyme_chain,
        "--fixed_residues", antigen_chain,   # <-- change this line
        "--number_of_batches", str(num_designs),
        "--batch_size", "20",
        "--model_type", "protein_mpnn"
    ]
    subprocess.run(cmd, check=True)
    print(f"[+] ProteinMPNN redesign done for {input_pdb}")


def redock_and_score(pdb_file, enzyme_chain, antigen_chain):
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

def evaluate_designs_parallel(pdb_folder, enzyme_chain, antigen_chain):
    pdbs = [os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_pdb = {executor.submit(redock_and_score, pdb, enzyme_chain, antigen_chain): pdb for pdb in pdbs}
        for future in as_completed(future_to_pdb):
            dG, out_file = future.result()
            results.append((dG, out_file))
            print(f"[✓] {out_file} ΔG: {dG:.2f}")
    results.sort(key=lambda x: x[0])
    return results

def plot_and_save_results(base_dir, antigen_name, history):
    rounds, top1_dG, top2_dG, summary_rows = [], [], [], []
    for entry in history:
        round_num = entry["round"]
        rounds.append(round_num)
        t1 = entry["top_designs"][0]["ΔG"]
        top1_dG.append(t1)
        summary_rows.append({"Round": round_num, "Rank": 1, "ΔG": t1, "PDB": entry["top_designs"][0]["pdb"]})
        if len(entry["top_designs"]) > 1:
            t2 = entry["top_designs"][1]["ΔG"]
            top2_dG.append(t2)
            summary_rows.append({"Round": round_num, "Rank": 2, "ΔG": t2, "PDB": entry["top_designs"][1]["pdb"]})
        else:
            top2_dG.append(None)

    plot_dir = os.path.join(base_dir, antigen_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

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

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(plot_dir, f"{antigen_name}_summary.csv"), index=False)
    print(f"[✓] Saved plot and summary for {antigen_name} in {plot_dir}")

# === HA Fusion Builder ===
def build_fusion_real(darpin_pdb, subtilisin_pdb, linker_seq="GGG", out_pdb="fusion.pdb"):
    fusion_pose = pose_from_pdb(darpin_pdb)
    subtil_pose = pose_from_pdb(subtilisin_pdb)
    linker_pose = pose_from_sequence(linker_seq, "fa_standard")
    append_pose_to_pose(fusion_pose, linker_pose, new_chain=False)
    append_pose_to_pose(fusion_pose, subtil_pose, new_chain=False)
    fusion_pose.dump_pdb(out_pdb)
    print(f"[+] Fusion created: {out_pdb} (DARPin + linker + Subtilisin)")
    return out_pdb


# === M1 Pipeline ===
def m1_pipeline():
    print("\n=== STARTING M1 PIPELINE ===")
    current_inputs = [m1_input_pdb]
    last_best_score = float("inf")
    history = []
    for round_num in range(1, max_rounds+1):
        round_dir = os.path.join(base_dir, "M1", f"Round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        mutant_dirs = []
        for idx, inp in enumerate(current_inputs):
            inp_dir = os.path.join(round_dir, f"input_{idx+1}")
            run_ligandmpnn(inp, inp_dir, num_designs, enzyme_chain="A", antigen_chain="B")
            mutant_dirs.append(inp_dir)
        all_results = []
        for d in mutant_dirs:
            all_results.extend(evaluate_designs_parallel(d, "A", "B"))
        all_results.sort(key=lambda x: x[0])
        top_designs = all_results[:top_k]
        current_inputs = []
        for rank, (score, pdb_file) in enumerate(top_designs, start=1):
            best_copy = os.path.join(round_dir, f"top{rank}_round{round_num}.pdb")
            shutil.copy(pdb_file, best_copy)
            current_inputs.append(best_copy)
        history.append({"round": round_num, "top_designs": [{"ΔG": s[0], "pdb": s[1]} for s in top_designs]})
        improvement = last_best_score - top_designs[0][0]
        if improvement < convergence_threshold:
            print(f"[!] M1 Convergence reached (ΔΔG={improvement:.2f}). Stopping.")
            break
        last_best_score = top_designs[0][0]
    plot_and_save_results(base_dir, "M1", history)
    print("\n=== M1 PIPELINE COMPLETE ===")
    return history

# === HA Pipeline ===
def ha_pipeline():
    print("\n=== STARTING HA PIPELINE ===")
    linker_options = ["GGG", "GGGG", "GGGGG", "GSG", "GSAG"]
    fusion_variants = []
    fusion_dir = os.path.join(base_dir, "HA", "fusion_variants")
    os.makedirs(fusion_dir, exist_ok=True)
    # Build fusions
    for linker in linker_options:
        fusion_pdb = os.path.join(fusion_dir, f"fusion_{linker}.pdb")
        build_fusion_real(ha_darpin_pdb, subtilisin_native_pdb, linker_seq=linker, out_pdb=fusion_pdb)
        fusion_variants.append(fusion_pdb)
    # Dock + score
    docking_results = []
    for f in fusion_variants:
        dG, scored_pdb = redock_and_score(f, "B", "A")
        docking_results.append((dG, scored_pdb))
    docking_results.sort(key=lambda x: x[0])
    top_fusions = docking_results[:top_k]
    current_inputs = [pdb for _, pdb in top_fusions]
    last_best_score = float("inf")
    history = []
    for round_num in range(1, max_rounds+1):
        round_dir = os.path.join(base_dir, "HA", f"Round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        mutant_dirs = []
        for idx, inp in enumerate(current_inputs):
            inp_dir = os.path.join(round_dir, f"input_{idx+1}")
            run_ligandmpnn(inp, inp_dir, num_designs, enzyme_chain="B", antigen_chain="A")
            mutant_dirs.append(inp_dir)
        all_results = []
        for d in mutant_dirs:
            all_results.extend(evaluate_designs_parallel(d, "B", "A"))
        all_results.sort(key=lambda x: x[0])
        top_designs = all_results[:top_k]
        current_inputs = []
        for rank, (score, pdb_file) in enumerate(top_designs, start=1):
            best_copy = os.path.join(round_dir, f"top{rank}_round{round_num}.pdb")
            shutil.copy(pdb_file, best_copy)
            current_inputs.append(best_copy)
        history.append({"round": round_num, "top_designs": [{"ΔG": s[0], "pdb": s[1]} for s in top_designs]})
        improvement = last_best_score - top_designs[0][0]
        if improvement < convergence_threshold:
            print(f"[!] HA Convergence reached (ΔΔG={improvement:.2f}). Stopping.")
            break
        last_best_score = top_designs[0][0]
    plot_and_save_results(base_dir, "HA", history)
    print("\n=== HA PIPELINE COMPLETE ===")
    return history

# === Consolidated results ===
def consolidate_results(out_dir, antigens_list):
    master_rows = []
    for antigen_name in antigens_list:
        history_file = os.path.join(out_dir, antigen_name, "design_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
            for entry in history:
                round_num = entry["round"]
                for rank, d in enumerate(entry["top_designs"], start=1):
                    master_rows.append({
                        "Antigen": antigen_name,
                        "Round": round_num,
                        "Rank": rank,
                        "ΔG": d["ΔG"],
                        "PDB": d["pdb"]
                    })

    df_master = pd.DataFrame(master_rows)
    master_csv = os.path.join(out_dir, "master_top_designs.csv")
    df_master.to_csv(master_csv, index=False)
    print(f"[✓] Master CSV saved: {master_csv}")

    plt.figure()
    for antigen_name in antigens_list:
        df_sub = df_master[(df_master["Antigen"] == antigen_name) & (df_master["Rank"] == 1)]
        if df_sub.empty:
            continue
        rounds = df_sub["Round"].tolist()
        top1_dG = df_sub["ΔG"].tolist()
        plt.plot(rounds, top1_dG, "-o", label=f"{antigen_name} Top1 ΔG")
    plt.xlabel("Round")
    plt.ylabel("Interface ΔG (kcal/mol)")
    plt.title("Top ΔG per Round: M1 vs HA")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(out_dir, "master_top1_ΔG_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[✓] Master ΔG plot saved: {plot_path}")

# === MAIN ===
if __name__ == "__main__":
    # TEMP: keep threads but add option to run sequentially to avoid segfault
    use_threads = False  # <-- set True to allow concurrent
    if use_threads:
        t1 = Thread(target=m1_pipeline)
        t2 = Thread(target=ha_pipeline)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    else:
        m1_pipeline()
        ha_pipeline()

    print("\n=== ALL DESIGN PIPELINES COMPLETE ===")












