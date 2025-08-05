#!/usr/bin/env python3
"""
Run single‐residue scoring via score.py, then annotate your PDB’s B‐factors
with the rounded (3-decimal) native–AA probabilities.

Usage:
    # Single file:
    python run_score_and_annotate.py \
        /path/to/input.pdb \
        /path/to/out_base_dir \
        /path/to/annotated_base_dir

    # Directory of PDBs:
    python run_score_and_annotate.py \
        /path/to/pdb_dir \
        /path/to/out_base_dir \
        /path/to/annotated_base_dir

This script:
  1. Scans the input path: if it's a .pdb file, processes that one; if it's a directory,
     finds all .pdb files inside (non-recursively).
  2. For each PDB, creates a per-structure output folder under out_base_dir/basename
     and writes the annotated PDB to annotated_base_dir/basename_annotated.pdb.
  3. Calls score.py to compute single‐AA scores (LigandMPNN/ProteinMPNN).
  4. Loads the .pt, extracts sequence, mean_probs_dict, and res_names.
  5. Computes the probability of the native AA at each residue.
  6. Reads the input PDB and replaces each atom’s B-factor (cols 61–66)
     for those residues with the rounded score.
  7. Writes out a new annotated PDB.
"""
import argparse
import subprocess
import sys
import os
import torch
from pathlib import Path

model_type = "ligand_mpnn"  # or "protein_mpnn"


def run_score(pdb_path: str, out_folder: str):
    """Call score.py to compute single‐amino‐acid scores."""
    os.makedirs(out_folder, exist_ok=True)
    cmd = [
        sys.executable,
        "score.py",
        "--model_type",
        model_type,
        "--seed",
        "111",
        "--single_aa_score",
        "1",
        "--pdb_path",
        pdb_path,
        "--out_folder",
        out_folder,
        "--use_sequence",
        "1",
        "--batch_size",
        "1",
        "--number_of_batches",
        "10",
        # "--homo_oligomer",
        # "1",
    ]
    subprocess.run(cmd, check=True)


def find_pt_file(out_folder: str) -> str:
    """Locate the single .pt file in the scoring output folder."""
    files = [f for f in os.listdir(out_folder) if f.endswith(".pt")]
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one .pt in {out_folder}, found: {files}")
    return os.path.join(out_folder, files[0])


def load_scores(pt_path: str):
    """
    Load the .pt file and return alphabet, sequence, mean_probs, res_names
    """
    data = torch.load(pt_path, map_location="cpu")
    alphabet = data["alphabet"]
    sequence = data["sequence"]
    mean_probs = data["mean_of_probs"]
    res_names = data["residue_names"]
    return sequence, mean_probs, res_names


def get_current_res_mean_probs(sequence, mean_probs_dict, res_names):
    """
    Given:
      sequence: list of native AAs,
      mean_probs_dict: dict mapping res_id -> {AA: prob},
      res_names: dict mapping idx -> res_id,
    Return:
      scores dict mapping res_id -> rounded native-AA probability.
    """
    scores = {}
    for idx, res_id in res_names.items():
        native_aa = sequence[idx]
        prob = mean_probs_dict.get(res_id, {}).get(native_aa)
        if prob is None:
            raise KeyError(f"No probability for {res_id} {native_aa}")
        scores[res_id] = round(prob, 3)
    return scores


def annotate_bfactor(pdb_in: str, scores: dict, pdb_out: str):
    """
    Read pdb_in, replace B-factors (cols 61–66) for atoms in scored residues,
    write to pdb_out.
    """
    with open(pdb_in, "r") as fin, open(pdb_out, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")):
                chain_id = line[21]
                resnum = line[22:26].strip()
                ins_code = line[26].strip()
                res_id = f"{chain_id}{resnum}{ins_code}"
                if res_id in scores:
                    new_b = f"{scores[res_id]:6.3f}"
                    line = line[:60] + new_b + line[66:]
            fout.write(line)


def process_single(
    pdb_path: Path,
    out_base: Path,
):
    """Process one PDB: score and annotate."""
    name = pdb_path.stem
    out_folder = out_base / name
    annotated_path = out_base / f"{name}_annotated.pdb"

    print(
        f"Processing {pdb_path.name} -> scores in {out_folder}, annotated -> {annotated_path}"
    )
    run_score(str(pdb_path), str(out_folder))
    pt_file = find_pt_file(str(out_folder))
    sequence, mean_probs_dict, res_names = load_scores(pt_file)
    scores = get_current_res_mean_probs(sequence, mean_probs_dict, res_names)
    annotate_bfactor(str(pdb_path), scores, str(annotated_path))


def main():
    parser = argparse.ArgumentParser(
        description="Run score.py and annotate PDB B-factors on a file or directory"
    )
    parser.add_argument(
        "input",
        help="Input PDB file or directory of PDB files",
    )
    parser.add_argument(
        "out_base",
        help="Base directory for score.py outputs (per-PDB subfolders)",
    )

    args = parser.parse_args()

    inp = Path(args.input)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    pdb_list = []
    if inp.is_dir():
        pdb_list = sorted(inp.glob("*.pdb"))
        if not pdb_list:
            print(f"No .pdb files found in directory {inp}")
            sys.exit(1)
    elif inp.is_file() and inp.suffix.lower() == ".pdb":
        pdb_list = [inp]
    else:
        print(f"Error: {inp} is not a .pdb file or directory containing .pdb files.")
        sys.exit(1)

    for pdb_path in pdb_list:
        try:
            process_single(pdb_path, out_base)
        except Exception as e:
            print(f"Error processing {pdb_path.name}: {e}")

    print("All done!")


if __name__ == "__main__":
    main()
