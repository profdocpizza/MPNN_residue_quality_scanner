#!/usr/bin/env python3
"""
Run single‐residue scoring via score.py, then annotate your PDB’s B‐factors
with the rounded (3-decimal) native–AA probabilities.

Usage:
    python run_score_and_annotate.py \
        /path/to/input.pdb \
        /path/to/output_folder_for_scoring \
        /path/to/annotated_output.pdb

This script:
  1. Calls score.py to compute single‐AA scores (LigandMPNN/ProteinMPNN).
  2. Locates the resulting .pt score file in the output folder.
  3. Loads the .pt, extracts sequence, mean_probs_dict, and res_names.
  4. Computes the probability of the native AA at each residue by looking up
     mean_probs_dict[res_id][native_aa].
  5. Reads the input PDB and replaces each atom’s B-factor (cols 61–66)
     for those residues with the rounded score.
  6. Writes out a new annotated PDB.
"""
import argparse
import subprocess
import sys
import os
import torch

model_type = "ligand_mpnn"  # or "protein_mpnn"


def run_score(pdb_path: str, out_folder: str):
    """Call score.py to compute single‐amino‐acid scores."""
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
        "--homo_oligomer",
        "1",
    ]
    subprocess.run(cmd, check=True)


def find_pt_file(out_folder: str) -> str:
    """Locate the single .pt file in the scoring output folder."""
    files = [f for f in os.listdir(out_folder) if f.endswith(".pt")]
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one .pt in {out_folder}, found: {files}")
    return os.path.join(out_folder, files[0])


def load_scores(pt_path: str) -> dict:
    """
    Load the .pt file and build a dict mapping residue IDs (e.g. 'A23')
    to their rounded native-AA probability.
    """
    data = torch.load(pt_path, map_location="cpu")
    alphabet = data["alphabet"]
    sequence = data["sequence"]
    mean_probs = data["mean_of_probs"]
    res_names = data["residue_names"]

    return alphabet, sequence, mean_probs, res_names


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


def main():
    parser = argparse.ArgumentParser(
        description="Run score.py and annotate PDB B-factors"
    )
    parser.add_argument("pdb", help="Input PDB file to score and annotate")
    parser.add_argument("out_folder", help="Folder with score.py output (.pt)")
    parser.add_argument("annotated_pdb", help="Path for output annotated PDB")
    args = parser.parse_args()

    # 1) Run scoring
    run_score(args.pdb, args.out_folder)

    # 2) Locate and load score data
    pt_file = find_pt_file(args.out_folder)
    sequence, mean_probs_dict, res_names = load_scores(pt_file)

    # 3) Compute per-residue native-AA probabilities
    scores = get_current_res_mean_probs(sequence, mean_probs_dict, res_names)

    # 4) Annotate B-factors and write new PDB
    annotate_bfactor(args.pdb, scores, args.annotated_pdb)

    print(
        f"PDB written to: {args.annotated_pdb} with {model_type} likelihoods written as beta factor for each residue (provided full backbone and sequence). Open in pymol and write: \ncolor tv_blue; color skyblue, b<0.1; color tv_orange, b<0.08; color tv_red, b<0.03 "
    )


if __name__ == "__main__":
    main()
