import pandas as pd
import numpy as np
import mygene
from pathlib import Path
import logging

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

def main():
    """
    Prepare RNA-seq expression matrix for ODE modelling.

    Steps:
    1. Map ODE gene symbols to Entrez IDs
    2. Subset RNA-seq expression matrix
    3. Log-transform expression values
    4. Align TCGA samples to patient IDs
    5. Clean missing values and duplicates
    6. Export processed matrix for downstream modelling
    """

    # -----------------------------------------------------------------
    # Gene annotation service
    # -----------------------------------------------------------------

    # Initialize MyGeneInfo - an online gene annotation service
    mg = mygene.MyGeneInfo()

    # Genes used in the ODE model.
    # TP53 is included as a negative control and should not be used later
    # for patient-specific parameterization.
    ode_genes = [
        'BRCA1', 'BRCA2', 'RAD51', 'PALB2', 'BRIP1',       # Homologous recombination (HR) repair
        'ATM', 'ATR',                                      # DNA damage sensing/checkpoint kinases
        'CHEK1', 'CHEK2',                                  # Checkpoint effectors/signalling
        'BCL2', 'BCL2L1', 'BAX', 'BAD',                    # Apoptotic threshold
        'TP53'                                             # Negative control
    ]

    logger.info(f"Querying MyGeneInfo for {len(ode_genes)} genes")

    # -----------------------------------------------------------------
    # Gene ID mapping
    # -----------------------------------------------------------------

    # Query MyGeneInfo for Entrez Gene IDs corresponding to the gene symbols.
    results = mg.querymany(
        ode_genes,
        scopes='symbol',
        fields='entrezgene',
        species='human',
        as_dataframe=True,
        verbose=False
    )

    # Drop unresolved genes explicitly
    valid_results = results.dropna(subset=["entrezgene"])

    # Create a lookup dictionary from gene symbols -> Entrez Gene IDs
    symbol_to_entrez = (
        valid_results["entrezgene"]
        .astype(int)
        .astype(str)
        .to_dict()
    )

    # Create a reverse mapping: Entrez ID -> gene symbol.
    entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}

    missing_genes = set(ode_genes) - set(valid_results.index)

    if missing_genes:
        logger.warning(f"Unmapped genes: {missing_genes}")

    logger.info(f"Mapped genes: {len(symbol_to_entrez)}/{len(ode_genes)}")

    # -----------------------------------------------------------------
    # Load RNA-seq data
    # -----------------------------------------------------------------
    # Rows are genes, columns are samples.
    expr = pd.read_csv(
        ROOT / "data" / "raw" / "expression" / "data_mrna_seq_fpkm.txt",
        sep="\t",
        index_col=0
    )

    # Convert row labels to strings so they match Entrez IDs stored as strings.
    expr.index = expr.index.astype(str)

    # -----------------------------------------------------------------
    # Subset to ODE genes
    # -----------------------------------------------------------------

    # Keep only genes that match our target Entrez IDs.
    target_entrez = set(entrez_to_symbol.keys())

    matched_genes = expr.index.intersection(target_entrez)

    if len(matched_genes) == 0:
        raise ValueError(
            "No genes matched between expression matrix and ODE gene set. "
            "Check whether RNA-seq IDs are Entrez IDs."
        )

    expr_ode = expr.loc[matched_genes].copy()

    # Rename gene rows from Entrez IDs to human readable gene symbols.
    expr_ode.index = expr_ode.index.map(entrez_to_symbol)
    expr_ode.index.name = "gene_symbol"

    # -----------------------------------------------------------------
    # Log transform
    # -----------------------------------------------------------------

    # Apply log2(FPKM + 1) transformation.
    # This reduces skew and makes expression values easier to use in modeling.
    # Adding + 1 to avoid log(0)
    expr_ode = np.log2(expr_ode + 1)

    logger.info(f"Kept {expr_ode.shape[0]} of 14 ODE genes")

    # -----------------------------------------------------------------
    # Save gene mapping (traceability)
    # -----------------------------------------------------------------

    gene_id_map = pd.DataFrame(
        list(entrez_to_symbol.items()),
        columns=["entrez_id", "symbol"]
    )

    gene_id_map.to_csv(
        ROOT / "data" / "processed" / "gene_id_map.csv",
        index=False
    )

    # -----------------------------------------------------------------
    # Sample alignment
    # -----------------------------------------------------------------

    expr_ode = expr_ode.T
    expr_ode.index.name = "SAMPLE_ID"

    # TCGA barcode truncation (assumption: standard TCGA format)
    expr_ode.index = expr_ode.index.str[:12]
    expr_ode.index.name = "PATIENT_ID"

    # -----------------------------------------------------------------
    # Missing data handling
    # -----------------------------------------------------------------

    missing_frac = expr_ode.isna().mean(axis=1)
    before_filter = len(expr_ode)

    expr_ode = expr_ode.loc[missing_frac <= 0.20].copy()

    logger.info(
        f"Removed {before_filter - len(expr_ode)} patients "
        f"due to missing expression values"
    )

    # -----------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------

    expr_ode["_missing"] = expr_ode.isna().sum(axis=1)

    expr_ode = (
        expr_ode.sort_values("_missing")
        .loc[~expr_ode.index.duplicated(keep="first")]
        .drop(columns="_missing")
    )

    logger.info(
        f"Final RNA cohort: {expr_ode.shape[0]} patients, {expr_ode.shape[1]} genes"
    )

    # -----------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "rna_clean.csv"

    expr_ode.reset_index().to_csv(out_path, index=False)

    logger.info(f"Saved processed RNA matrix to {out_path}")


    # -----------------------------------------------------------------
    # Final cohort comparison debug (IMPORTANT)
    # -----------------------------------------------------------------

    clinical_path = ROOT / "data" / "processed" / "clinical_clean.csv"
    clinical = pd.read_csv(clinical_path)

    clinical_ids = set(clinical["PATIENT_ID"])
    rna_ids = set(expr_ode.index)

    logger.info(
        "\nCohort overlap debug:\n"
        f"  Clinical patients : {len(clinical_ids)}\n"
        f"  RNA patients      : {len(rna_ids)}\n"
        f"  Overlap           : {len(clinical_ids & rna_ids)}\n"
        f"  Only clinical     : {len(clinical_ids - rna_ids)}\n"
        f"  Only RNA          : {len(rna_ids - clinical_ids)}\n"
    )

if __name__ == "__main__":
    main()