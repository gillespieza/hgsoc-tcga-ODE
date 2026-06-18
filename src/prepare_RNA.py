import pandas as pd
import numpy as np
import mygene
from pathlib import Path

# set the root path of the project
ROOT = Path(__file__).resolve().parent.parent

def main():

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

    # Query MyGeneInfo for Entrez Gene IDs corresponding to the gene symbols.
    results = mg.querymany(
        ode_genes,
        scopes='symbol',
        fields='entrezgene',
        species='human',
        as_dataframe=True,
        verbose=False
    )

    # Create a lookup dictionary from gene symbols -> Entrez Gene IDs, dropping genes that did not resolve.
    symbol_to_entrez = results['entrezgene'].dropna().astype(str).to_dict()

    # Create a reverse mapping: Entrez ID -> gene symbol.
    entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}

    # Load the raw RNA-seq matrix.
    # Rows are genes, columns are samples.
    expr = pd.read_csv(
        ROOT / "data" / "raw" / "expression" / "data_mrna_seq_fpkm.txt",
        sep="\t",
        index_col=0
    )

    # Convert row labels to strings so they match Entrez IDs stored as strings.
    expr.index = expr.index.astype(str)

    # Keep only genes that match our target Entrez IDs.
    target_entrez = set(entrez_to_symbol.keys())
    expr_ode = expr.loc[expr.index.intersection(target_entrez)].copy()

    # Rename gene rows from Entrez IDs to human readable gene symbols.
    expr_ode.index = expr_ode.index.map(entrez_to_symbol)
    expr_ode.index.name = 'gene_symbol'

    # Apply log2(FPKM + 1) transformation.
    # This reduces skew and makes expression values easier to use in modeling.
    # Adding + 1 to avoid log(0)
    expr_ode = np.log2(expr_ode + 1)

    print(f"\nKept {len(expr_ode)} of 14 ODE genes")

    ## Save the gene ID mapping for traceability - to a DataFrame and a CSV file.
    gene_id_map = pd.DataFrame(
        list(entrez_to_symbol.items()),
        columns=['entrez_id', 'symbol']
    )
    out_path = ROOT / "data" / "processed" / "gene_id_map.csv"
    gene_id_map.to_csv(out_path, index=False)

    # Transpose so rows become samples and columns become genes.
    expr_ode_log = expr_ode.T
    expr_ode_log.index.name = 'SAMPLE_ID'

    # Convert TCGA sample barcodes to 12-character patient IDs.
    # Example: TCGA-04-1331-01A -> TCGA-04-1331
    expr_ode_log.index = expr_ode_log.index.str[:12]
    expr_ode_log.index.name = 'PATIENT_ID'

    # Remove patients with more than 20% missing values across the gene panel.
    missing_frac = expr_ode_log.isna().mean(axis=1)
    expr_ode_log = expr_ode_log.loc[missing_frac <= 0.20].copy()

    # Keep one row per patient.
    # If duplicate patient IDs exist, keep the row with the fewest missing values.
    expr_ode_log = expr_ode_log.assign(_missing=expr_ode_log.isna().sum(axis=1))
    expr_ode_log = expr_ode_log.sort_values("_missing")
    expr_ode_log = expr_ode_log[~expr_ode_log.index.duplicated(keep="first")]
    expr_ode_log = expr_ode_log.drop(columns="_missing")

    print(f"\nUnique patients after filtering: {expr_ode_log.shape[0]}")
    print(f"\nExpression matrix shape: {expr_ode_log.shape}\n")
    print(expr_ode_log.head())

    # Save the cleaned matrix for the merge step.
    expr_out = ROOT / "data" / "processed" / "rna_clean.csv"
    expr_out.parent.mkdir(parents=True, exist_ok=True)
    expr_ode_log.reset_index().to_csv(expr_out, index=False)

if __name__ == "__main__":
    main()