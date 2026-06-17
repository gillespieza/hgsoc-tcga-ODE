# %%
import pandas as pd
import mygene
from pathlib import Path

# set the root path of the project
ROOT = Path(__file__).resolve().parent.parent

def main():



    # %%

    # Initialize MyGeneInfo - an online gene annotation service
    mg = mygene.MyGeneInfo()

    # The 14 genes used for ODE parameterisation
    ode_genes = [
        'BRCA1', 'BRCA2', 'RAD51', 'PALB2', 'BRIP1',       # Homologous recombination repair
        'ATM', 'ATR',                                      # DNA damage sensing/checkpoint kinases
        'CHEK1', 'CHEK2',                                  # Checkpoint effectors/signalling
        'BCL2', 'BCL2L1', 'BAX', 'BAD',                    # Apoptotic threshold
        'TP53'                                             # Negative control
    ]

    # Query MyGeneInfo for Entrez Gene IDs corresponding to the gene symbols
    # convert gene symbols to Entrez Gene IDs using MyGeneInfo
    results = mg.querymany(
        ode_genes,
        scopes='symbol',
        fields='entrezgene',
        species='human',
        as_dataframe=True,
        verbose=False
    )
    # print(results)

    # %%
    # Create a lookup dictionary from gene symbols to Entrez Gene IDs
    symbol_to_entrez = results['entrezgene'].dropna().astype(str).to_dict()

    #print("Symbol to Entrez mapping:")
    #print(symbol_to_entrez)

    # Create a reverse mapping from Entrez Gene IDs to gene symbols
    entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}

    # Load RNA-seq data
    expr = pd.read_csv(
        ROOT / "data" / "raw" / "expression" / "data_mrna_seq_fpkm.txt",
        sep="\t",
        index_col=0
    )

    # make sure row names are strings for mapping
    expr.index = expr.index.astype(str)

    # Keep only rows matching target Entrez IDs
    target_entrez = set(entrez_to_symbol.keys())
    expr_ode = expr.loc[expr.index.intersection(target_entrez)].copy()

    # Rename rows to gene symbols
    expr_ode.index = expr_ode.index.map(entrez_to_symbol)
    expr_ode.index.name = 'gene_symbol'

    print(f"Kept {len(expr_ode)} of 14 ODE genes")
    print(expr_ode.head())

    # %%
    # Create mapping DataFrame
    gene_id_map = pd.DataFrame(
        list(entrez_to_symbol.items()),
        columns=['entrez_id', 'symbol']
    )

    # Save to data/processed/gene_id_map.csv
    out_path = ROOT / "data" / "processed" / "gene_id_map.csv"
    gene_id_map.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(gene_id_map)

    # %%
    # expr_ode is genes × samples
    # Transpose to samples × genes
    expr_ode_log = expr_ode.T
    expr_ode_log.index.name = 'SAMPLE_ID'

    # Truncate sample IDs to 12-character patient barcodes (e.g. TCGA-04-1331-01A → TCGA-04-1331).
    expr_ode_log.index = expr_ode_log.index.str[:12]
    expr_ode_log.index.name = 'PATIENT_ID'

    print(f"Expression matrix shape: {expr_ode_log.shape}")
    print(expr_ode_log.head())

    # Save expression matrix for downstream merge step
    expr_out = ROOT / "data" / "processed" / "expression" / "ode_genes_fpkm_subset.csv"
    expr_out.parent.mkdir(parents=True, exist_ok=True)

    expr_ode_log.reset_index().to_csv(expr_out, index=False)

    print("Saved:", expr_out)

if __name__ == "__main__":
    main()
# %%
