# Code for searching BioMart

from pybiomart import Dataset

# Connect to the Ensembl BioMart server and select the human gene dataset
dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')

# gene_list = ['EGFR', 'BRAF', 'ERBB2', 'MYC', 'RHOA', 'CTNNB1', 'PTEN', 'PIK3CA', 'KRAS', 'APC', 'TP53', 'EML4', 'ALK']  # add your gene names as strings here
gene_list = ['ENSG00000201098',
             'ENSG00000216184',
             'ENSG00000221326',
             'ENSG00000252316']
gene_list = sorted(gene_list)

# Query Ensembl gene IDs for a list of gene symbols
results = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])

filtered_results = results[results['Gene stable ID'].isin(gene_list)].sort_values(by='Gene stable ID')
ensemble_ids = list(filtered_results['Gene name'])

print(gene_list)
print(filtered_results)
print(ensemble_ids)