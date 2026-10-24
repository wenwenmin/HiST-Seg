def find_common_genes(slice_names):
    'Implementation of find_common_genes for this baseline experiment.'
    
    first_file = f"tmp_gene_data/{slice_names[0]}_genes.txt"
    with open(first_file, 'r') as f:
        common_genes = {line.strip() for line in f}

    
    for slice_name in slice_names[1:]:
        file_name = f"tmp_gene_data/{slice_name}_genes.txt"
        with open(file_name, 'r') as f:
            file_genes = {line.strip() for line in f}
            common_genes &= file_genes  

    return sorted(common_genes)  


if __name__ == "__main__":
    
    slice_names = ['v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'v10', 'IDC']

    
    common_genes = find_common_genes(slice_names)

    
    print(f"Number of genes common to all slices:{len(common_genes)}")
    print('Common gene list:')
    for gene in common_genes:
        print(gene)

    
    with open("common_genes.txt", 'w') as f:
        f.write("\n".join(common_genes))
    print('Results_saved_to_common_genes.txt')