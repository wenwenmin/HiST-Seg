def find_common_genes(slice_names):
    """
    获取多个切片文件中共同涉及的基因

    参数:
        slice_names: 切片名称列表，如 ['v03','v04','v05','v06','v07','v08','v09','v10','IDC']

    返回:
        所有切片文件中共有的基因列表
    """
    # 初始化第一个文件的所有基因集合
    first_file = f"tmp_gene_data/{slice_names[0]}_genes.txt"
    with open(first_file, 'r') as f:
        common_genes = {line.strip() for line in f}

    # 与后续文件的基因取交集
    for slice_name in slice_names[1:]:
        file_name = f"tmp_gene_data/{slice_name}_genes.txt"
        with open(file_name, 'r') as f:
            file_genes = {line.strip() for line in f}
            common_genes &= file_genes  # 取交集

    return sorted(common_genes)  # 返回排序后的列表


if __name__ == "__main__":
    # 切片名称列表
    slice_names = ['v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'v10', 'IDC']

    # 获取共同基因
    common_genes = find_common_genes(slice_names)

    # 打印结果
    print(f"所有切片共同涉及的基因数量: {len(common_genes)}")
    print("共同基因列表:")
    for gene in common_genes:
        print(gene)

    # 将结果保存到文件
    with open("common_genes.txt", 'w') as f:
        f.write("\n".join(common_genes))
    print("结果已保存到 common_genes.txt")