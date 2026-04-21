from DataProcess import load_coverage

train_val, test = load_coverage()

# Access a specific sample
print(train_val['SRX1096548'])        # {chrom: np.array}
print(train_val['SRX1096548']['Chr1']) # np.array of coverage values

# Access test data
print(test['SRX1096548']['Chr5'])