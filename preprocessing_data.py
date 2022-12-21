import pandas as pd

# Load dataset to DataFrame
hek293t = pd.read_csv('hek293t.csv', names=['','sgRNA','DNA','Label'])
k562 = pd.read_csv('K562.csv', names=['','sgRNA','DNA','Label'])
df_hek293t = pd.DataFrame(hek293t)
df_K562 = pd.DataFrame(k562)

# Integer Encoding Dictionary for every base pair of sgRNA and DNA
int_encode_dict = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,"GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}

# Save dataset to list
hek_sgRNA_list = df_hek293t['sgRNA'].values.tolist()
hek_DNA_list = df_hek293t['DNA'].values.tolist()
hek_Label_list = df_hek293t['Label'].values.tolist()

K562_sgRNA_list = df_K562['sgRNA'].values.tolist()
K562_DNA_list = df_K562['DNA'].values.tolist()
K562_Label_list = df_K562['Label'].values.tolist()

# Integer encoding for every base pair of sgRNA and DNA
def int_encode(sgRNA, DNA, label):
    data = []
    for i in range(len(sgRNA)):
        temp = []
        temp.append(sgRNA[i])
        temp.append(DNA[i])
        temp.append(label[i])

        for j in range(len(sgRNA[i])):
            encoded = (int_encode_dict[sgRNA[i][j]+DNA[i][j]]-1)
            temp.append(encoded)
        data.append(temp)
    return data

# Save encoded data to csv
encoded_K562 = int_encode(K562_sgRNA_list,K562_DNA_list,K562_Label_list)
encoded_hek293t = int_encode(hek_sgRNA_list,hek_DNA_list,hek_Label_list)
encoded_all_data =encoded_K562 + encoded_hek293t

encoded_all_data =  pd.DataFrame(encoded_all_data)
encoded_all_data.to_csv('encoded_all_data.csv', header=False)

encoded_K562 =  pd.DataFrame(encoded_K562)
encoded_K562.to_csv('encoded_K562.csv', header=False)

encoded_hek293t =  pd.DataFrame(encoded_hek293t)
encoded_hek293t.to_csv('encoded_hek293t.csv', header=False)


