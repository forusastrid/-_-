data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url , sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print("데이터의 형태 :", data.shape)
