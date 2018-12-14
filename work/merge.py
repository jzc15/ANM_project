import os
import pandas as pd

rl = os.listdir('.')
k = []
v = []
p = []
for filename in rl:
	if filename.split('_')[-1] == 'result.csv':
		df = pd.read_csv(filename)
		c = df['case'].tolist()
		t = df['timestamp'].tolist()
		s = df['predict'].tolist()

		for i in range(len(c)):
			k.append(c[i])
			v.append(t[i])
			p.append(s[i])
pd.DataFrame({'KPI ID':k, 'timestamp':v, 'predict':p}, columns = ['KPI ID', 'timestamp', 'predict']).to_csv('r.csv', index=False)
