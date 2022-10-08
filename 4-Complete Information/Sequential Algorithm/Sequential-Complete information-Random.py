import networkx as nx
import csv
import numpy as np
from numpy.linalg import inv
import pandas as pd
import random
import math
import copy

def RA(G):
	outfile = open("ra.csv", "w", newline='')
	field_names = ['x', 'y', 'value']
	csv_writer = csv.DictWriter(outfile, fieldnames=field_names)
	csv_writer.writeheader()

	for e in G.edges():
		i = e[0]
		j = e[1]
		preds = nx.resource_allocation_index(G, [(i, j)])
		for u, v, p in preds:
			'(%d, %d) -> %d' % (u, v, p)
			z = {'x': u, 'y': v, 'value': p}
			csv_writer.writerow(z)

	outfile.close()


def CN(G):
	outfile = open("cn.csv", "w", newline='')
	field_names = ['x', 'y', 'value']
	csv_writer = csv.DictWriter(outfile, fieldnames=field_names)
	csv_writer.writeheader()

	for e in G.edges():
		i = e[0]
		j = e[1]
		preds = sorted(nx.common_neighbors(G, node, noeud))
		z = len(preds)
		csv_writer.writerow(z)

	outfile.close()


def JC(G):
	outfile = open("jc.csv", "w", newline='')
	field_names = ['x', 'y', 'value']
	csv_writer = csv.DictWriter(outfile, fieldnames=field_names)
	csv_writer.writeheader()

	for e in G.edges():
		i = e[0]
		j = e[1]
		preds = nx.resource_allocation_index(G, [(i, j)])
		for u, v, p in preds:
			'(%d, %d) -> %d' % (u, v, p)
			z = {'x': u, 'y': v, 'value': p}
			csv_writer.writerow(z)

	outfile.close()


def AA(G):
	outfile = open("aa.csv", "w", newline='')
	field_names = ['x', 'y', 'value']
	csv_writer = csv.DictWriter(outfile, fieldnames=field_names)
	csv_writer.writeheader()

	for e in G.edges():
		i = e[0]
		j = e[1]
		preds = sorted(nx.adamic_adar_index(G, [(i, j)]))
		for u, v, p in preds:
			'(%d, %d) -> %d' % (u, v, p)
			z = {'x': u, 'y': v, 'value': p}
			csv_writer.writerow(z)

def PA(G):
	outfile = open("pa.csv", "w", newline='')
	field_names = ['x', 'y', 'value']
	csv_writer = csv.DictWriter(outfile, fieldnames=field_names)
	csv_writer.writeheader()

	for e in G.edges():
		i = e[0]
		j = e[1]
		preds = sorted(nx.preferential_attachment(G, [(i, j)]))
		for u, v, p in preds:
			'(%d, %d) -> %d' % (u, v, p)
			z = {'x': u, 'y': v, 'value': p}
			csv_writer.writerow(z)


def Katz(alpha, A):
	I = np.eye(n, n)
	result = inv(I - (alpha * A)) - I
	return result
	outfile.close()


def Norm_Katz(result):
	maxi = np.max(result)
	mini = np.min(result)
	norm = result
	for i in range(n):
		for j in range(n):
			norm[i, j] = ((result[i, j]) - mini) / (maxi - mini)
	return norm

# Enter the network you want to test
G = nx.read_edgelist('soc-karate.txt', nodetype=int)
# set the threshold for the network
valjc = 0.1
valcn = 0.1
valaa = 0.1
valra = 0.1
valpa = 0.05
valk = 0.0005
valrw = 0.00075

K = nx.Graph(nodetype=int)
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
L = list()
f = G.size()

def percentage(percent, whole):
	return (percent * whole) / 100.0

# define the ratio of the links you want to delete from the graph in order to test the algorithms
# 
ratio = 10
N = int(percentage(ratio, f)) + 1
N1 = N

# This is for test purposes
# This will delete 10% of the edges from the graph while keeping the graph connected
while N1:
	x = random.sample(list(G), k=2)
	a = x[0]
	b = x[1]
	if (G.degree(a)== 1) or (G.degree(b)== 1):
		continue
	elif x in G.edges():
		if x not in K.edges():  
			G.remove_edge(a,b)
			if nx.is_connected(G):
				N1 = N1 - 1 
				K.add_edge(a,b)
				L.append(x)
			else:
				G.add_edge(a,b)

# we also select 10% of non-existent edges to add to the test set
while N:
    x = random.sample(list(G), k=2)
    if x not in G.edges():
        L.append(x)
        N = N - 1


# This portion of the code if to calculate the Min and Max values from the graph in order to Normalize the values of the indexes 

RA(G)
AA(G)
PA(G)
JC(G)
CN(G)

ra_data = pd.read_csv('ra.csv')
aa_data = pd.read_csv('aa.csv')
pa_data = pd.read_csv('pa.csv')
cn_data = pd.read_csv('cn.csv')
jc_data = pd.read_csv('jc.csv')

value_pa = pa_data[['value']]
pamax = float(value_pa.max())
pamin = float(value_pa.min())
value_ra = ra_data[['value']]
ramax = float(value_ra.max())
ramin = float(value_ra.min())
value_aa = aa_data[['value']]
aamax = float(value_aa.max())
aamin = float(value_aa.min())
value_cn = cn_data[['value']]
cnmax = float(value_cn.max())
cnmin = float(value_cn.min())
value_jc = jc_data[['value']]
jcmax = float(value_jc.max())
jcmin = float(value_jc.min())

l_mesure = ["JC", "AA", "CN", "RA","PA"]

T = list()
for line in L:
	node = line[0]
	noeud = line[1]
	choice = random.sample(l_mesure)

	if choice == "RA":
		val_ra = sorted(nx.resource_allocation_index(G, [(node, noeud)]))
		for x, y, p in val_ra:
			m = (p - ramin) / (ramax - ramin)

		if valra <= m:
			G.add_edge(node, noeud)
		else:
			T.append(line)

	elif choice == "AA":
		val_aa = sorted(nx.adamic_adar_index(G, [(node, noeud)]))
		for x, y, p in val_aa:
			m = (p - aamin) / (aamax - aamin)

		if valaa <= m:
			G.add_edge(node, noeud)
		else:
			T.append(line)

	elif choice == "PA":
		val_pa = sorted(nx.preferential_attachment(G, [(node, noeud)]))
		for x, y, p in val_pa:
			m = (p - pamin) / (pamax - pamin)

		if valpa <= m:
			G.add_edge(node, noeud)
		else:
			T.append(line)

	elif choice == "JC":
		val_jc = sorted(nx.jaccard_coefficient(G, [(node, noeud)]))
		for x, y, p in val_jc:
			m = (p - jcmin) / (jcmax - jcmin)

		if valjc <= m:
			G.add_edge(node, noeud)
		else:
			T.append(line)

	elif choice == "CN":
		val_cn = sorted(nx.common_neighbors(G, node, noeud))
		p =  len(val_cn):
		m = (p - cnmin) / (cnmax - cnmin)

		if valcn <= m:
			G.add_edge(node, noeud)
		else:
			T.append(line)

T1 = tuple(T)
alpha = 0.001
A = nx.adjacency_matrix(G).todense()
n = A.shape[0]
result = Katz(alpha, A)
norm_katz = Norm_Katz(result)
g_mesure = ["Katz", "rwwr"]

for line in T1:
	choic = random.choice(g_mesure)	
	if choic == 'Katz':
		m = norm_katz[node, noeud]
		if valk <= m:
			G.add_edge(node, noeud)

	if choice == 'rwwr':
		pr = nx.pagerank_numpy(G, alpha=0.5, personalization={node: 1})
		m = pr[noeud]
		if valrw <= m:
			G.add_edge(node, noeud)
