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
valjc = 0.05
valcn = 0.45
valaa = 0.35
valra = 0.35
valpa = 0.1
valk = 0.0005
valrw = 0.005

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


# This is for test purposes
# This will delete 10% of the edges from the graph while keeping the graph connected
L =list()
while N:
	x = random.sample(list(G), k=2)
	a = x[0]
	b = x[1]
	if (G.degree(a)== 1) or (G.degree(b)== 1):
		continue
	elif x in G.edges():
		if x not in K.edges():  
			G.remove_edge(a,b)
			if nx.is_connected(G):
				N = N - 1 
				K.add_edge(a,b)
				# add one extremity on the deleted edge to the list L
				# In Target exploration the list L contains a subset of nodes that are suspected to be incident to missing links
				n = random.choice([a, b])
                L.append(n)
			else:
				G.add_edge(a,b)

# The number of edges to predict here is the same as the number of edges we deleted for test purposes 
Numb = int(percentage(ratio, f)) + 1


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

# create the adjacency matrix
A = nx.adjacency_matrix(G).todense()
alpha = 0.001
n = A.shape[0]
result = Katz(alpha, A)
norm_katz = Norm_Katz(result)

# Choose a node at random to start the exploratin process
node = random.choice(L)
u = 0
v = 0
# Process with local indexes
while L and Numb:	
	choice = ""
	m = -math.inf
	for noeud in range(n):
		if A[node, noeud] == 0:
			if node == noeud :
				continue
			else:
				val_ra = sorted(nx.resource_allocation_index(G, [(node, noeud)]))
				for x, y, p in val_ra:
					z = (p - ramin) / (ramax - ramin)
					if m < z:
						m = z
						u = node
						v = noeud
						choice = 'RA'
				val_aa = sorted(nx.adamic_adar_index(G, [(node, noeud)]))
				for x, y, p in val_aa:
					z = (p - aamin) / (aamax - aamin)
					if m < z:
						m = z
						u = node
						v = noeud
						choice = 'AA'
				val_pa = sorted(nx.preferential_attachment(G, [(node, noeud)]))
				for x, y, p in val_pa:
					z = (p - pamin) / (pamax - pamin)
					if m < z:
						m = z
						u = node
						v = noeud
						choice = 'PA'
				val_jc = sorted(nx.jaccard_coefficient(G, [(node, noeud)]))
				for x, y, p in val_jc:
					z = (p - jcmin) / (jcmax - jcmin)
					if m < z:
						m = z
						u = node
						v = noeud
						choice = 'JC'
				val_cn = sorted(nx.common_neighbors(G, node, noeud))
				p =  len(val_cn)
				z = (p - cnmin) / (cnmax - cnmin)
				if m < z:
					m = z
					u = node
					v = noeud
					choice = 'CN'
	
	if choice == "RA":
		if valra <= m:
			G.add_edge(u, v)
			A[u, v] = 1
			# decrease the target number of edges to predict
			Numb = Numb - 1
		else:
			choice = ""
			m = -math.inf	
			for noeud in range(n):
				if A[node, noeud] == 0:
					if node == noeud :
						continue
					else:
						z = norm_katz[node, noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "Katz"
						pr = nx.pagerank_numpy(G, alpha=0.5, personalization={node: 1})
						z = pr[noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "rwwr"

			if choic == 'Katz':
				if valk <= m:
					G.add_edge(u, v)
					A[u, v] = 1
					Numb = Numb - 1

			if choice == 'rwwr':		
				if valrw <= m:
					G.add_edge(node, noeud)
					A[u, v] = 1
					Numb = Numb - 1

	elif choice == "AA":
		if valaa <= m:
			G.add_edge(u, v)
			A[u, v] = 1
			# decrease the target number of edges to predict
			Numb = Numb - 1
		else:
			choice = ""
			m = -math.inf	
			for noeud in range(n):
				if A[node, noeud] == 0:
					if node == noeud :
						continue
					else:
						z = norm_katz[node, noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "Katz"
						pr = nx.pagerank_numpy(G, alpha=0.5, personalization={node: 1})
						z = pr[noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "rwwr"
							
			if choic == 'Katz':
				if valk <= m:
					G.add_edge(u, v)
					A[u, v] = 1
					Numb = Numb - 1

			if choice == 'rwwr':		
				if valrw <= m:
					G.add_edge(node, noeud)
					A[u, v] = 1
					Numb = Numb - 1

	elif choice == "PA":

		if valpa <= m:
			G.add_edge(u, v)
			A[u, v] = 1
			# decrease the target number of edges to predict
			Numb = Numb - 1
		else:
			choice = ""
			m = -math.inf	
			for noeud in range(n):
				if A[node, noeud] == 0:
					if node == noeud :
						continue
					else:
						z = norm_katz[node, noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "Katz"
						pr = nx.pagerank_numpy(G, alpha=0.5, personalization={node: 1})
						z = pr[noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "rwwr"
							
			if choic == 'Katz':
				if valk <= m:
					G.add_edge(u, v)
					A[u, v] = 1
					Numb = Numb - 1

			if choice == 'rwwr':		
				if valrw <= m:
					G.add_edge(node, noeud)
					A[u, v] = 1
					Numb = Numb - 1

	elif choice == "JC":
		if valjc <= m:
			G.add_edge(u, v)
			A[u, v] = 1
			# decrease the target number of edges to predict
			Numb = Numb - 1
		else:
			choice = ""
			m = -math.inf	
			for noeud in range(n):
				if A[node, noeud] == 0:
					if node == noeud :
						continue
					else:
						z = norm_katz[node, noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "Katz"
						pr = nx.pagerank_numpy(G, alpha=0.5, personalization={node: 1})
						z = pr[noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "rwwr"
							
			if choic == 'Katz':
				if valk <= m:
					G.add_edge(u, v)
					A[u, v] = 1
					Numb = Numb - 1

			if choice == 'rwwr':		
				if valrw <= m:
					G.add_edge(node, noeud)
					A[u, v] = 1
					Numb = Numb - 1

	elif choice == "CN":
		if valcn <= m:
			G.add_edge(u, v)
			A[u, v] = 1
			# decrease the target number of edges to predict
			Numb = Numb - 1
		else:
			choice = ""
			m = -math.inf	
			for noeud in range(n):
				if A[node, noeud] == 0:
					if node == noeud :
						continue
					else:
						z = norm_katz[node, noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "Katz"
						pr = nx.pagerank_numpy(G, alpha=0.5, personalization={node: 1})
						z = pr[noeud]
						if m < z:
							m = z
							u = node
							v = noeud
							choice = "rwwr"
							
			if choic == 'Katz':
				if valk <= m:
					G.add_edge(u, v)
					A[u, v] = 1
					Numb = Numb - 1

			if choice == 'rwwr':		
				if valrw <= m:
					G.add_edge(node, noeud)
					A[u, v] = 1
					Numb = Numb - 1


	if node in L:
		L.remove(node)
	if len(L) == 0:
		print("we are done")
	else:
		node = random.choice(L)

