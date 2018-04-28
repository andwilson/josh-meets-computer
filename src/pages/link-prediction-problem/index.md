---
title: "Social Network Prediction Problem"
date: "2018-04-17"
path: "/link-prediction-problem/"
category: "Notes"
section: "Network Analysis"
---


Given a fixed network, can we predict how the network is going to look in the future? This certain problem has a lot of applications in Social Networks; given a group of friends, can you build an effective recommender system to suggest new connections?

## Network Measurements

### Common Neighbors
common_neighbors = N1<sup>neighbors</sup> &#8746; N2<sup>neighbors</sup>


```python
import networkx as nx

def common_neighbors(G, n1, n2):
    return len(set(G.neighbors(n1)) & set(G.neighbors(n2)))
    
G = nx.karate_club_graph()

# get number of common neighbors for all nodes that are not connected
c = [(e[0], e[1], common_neighbors(G, e[0], e[1])) for e in nx.non_edges(G)]

pd.DataFrame(c, columns=['node1', 'node2', 'Common Neighbors']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node2</th>
      <th>Common Neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Jaccard Coefficient
Similar to Common neighbors; denotes the number of common neighbors normalized by total number of neighbors. A Jaccard Coefficient of 1 indicates that two unconnected nodes share all the same neighbors. An example of this would be a triadic closure -- node A and B are both only connected to C, but not connected themselves. They both share C, and the total number of neighbors is just C. This would indicate a high probability of the two nodes connecting.

---


```python
import pandas as pd
def total_neighbors(G, n1, n2):
    return len(set(G.neighbors(n1)) | set(G.neighbors(n2)))

def jaccard_coefficient(G):
    for n in nx.non_edges(G):
        jaccard = common_neighbors(G, n[0], n[1]) / total_neighbors(G, n[0], n[1])
        yield (n[0], n[1], jaccard)

pd.DataFrame([i for i in jaccard_coefficient(G)], columns=['node1', 'node2', 'Jaccard Coeff.']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node2</th>
      <th>Jaccard Coeff.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>0.120000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>0.137931</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



###  Resource Allocation Index

describes the fraction of a 'resource' that one node can send to another through their respective neighbors. It is calculated by looked at each of the common neighbor nodes, and summing the inverse of degree for each of these nodes. The intution is that if a lot of the common neighbor nodes have low degrees, then there is a high likelihood that information being sent from the first node to the second (through the common neighbors) will reach the second node. If their common neighbors have a ton of other connections (higher degree), this will result in a lower Resource Allocation Index. At the same time, since the index sums over all common neighbors, two nodes with a lot of common neighbors will have more opportunities to send information to each other, denoting a higher Resource Allocation Index as well.

---


```python
import matplotlib.pyplot as plt

def resource_allocation_index(G):
    for n in nx.non_edges(G):
        common_neighbors = set(G.neighbors(n[0])) & set(G.neighbors(n[1]))
        index = 0
        
        for c_n in common_neighbors:
            index += 1 / len(G.neighbors(c_n))
        yield (n[0], n[1], index)
        
df = pd.DataFrame([i for i in resource_allocation_index(G)], columns=['Node 1', 'Node 2', 'RAI'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Node 1</th>
      <th>Node 2</th>
      <th>RAI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Adamic-Adar Index
Similar to Resource Allocation Index, but divides by the log of the degree for each common neighbor instead of just the degree

---


```python
import math

def adamic_adar_index(G):
    for n in nx.non_edges(G):
        common_neighbors = set(G.neighbors(n[0])) & set(G.neighbors(n[1]))
        index = 0
        for a_n in common_neighbors:
            index += 1 / math.log(len(G.neighbors(a_n)))
        yield (n[0], n[1], index)
        
df = pd.DataFrame([i for i in adamic_adar_index(G)], columns=['Node 1', 'Node 2', 'AAI'])
df.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Node 1</th>
      <th>Node 2</th>
      <th>AAI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>1.613740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>2.711020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>0.434294</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Preferential Attachment Score
Intuition is that if two nodes have a very high degree, it's very likely that they'll be connected in the future. Two popular nodes (they are connected to a lot of other nodes) stand a very good chance at becoming connected (aware) of each other.

---


```python
def preferential_attachment_score(G):
    for n in nx.non_edges(G):
        yield (n[0], n[1], len(G.neighbors(n[0])) * len(G.neighbors(n[1])))

df = pd.DataFrame([i for i in preferential_attachment_score(G)], columns=['Node 1', 'Node 2', 'PAS'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Node 1</th>
      <th>Node 2</th>
      <th>PAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>272</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>



## Community Structure of Network
Pairs of nodes who belong to the same community and have many common neighbors have a high likelihood of forming an edge. This measures the number of common neighbors between two nodes, but gives a bonus to this measurement if the two nodes belong to the same community.


### Common Neighbor Soundarajan-Hopcroft Score
This score indicates the count of common neighbors between two nodes, summed together with the count of common neighbors that belong to the same community as the two nodes.

---


```python
nx.set_node_attributes(G, 'community', 1)

def community(G, n):
    com = nx.get_node_attributes(G, 'community')
    return com[n]

def soundarajan_hopcroft_score(G):
    for n in nx.non_edges(G):
        common_neighbors = set(G.neighbors(n[0])) & set(G.neighbors(n[1]))
        score = 0
        for c_n in common_neighbors:
            score += 1 + (lambda n, c: 1 if community(G, n[0]) == community(G, n[1]) == community(G, c) else 0)(n, c_n)
            
        yield (n[0], n[1], score)
        
df = pd.DataFrame([i for i in soundarajan_hopcroft_score(G)], columns=['Node 1', 'Node 2', 'SHC'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Node 1</th>
      <th>Node 2</th>
      <th>SHC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Resource Allocation Soundarajan-Hopcroft Score
Similar to Resource Aloocation Score, counting the number of degrees each common neighbor has between two nodes. However there is an additional filter criteria where it only counts these instances when the common neighbor is in the same community as the two nodes.


```python
def community(G, n):
    com = nx.get_node_attributes(G, 'community')
    return com[n]

def community_degrees(G, n, c):
    """
    A function to calculate the individual Resource Allocation Soundarajan Hopcroft
    Score for each common neighbor node.
    args:
        G (graph): the graph
        n (tuple): two non connected nodes
        c (int): a single node that is a connected neighbor of the two non connected nodes
    
    returns:
        score: the resource allocation score of this common node
    """
    
    if community(G, c) == community(G, n[0]) == community(G, n[1]):
        return 1 / len(G.neighbors(c))
    else:
        return 0
    
def resource_allocation_soundarajan_hopcroft(G):
    for n in nx.non_edges(G):
        common_neighbors = set(G.neighbors(n[0])) & set(G.neighbors(n[1]))
        score = 0
        for c_n in common_neighbors:
            score += community_degrees(G, n, c_n)
        yield (n[0], n[1], score)

df = pd.DataFrame([i for i in resource_allocation_soundarajan_hopcroft(G)], columns=['Node 1', 'Node 2', 'RA-SHS'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Node 1</th>
      <th>Node 2</th>
      <th>RA-SHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>32</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>33</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>9</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>15</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


