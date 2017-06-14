
from sets import Set

filename = 'train.tsv'
relations = Set()

with open(filename, 'r') as f:
  for line in f.readlines():
    relation = line.rstrip('\n').split('\t')[-1]
    relations.add(relation)

print relations

