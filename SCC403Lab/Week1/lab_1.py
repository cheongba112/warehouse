# encoding=utf-8

import math

'''
import numpy as np

# 1
a_a = np.array([1, 2, 3, 4])
a_b = np.array([1, 2, 3, 4])
lumbda = 3
print(a_a + a_b)
print(lumbda * a_a)
# cannot find the dot product function for numpy array

# 2
m_a = np.matrix('1, 2, 3;' + 
                '4, 5, 6;' +
                '7, 8, 9')
m_b = np.matrix([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(m_a * m_b)
print(np.dot(m_a, m_b))
print(m_a.dot(m_b))

# 3
print(np.transpose(m_a))

# 4
# m_a is a singular matrix hence it has no inverse matrix
m_c = np.matrix([[1, 2], [3, 4]])
print(np.linalg.inv(m_c))

# 5

# 6

'''

# 1
def Sum(a, b):
    if len(a) != len(b):
        print('Error')
        return
    else:
        rlist = []
        for i in range(len(a)):
            rlist.append(a[i] + b[i])
        return rlist

def Mult(a, lumbda):
    return [lumbda * x for x in a]

def DotProduct(a, b):
    if len(a) != len(b):
        print('Error')
        return
    else:
        r = 0
        for i in range(len(a)):
            r += a[i] * b[i]
        return r

# 2
def mult(a, b):
    if (not a) or (not b) or (len(a[0]) != len(b)):
        print('Error')
        return
    else:
        rmat = []
        for row in range(len(a)):
            rrow = []
            for col in range(len(b[0])):
                r = 0
                for i in range(len(a[0])):
                    r += a[row][i] * b[i][col]
                rrow.append(r)
            rmat.append(rrow)
        return rmat

# 3
# ab^T means a . (b^T)?
def transpose(a):
    if not a:
        return []
    else:
        rmat = []
        for i in range(len(a[0])):
            rrow = []
            for j in range(len(a)):
                rrow.append(a[j][i])
            rmat.append(rrow)
        return rmat

# 4
def isInverse(a, b):
    if (not a) or (not b):
        return False
    elif (len(a) != len(a[0])) or (len(b) != len(b[0])):
        return False
    elif len(a) != len(b):
        return False
    else:
        rmat = mult(a, b)
        for i in range(len(rmat)):
            for j in range(len(rmat)):
                if (i == j) and (rmat[i][j] != 1):
                    return False
                elif (i != j) and (rmat[i][j] != 0):
                    return False
        return True

# 5
def dist(a, b):
    if (not a) or (not b) or (len(a) != len(b)):
        print('Error')
        return
    else:
        r = 0
        for i in range(len(a)):
            r += (a[i] - b[i]) ** 2
        return math.sqrt(r)

def lowDist(a):
    rowa = rowb = 0
    rowdist = 1000000.0
    for i in range(len(a)):
        for j in range(i, len(a)):
            if i != j:
            	s = dist(a[i],a[j])
            	if s < rowdist:
                    rowa = i
                    rowb = j
                    rowdist = s
    return rowa, rowb

# 6
def cosSimilarity(a, b):
	if (not a) or (not b):
		return 0
	elif len(a) != len(b):
		print('Error')
		return
	else:
		m1 = m2 = d1 = 0
		for i in range(len(a)):
			m1 += a[i] ** 2
			m2 += b[i] ** 2
			d1 += a[i] * b[i]
		return d1 / (math.sqrt(m1) * math.sqrt(m2))


a = [1, 2, 3, 4]
b = [4, 3, 2, 1]
lumbda = 2

print(Sum(a, b))
print(Mult(a, lumbda))
print(DotProduct(a, b))

print(dist(a, b))
print(cosSimilarity(a, b))

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [[1, 2], [3, 4], [5, 6]]
ina = [[1, 2],
       [3, 4]]
inb = [[-2,  1],
       [1.5, -0.5]]

print(mult(a, b))
print(transpose(a))
print(isInverse(ina, inb))
print(lowDist(a))

'''
Answer Check:
Exercises 1
zip() function : 'for elemA, elemB in zip(a, b):'

Exercises 2
for j in range(len(B))? not len(B[0])?

Exercises 3
To be noticed, 'a' and 'b' are vectors, so ab^T equals to a . b(dot product).
'''
