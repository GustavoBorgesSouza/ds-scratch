#  [markdown]
# # Linear algebra

#  [markdown]
# ## Vectors
# 
# Concretely (for us), vectors are points in some finite-dimensional space. Although you might not think of your data as vectors, they are often a useful way to represent numeric data.
# 
# The simplest from-scratch approach is to represent vectors as lists of numbers.

# 
from typing import List

Vector = List[float]

height_weight_age: Vector = [180,70,25]

#  [markdown]
# ## Vectors arithmetics

#  [markdown]
# ### Vector sum/add
# 
# Vectors add componentwise, same size vectors v and w is:
# 
# vector_sum = [ v[0] + w[0], v[1] + w[1], ... , v[n]  + w[n] ]  

# 
def add(v: Vector, w: Vector) -> Vector:
    """ Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be same lenght"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1,2,3], [5,8,2]) == [6,10,5]

#  [markdown]
# ### Vector subtracting
# 
# simiilar to vector sum

# 
def subtract(v: Vector, w: Vector) -> Vector:
    """ Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be same lenght"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([9,8,7], [5,8,2]) == [4,0,5]

#  [markdown]
# ### Componentwise sum list of vectors

# 
def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements in the vector list"""
    assert vectors, "no vectors provided"
    
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different vector sizes"
    
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

#  [markdown]
# ### Multiply by scalar

# 
def scalar_multiply(s: float, v: Vector) -> Vector:
    """ Multiplies every elements by a scalar"""
    return [s * v_i for v_i in v]

assert scalar_multiply(2, [1,2,3]) == [2,4,6]

#  [markdown]
# ### Componentwise mean of vectors

# 
def vector_mean(vectors: List[Vector]) -> Vector:
    """Gets mean corresponding elements in the vector list"""
    assert vectors, "no vectors provided"
    
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

#  [markdown]
# ### Dot product
# 
# The dot product of two vectors is the sum of
# their componentwise products:

# 
def dot(v: Vector, w: Vector) -> float:
    """ Computes v_1 * w_1 + ... +  v_n * w_n"""
    assert len(v) == len(w), "vectors must be same lenght"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32 # 1 * 4 + 2 * 5 + 3 * 6

#  [markdown]
# ### Vector's sum of squares
# with dot product this gets way easier

# 
def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

#  [markdown]
# ### Vectors magnitude/lenght

# 
import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or lenght) of v"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3,4]) == 5

#  [markdown]
# ### Compute distance beetween two vectors

# 
def squared_distance(v: Vector,w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

# 
def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))
    # or
    return magnitude(subtract(v, w))

#  [markdown]
# ## Matrices
# 
# A matrix is a two-dimensional collection of numbers. We will represent matrices as
# lists of lists, with each inner list having the same size and representing a row of the
# matrix. If A is a matrix, then A[i][j] is the element in the ith row and the jth column.
# Per mathematical convention, we will frequently use capital letters to represent matri‐
# ces

# 
Matrix = List[List[float]]

#  [markdown]
# ### Get matrix shape

# 
from typing import Tuple

def shape(A: Matrix) -> Tuple:
    """ Returns the number of rows in A, and number of columns in A. (rows, columns) """
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3) # 2 rows, 3 columns

#  [markdown]
# ### Get rows and columns
# 
# think of each row of an n × k matrix as a vector of length k, and
# each column as a vector of length n:

# 
def get_rows(A:Matrix, i:int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]

# 
def get_columns(A: Matrix, j:int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j] for A_i in A]

#  [markdown]
# ### Creating matrices
# 
# crete matrices based on shape and function for generating elements

# 
from typing import Callable

def make_matix(num_rows:int, num_cols:int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """Returns a num_rows x num_cols matrix whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

# 
def identity_matrix(n:int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matix(n,n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == \
[[1, 0, 0, 0, 0],
[0, 1, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 1, 0],
[0, 0, 0, 0, 1]]


