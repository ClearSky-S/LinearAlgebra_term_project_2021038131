import random

import numpy as np
import matplotlib.pyplot as plt

def generate_vector100():
    v = []

    # 처음 10개의 벡터는 2개씩 총 5쌍의 관계를 가진다.
    for i in range(0, 5):
        a = random.randrange(-100 // (i + 1), 101 // (i + 1))
        v.append(a)
        v.append(a * (i + 1))
    for i in range(0, 45):
        a = random.randrange(-20, 21)
        v.append(a)
    for i in range(0, 45):
        a = random.randrange(-5, 6)
        v.append(a)
    return v


def generate_matrix(n):
    a = np.array(generate_vector100()).reshape(100, 1)
    for i in range(1, n):
        a = np.column_stack((a, generate_vector100()))
    return a

def get_distance(mat_a, mat_b):
    # mat_a is basis matrix
    # mat_b is vector

    mat_at = np.transpose(mat_a)
    x_hat = ( np.linalg.inv(mat_at@mat_a) )@(mat_at@mat_b)
    # Ax - b
    distance = np.linalg.norm(mat_a@x_hat - mat_b)
    return distance

def plotDots(x_data, y_data, color='#e35f62', marker='o', linestyle=''):
    plt.plot(x_data, y_data, color=color, marker=marker, linestyle=linestyle)
    plt.show()

def vector_representation(a, u, rank):
    x_axis=[]
    y_axis=[]
    for i in range(2, (rank//5)+1):
        num = 5*i
        x_axis.append(num)
        basis = np.array(u[:,0:num])
        total_distance = 0
        for j in range(0, a.shape[1]):
            total_distance += get_distance(basis, np.array(a[:,j]).reshape(a.shape[0],1))
        y_axis.append(total_distance/a.shape[1])
    plotDots(x_axis, y_axis)
if __name__ == '__main__':
    # n should be 1000 on actual run
    n = 1000
    a = generate_matrix(n)

    u, s, vh = np.linalg.svd(a, full_matrices=True)
    rank = np.linalg.matrix_rank(a)
    # print(s)
    s5 =[]
    for i in range(0, (rank // 5) + 1):
        s5.append(s[i])
    print(s5)
    # print(rank)
    vector_representation(a, u, rank)

