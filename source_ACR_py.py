import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import binom
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities.iterables import multiset_permutations

def MAK(t, x, kappa, source_mat, stoi_mat):
    f = np.zeros(len(kappa))
    for i in range(len(kappa)):
        f[i] = np.prod(np.power(x, source_mat[:,i]))
    r = kappa * f
    dx_vec = np.matmul(stoi_mat, r)
    return dx_vec

def MAK_rescaled(t, x, kappa, source_mat, stoi_mat):
    f = np.zeros(len(kappa))
    for i in range(len(kappa)):
        f[i] = np.prod(np.power(x, source_mat[:,i]))
    r = kappa * f
    dx_vec = np.matmul(stoi_mat, r)
    dx_vec_rsc = dx_vec / (1 + np.sqrt(np.sum(dx_vec**2)))
    return dx_vec_rsc

def make_total_complexes(num_S, max_order):
    num_total_C = int(binom(max_order + num_S, num_S))
    tmp_cmplx = np.zeros(num_S, dtype = int)
    total_complex = np.zeros((num_S, num_total_C), dtype = int)

    for i in range(num_total_C):
        for j in range(num_S):
            if np.sum(tmp_cmplx) > max_order:
                tmp_cmplx[j] = 0
                tmp_cmplx[j + 1] += 1
        #total_complex[:, i] = tmp_cmplx.copy()
        total_complex[:, i] = tmp_cmplx.copy()
        tmp_cmplx[0] += 1
        
    return total_complex

def complex_to_complex_id(cmplx, num_S, max_order):
    num_total_C = int(binom(max_order + num_S, num_S)) # == total_complex.shape[1]
    total_complex = make_total_complexes(num_S, max_order)
    
    # brute force searching
    for cmplx_id in range(1,num_total_C+1):
        if np.prod(total_complex[:,cmplx_id-1] == cmplx):
            return cmplx_id

def reaction_id_to_complex_id(R_id, num_total_C):
    source_id = int(np.floor((R_id - 1) / (num_total_C - 1))) + 1
    product_id = int(np.mod((R_id - 1), (num_total_C - 1))) + 1

    if source_id <= product_id:
        product_id += 1

    return source_id, product_id

def complex_id_to_reaction_id(source_id, product_id, num_total_C):
    reaction_id = (source_id - 1)*(num_total_C-1) + product_id
    
    if source_id <= product_id:
        reaction_id -= 1
    
    return reaction_id

def net_id_to_R_id_list(net_id, num_total_R, num_R):
    tmp_id = net_id
    reaction_id_list = np.zeros(num_R, dtype=int) # the list contains the id for reaction r_1, r_2, ..., r_{num_R}.
    # Those id's have to satisfy 1 <= r_1 < r_2 < ... < r_{num_R} <= num_total_R.

    if net_id > int(binom(num_total_R, num_R)) or net_id < 1:
        print("The given network id must be between 1 and binom(num_total_R, num_R). 'None' is returned.")
        return None
    # Determine the first reaction id, r_1.
    for j in range(1, num_total_R - num_R + 2):
        tmp_id -= int(binom(num_total_R - j, num_R - 1))

        if tmp_id <= 0:
            tmp_id += int(binom(num_total_R - j, num_R - 1))
            reaction_id_list[0] = j
            break
    
    # Determine the second to last reactions id, r_2, r_3, ... , r_{num_R}.
    for i in range(2, num_R + 1):
        for j in range(reaction_id_list[i - 2] + 1, num_total_R - num_R + i + 1):
            tmp_id -= int(binom(num_total_R - j, num_R - i))

            if tmp_id <= 0:
                tmp_id += int(binom(num_total_R - j, num_R - i))
                reaction_id_list[i - 1] = j
                break

    return reaction_id_list

def R_id_list_to_net_id(R_id_list_in, num_total_R):
    R_id_list = np.sort(R_id_list_in)
    num_R = len(R_id_list)
    num_total_net = int(binom(num_total_R, num_R))
    single_id = np.sum(R_id_list * (((num_total_R+1) * np.ones(num_R)) ** np.flip(range(num_R))))
    # this single_id is a unique identifier for R_id_list

    search_lb = 1 # lower bound of the searching interval
    search_ub = num_total_net # upper bound of the searching interval
    
    # perform binary search to find the correct net_id
    while True:
        search_net_id = int(np.floor((search_lb + search_ub)/2))
        tmp_R_id_list = net_id_to_R_id_list(search_net_id, num_total_R, num_R)
        tmp_single_id = np.sum(tmp_R_id_list * (((num_total_R+1) * np.ones(num_R)) ** np.flip(range(num_R))))
        #print(search_net_id)
        if tmp_single_id == single_id:
            return search_net_id
        elif tmp_single_id > single_id:
            search_ub = search_net_id - 1
        else:
            search_lb = search_net_id + 1

def crn_writing(source_mat, product_mat):
    num_S, num_R = source_mat.shape

    source_txt = [""] * num_R
    product_txt = [""] * num_R
    source_txt_short = [""] * num_R
    product_txt_short = [""] * num_R
    network_txt = [""] * num_R
    network_txt_short = [""] * num_R

    for i in range(num_R):
        plus_flag_source = 0 # flag variable for source complexes to decide adding "+" symbol in strings
        plus_flag_product = 0 # flag variable for product complexes to decide adding "+" symbol in strings
        for j in range(num_S):
            # create a long version of strings, e.g., 0A+3B+1C
            source_txt[i] += str(source_mat[j, i]) + chr(64 + j + 1)
            product_txt[i] += str(product_mat[j, i]) + chr(64 + j + 1)
            if j < num_S - 1:
                source_txt[i] += "+"
                product_txt[i] += "+"
            
            # create a short version of strings, e.g., 3B+C instead of 0A+3B+1C 
            if source_mat[j, i] > 1:
                source_txt_short[i] += str(source_mat[j, i]) + chr(64 + j + 1)
                plus_flag_source = 1
            elif source_mat[j, i] == 1:
                source_txt_short[i] += chr(64 + j + 1)
                plus_flag_source = 1
                # omit "1" for the coefficient, i.e., use "A" instead of "1A"

            if product_mat[j, i] > 1:
                product_txt_short[i] += str(product_mat[j, i]) + chr(64 + j + 1)
                plus_flag_product = 1
            elif product_mat[j, i] == 1:
                product_txt_short[i] += chr(64 + j + 1)
                plus_flag_product = 1
                # omit "1" for the coefficient, i.e., use "A" instead of "1A"

            if np.sum(source_mat[j + 1:, i]) > 0 and plus_flag_source > 0:
                source_txt_short[i] += "+" # add "+" symbol if there is anything else to add. 
                plus_flag_source = 0

            if np.sum(product_mat[j + 1:, i]) > 0 and plus_flag_product > 0:
                product_txt_short[i] += "+" # add "+" symbol if there is anything else to add. 
                plus_flag_product = 0

        if source_txt_short[i] == "":
            source_txt_short[i] = "0" 
            # add "0" if the source complex of a reaction is the zero complex.

        if product_txt_short[i] == "":
            product_txt_short[i] = "0" 
            # add "0" if the source complex of a reaction is the zero complex.

    network_txt = [f"{source} -> {product}" for source, product in zip(source_txt, product_txt)]
    network_txt_short = [f"{source} -> {product}" for source, product in zip(source_txt_short, product_txt_short)]

    return network_txt, network_txt_short

def crn_embedding(source_mat, product_mat, shake=True):
    num_S, num_R = source_mat.shape
    axes_lim = [-0.1, 2.1]

    if num_S == 2:
        fig, ax = plt.subplots()
        # ax.plot([0.0, maxlim], [0.0, maxlim])
        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.set_xticks([0,1,2]);
        ax.set_yticks([0,1,2]);
        ax.set_xlabel('A');
        ax.set_ylabel('B');
        ax.grid()
        for i in range(num_R):
            if shake:
                move = np.random.rand(num_S) * 0.14 - 0.07
            else:
                move = np.zeros(num_S)

            ax.arrow(source_mat[0, i] + move[0], source_mat[1, i] + move[1],
                     product_mat[0, i] - source_mat[0, i], product_mat[1, i] - source_mat[1, i],
                     head_width=0.15, head_length=0.15, length_includes_head = True, fc='black', ec='black')
    elif num_S == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot([0.0, maxlim], [0.0, maxlim], [0.0, maxlim])
        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.set_zlim(axes_lim)
        ax.set_xticks([0,1,2]);
        ax.set_yticks([0,1,2]);
        ax.set_zticks([0,1,2]);
        ax.set_xlabel('A');
        ax.set_ylabel('B');
        ax.set_zlabel('C');
        ax.grid()
        for i in range(num_R):
            if shake:
                move = np.random.rand(num_S) * 0.14 - 0.07
            else:
                move = np.zeros(num_S)

            ax.quiver(source_mat[0, i] + move[0], source_mat[1, i] + move[1], source_mat[2, i] + move[2],
                      product_mat[0, i] - source_mat[0, i], product_mat[1, i] - source_mat[1, i],
                      product_mat[2, i] - source_mat[2, i], color='black')
    else:
        print("Visualization is possible only when the number of species is either 2 or 3. 'None' is returned")
        return None

    # plt.show()
    return fig, ax

def crn_embedding_info(source_mat, product_mat, acr_id, unbnd_id, shake=True):
    num_S, num_R = source_mat.shape
    axes_lim = [-0.1, 2.1]
    acr_id_tf = acr_id > 0
    unbnd_id_tf = unbnd_id > 0
    r = np.linalg.matrix_rank(product_mat - source_mat)

    if num_S == 2:
        text_id_acr = acr_id_tf[0] * 1 + acr_id_tf[1] * 2
        acr_sp = ["∅", "A", "B", "A,B"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[0] * 1 + unbnd_id_tf[1] * 2
        unbnd_sp = ["∅", "A", "B", "A,B"][text_id_unbnd]

        fig, ax = plt.subplots()
        #ax.plot([0.0, maxlim], [0.0, maxlim])
        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.set_xticks([0,1,2]);
        ax.set_yticks([0,1,2]);
        ax.set_xlabel('A');
        ax.set_ylabel('B');
        ax.grid()
        for i in range(num_R):
            if shake:
                move = np.random.rand(num_S) * 0.14 - 0.07
            else:
                move = np.zeros(num_S)

            ax.arrow(source_mat[0, i] + move[0], source_mat[1, i] + move[1],
                     product_mat[0, i] - source_mat[0, i], product_mat[1, i] - source_mat[1, i],
                     head_width=0.15, head_length=0.15, length_includes_head = True, fc='black', ec='black', linewidth=1.5)

        ax.text(1.2, 1.4, f"ACR: {acr_sp}\nUnb: {unbnd_sp}\ndim(S): {r}", fontsize=9)
    elif num_S == 3:
        text_id_acr = acr_id_tf[0] * 1 + acr_id_tf[1] * 2 + acr_id_tf[2] * 4
        acr_sp = ["∅", "A", "B", "A,B", "C", "A,C", "B,C", "A,B,C"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[0] * 1 + unbnd_id_tf[1] * 2 + unbnd_id_tf[2] * 4
        unbnd_sp = ["∅", "A", "B", "A,B", "C", "A,C", "B,C", "A,B,C"][text_id_unbnd]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot([0.0, maxlim], [0.0, maxlim], [0.0, maxlim])
        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.set_zlim(axes_lim)
        ax.set_xticks([0,1,2]);
        ax.set_yticks([0,1,2]);
        ax.set_zticks([0,1,2]);
        ax.set_xlabel('A');
        ax.set_ylabel('B');
        ax.set_zlabel('C');
        ax.grid()
        for i in range(num_R):
            if shake:
                move = np.random.rand(num_S) * 0.14 - 0.07
            else:
                move = np.zeros(num_S)

            ax.quiver(source_mat[0, i] + move[0], source_mat[1, i] + move[1], source_mat[2, i] + move[2],
                      product_mat[0, i] - source_mat[0, i], product_mat[1, i] - source_mat[1, i],
                      product_mat[2, i] - source_mat[2, i], color='black')

        ax.text(1.2, 1.4, 1.4, f"ACR: {acr_sp}\nUnb: {unbnd_sp}\ndim(S): {r}", fontsize=9)
    else:
        print("Visualization is possible only when the number of species is either 2 or 3. 'None' is returned")
        return None

    # plt.show()
    return fig, ax

def add_CRN_to_Axes(axes, row, col, source_mat, product_mat, acr_id, unbnd_id, net_id):
    ax = axes[row,col]
    num_S, num_R = source_mat.shape
    axes_lim = [-0.1, 2.1]
    shake = True

    acr_id_tf = acr_id > 0
    unbnd_id_tf = unbnd_id > 0
    r = np.linalg.matrix_rank(product_mat - source_mat)
        
    if num_S == 2:
        text_id_acr = acr_id_tf[0] * 1 + acr_id_tf[1] * 2
        acr_sp = ["∅", "A", "B", "A,B"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[0] * 1 + unbnd_id_tf[1] * 2
        unbnd_sp = ["∅", "A", "B", "A,B"][text_id_unbnd]

        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.set_xticks([0,1,2]);
        ax.set_yticks([0,1,2]);
        ax.set_xlabel('A');
        ax.set_ylabel('B');
        ax.grid()
        for i in range(num_R):
            if shake:
                move = np.random.rand(num_S) * 0.14 - 0.07
            else:
                move = np.zeros(num_S)
            
            ax.arrow(source_mat[0, i] + move[0], source_mat[1, i] + move[1],
                     product_mat[0, i] - source_mat[0, i], product_mat[1, i] - source_mat[1, i],
                     head_width=0.15, head_length=0.15, length_includes_head = True, fc='black', ec='black')
        ax.text(1.2, 1.4, f"ACR: {acr_sp}\nUnb: {unbnd_sp}\ndim(S): {r}\nID: {net_id}", fontsize=9)
    elif num_S == 3:
        text_id_acr = acr_id_tf[0] * 1 + acr_id_tf[1] * 2 + acr_id_tf[2] * 4
        acr_sp = ["∅", "A", "B", "A,B", "C", "A,C", "B,C", "A,B,C"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[0] * 1 + unbnd_id_tf[1] * 2 + unbnd_id_tf[2] * 4
        unbnd_sp = ["∅", "A", "B", "A,B", "C", "A,C", "B,C", "A,B,C"][text_id_unbnd]

        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.set_zlim(axes_lim)
        ax.set_xticks([0,1,2]);
        ax.set_yticks([0,1,2]);
        ax.set_zticks([0,1,2]);
        ax.set_xlabel('A');
        ax.set_ylabel('B');
        ax.set_zlabel('C');
        ax.grid()
        for i in range(num_R):
            if shake:
                move = np.random.rand(num_S) * 0.14 - 0.07
            else:
                move = np.zeros(num_S)
                
            ax.quiver(source_mat[0, i] + move[0], source_mat[1, i] + move[1], source_mat[2, i] + move[2],
                      product_mat[0, i] - source_mat[0, i], product_mat[1, i] - source_mat[1, i],
                      product_mat[2, i] - source_mat[2, i], color='black', linewidth=1.5)
        ax.text(1.2, 1.4, 1.4, f"ACR: {acr_sp}\nUnb: {unbnd_sp}\ndim(S): {r}\nID: {net_id}", fontsize=9)
    else:
        print("Visualization is possible only when the number of species is either 2 or 3. 'None' is returned")
        return None
    return None

def grouping_network(source_mat, product_mat, acr_id, unbnd_id):
    num_S = source_mat.shape[0]
    acr_id_tf = acr_id > 0
    unbnd_id_tf = unbnd_id > 0
    r = np.linalg.matrix_rank(product_mat - source_mat)
    #gp = None

    if num_S == 2:
        text_id_acr = acr_id_tf[0] * 1 + acr_id_tf[1] * 2
        #acr_sp = ["∅", "A", "B", "A,B"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[0] * 1 + unbnd_id_tf[1] * 2
        #unbnd_sp = ["∅", "A", "B", "A,B"][text_id_unbnd]
        gp_str = str(r)+str(text_id_acr)+str(text_id_unbnd)
    elif num_S == 3:
        text_id_acr = acr_id_tf[0] * 1 + acr_id_tf[1] * 2 + acr_id_tf[2] * 4
        #acr_sp = ["∅", "A", "B", "A,B", "C", "A,C", "B,C", "A,B,C"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[0] * 1 + unbnd_id_tf[1] * 2 + unbnd_id_tf[2] * 4
        #unbnd_sp = ["∅", "A", "B", "A,B", "C", "A,C", "B,C", "A,B,C"][text_id_unbnd]
        gp_str = str(r)+str(text_id_acr)+str(text_id_unbnd)
    else:
        print("Grouping network is currently possible only when the number of species is either 2 or 3. 'None' is returned")
        return None

    return gp_str

def find_equiv_network_ids(net_id, num_S, max_order, num_R):

    num_total_C = int(binom(max_order + num_S, num_S))
    total_complex = make_total_complexes(num_S, max_order)

    num_total_R = num_total_C * (num_total_C - 1)
    num_total_net = int(binom(num_total_R, num_R))
    list_R_id = net_id_to_R_id_list(net_id, num_total_R, num_R)
    list_source_id, list_product_id = zip(*[reaction_id_to_complex_id(R_id, num_total_C) for R_id in list_R_id])
    source_mat = total_complex[:, np.array(list_source_id)-1]
    product_mat = total_complex[:, np.array(list_product_id)-1]

    S_sorted = np.array(range(num_S))
    equiv_net_ids = np.array([], dtype=int)
    for S_permuted in multiset_permutations(S_sorted):
        new_source_mat = source_mat[S_permuted,:]
        new_product_mat = product_mat[S_permuted,:]
        new_list_source_id = [complex_to_complex_id(new_source_mat[:,i], num_S, max_order) for i in range(num_R)]
        new_list_product_id = [complex_to_complex_id(new_product_mat[:,i], num_S, max_order) for i in range(num_R)]
        new_list_R_id = [complex_id_to_reaction_id(new_list_source_id[i], new_list_product_id[i], num_total_C) 
                         for i in range(num_R)]
        new_net_id = R_id_list_to_net_id(new_list_R_id, num_total_R)
        equiv_net_ids = np.append(equiv_net_ids, new_net_id)

    return equiv_net_ids
    
    
    