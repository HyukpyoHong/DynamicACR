include("source_ACR.jl")
import source_ACR as acr

## Initialization
using Pkg
using CSV
using DataFrames
# using DifferentialEquations
# using Interpolations
using Plots
using Printf
# using PlotlyJS
using StatsBase, Statistics, Distributions
using LinearAlgebra

num_S = 2 # number of species
max_order = 2 # the maximum order of reaction. e.g, 1: monomolecular, 2:bimolecular, ...
num_R = 3 # the number of reactions that a random network contains.
num_total_C = binomial(max_order + num_S, num_S)
# num_total_R = num_total_C * (num_total_C - 1)
total_complex = acr.make_total_complexes(num_S, max_order)

path = "/Users/hyukpyohong/Dropbox/CRN_dynamicACR/Codes/data/"
# These lines below are for loading saved data
list_acr_id = Matrix(CSV.read(path * "list_acr_id_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * ".csv", DataFrame, header=false));
list_unbnd_id = Matrix(CSV.read(path * "list_unbnd_id_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * ".csv", DataFrame, header=false));
matrix_R_id = Matrix(CSV.read(path * "matrix_R_id_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * ".csv", DataFrame, header=false));

net_list_with_acr = findall(list_acr_id[1, :] .> 0);
num_total_net = size(list_acr_id)[2]
# num_total_net = binomial(num_total_R, num_R)


# p_arr_2s2r = Array{Plots.Plot{Plots.GRBackend},1}(undef, num_total_net)
# gp_list_2s2r = fill(0, num_total_net)

p_arr_2s3r = Array{Plots.Plot{Plots.GRBackend},1}(undef, num_total_net)
gp_list_2s3r = fill(0, num_total_net)
for check_net_id in 1:num_total_net ## the code below is now designed for one instance of reaction network not for loop.
    if mod(check_net_id, 10) == 1
        @printf("======= Drawing network %4d =======\n", check_net_id)
    end
    list_R_id = matrix_R_id[:, check_net_id]
    list_source_id = fill(0, num_R)
    list_product_id = fill(0, num_R)
    for i in 1:num_R
        s_id, p_id = acr.reaction_to_complex(list_R_id[i], num_total_C)
        list_source_id[i] = s_id
        list_product_id[i] = p_id
    end
    source_mat = total_complex[:, list_source_id]
    product_mat = total_complex[:, list_product_id]
    # The function below is for "writing down" a reaction network. 
    network_txt, network_txt_short = acr.crn_writing(source_mat, product_mat)

    # The function below is for drawing a reaction network. 
    acr_id = list_acr_id[:, check_net_id]
    unbnd_id = list_unbnd_id[:, check_net_id]

    # p_arr_2s2r[check_net_id] = acr.crn_embedding_info(source_mat, product_mat, acr_id, unbnd_id)
    p_arr_2s3r[check_net_id] = acr.crn_embedding_info(source_mat, product_mat, acr_id, unbnd_id)

    # gp_list_2s2r[check_net_id] = acr.grouping_network(source_mat, product_mat, acr_id, unbnd_id)
    gp_list_2s3r[check_net_id] = acr.grouping_network(source_mat, product_mat, acr_id, unbnd_id)
end

path = "/Users/hyukpyohong/Dropbox/CRN_dynamicACR/Codes/plots/"

layout1 = (3, 4)
num_grid = prod(layout1);
blank = plot();

# This code is for making the number of plot be a multiple of the number of grid.

# p_arr = p_arr_2s2r
p_arr_total = p_arr_2s3r
gp_size_check_2s2r = fill(0, 25)
gp_size_check_2s3r = fill(0, 25)
gp_list_2s2r
gp_list_2s3r
for i in gp_list_2s2r
    gp_size_check_2s2r[i] += 1
end

for i in gp_list_2s3r
    gp_size_check_2s3r[i] += 1
end
#println(gp_size_check_2s2r)
#println(gp_size_check_2s3r)

gp = 1
bad_id = findall(x -> x == gp, gp_list_2s2r)
plot(p_arr_2s2r[bad_id[1]])

for gp_id in 1:25
    p_arr_subset = p_arr_total[findall(x -> x == gp_id, gp_list_2s3r)]
    p_arr = p_arr_subset
    tmp1 = length(p_arr)
    if mod(tmp1, num_grid) != 0
        for i in 1:(num_grid-mod(tmp1, num_grid))
            push!(p_arr, blank)
        end
    end

    num_paper = Int(length(p_arr) / num_grid)
    for iter_paper in 1:num_paper
        if mod(iter_paper, 10) == 1
            @printf("======= Drawing paper sheet %4d =======\n", iter_paper)
        end
        ## layout == (3,3) version
        # plot(p_arr[num_grid*(iter_paper-1)+1], p_arr[num_grid*(iter_paper-1)+2], p_arr[num_grid*(iter_paper-1)+3],
        #     p_arr[num_grid*(iter_paper-1)+4], p_arr[num_grid*(iter_paper-1)+5], p_arr[num_grid*(iter_paper-1)+6],
        #     p_arr[num_grid*(iter_paper-1)+7], p_arr[num_grid*(iter_paper-1)+8], p_arr[num_grid*(iter_paper-1)+9],
        #     layout=layout1)

        ## layout == (3,4) version
        plot(p_arr[num_grid*(iter_paper-1)+1], p_arr[num_grid*(iter_paper-1)+2], p_arr[num_grid*(iter_paper-1)+3], p_arr[num_grid*(iter_paper-1)+4],
            p_arr[num_grid*(iter_paper-1)+5], p_arr[num_grid*(iter_paper-1)+6], p_arr[num_grid*(iter_paper-1)+7], p_arr[num_grid*(iter_paper-1)+8],
            p_arr[num_grid*(iter_paper-1)+9], p_arr[num_grid*(iter_paper-1)+10], p_arr[num_grid*(iter_paper-1)+11], p_arr[num_grid*(iter_paper-1)+12],
            layout=layout1)

        ## layout == (4,5) version
        # plot(p_arr[num_grid*(iter_paper-1)+1], p_arr[num_grid*(iter_paper-1)+2], p_arr[num_grid*(iter_paper-1)+3],p_arr[num_grid*(iter_paper-1)+4], p_arr[num_grid*(iter_paper-1)+5],
        # p_arr[num_grid*(iter_paper-1)+6], p_arr[num_grid*(iter_paper-1)+7], p_arr[num_grid*(iter_paper-1)+8],p_arr[num_grid*(iter_paper-1)+9], p_arr[num_grid*(iter_paper-1)+10],
        # p_arr[num_grid*(iter_paper-1)+11], p_arr[num_grid*(iter_paper-1)+12], p_arr[num_grid*(iter_paper-1)+13],p_arr[num_grid*(iter_paper-1)+14], p_arr[num_grid*(iter_paper-1)+15],
        # p_arr[num_grid*(iter_paper-1)+16], p_arr[num_grid*(iter_paper-1)+17], p_arr[num_grid*(iter_paper-1)+18],p_arr[num_grid*(iter_paper-1)+19], p_arr[num_grid*(iter_paper-1)+20],
        #     layout=layout1)
        plot!(size=(700, 500))
        savefig(path * "Grouped_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * "_gp" * string(gp_id) * "_" * string(iter_paper) * ".png")
        #savefig(path * "Filtered_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * "_" * string(iter_paper) * ".png")
    end
end

