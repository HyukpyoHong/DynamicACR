include("source_ACR.jl")
import .source_ACR as acr

## Initialization
# pakages
# Pkg.add("Plots")
using Pkg
using CSV
using DataFrames
using DifferentialEquations
using Interpolations
using Plots
using Printf
# using PlotlyJS
using StatsBase, Statistics, Distributions
using LinearAlgebra

eps_acr = 0.01 # the threshold value for checking ACR. 
# If 0.4 and 0.6 quantiles of steady-state values differ by less than eps_acr, 
# then we say the network has an ACR for a given parameter set.
thres_positive = 0.01 # the threshold value for checking positivity of a steady-state value. 

num_S = 2 # number of species
max_order = 2 # the maximum order of reaction. e.g, 1: monomolecular, 2:bimolecular, ...
num_R = 3 # the number of reactions that a random network contains.

source_mat = nothing # Initialization of a variable that stores source complexes of a network.
product_mat = nothing # Initialization of a variable that stores product complexes of a network. 
stoi_mat = nothing  # Initialization of a variable for a stoichiometric matrix. 
# These initializations are necessary to use them as global variables outside of the for loops.  

# Define functions at the top of the code. 
function MAK(dx, x, kappa, t)
    # f: vector of the rate functions WITHOUT rate constants.
    # kappa: vector of the rate constants
    f = zeros(length(kappa), 1)
    for i in 1:length(kappa)
        f[i] = prod(x .^ source_mat[:, i])
    end

    r = kappa .* f #r: vector of the rate functions WITH rate constants.
    dx_vec = stoi_mat * r

    for i in 1:length(x)
        dx[i] = dx_vec[i]
    end
end

function MAK_rsc(dx, x, kappa, t)
    # f: vector of the rate functions WITHOUT rate constants.
    # kappa: vector of the rate constants
    f = zeros(length(kappa), 1)
    for i in 1:length(kappa)
        f[i] = prod(x .^ source_mat[:, i])
    end

    r = kappa .* f #r: vector of the rate functions WITH rate constants.
    dx_vec = stoi_mat * r

    for i in 1:length(x)
        dx[i] = dx_vec[i] / (1 + sqrt(sum(dx_vec .^ 2)))
        # Here, it would be better to change the constant term 1 in the denominator to 0.1 or 0.01 to speed up the convergence of ODEs.
    end
end

num_total_C = binomial(max_order + num_S, num_S)
total_complex = acr.make_total_complexes(num_S, max_order)
# total_complex contains all possible complexes under the max_order (e.g., bimolecular, trimolecular, ...)

## Generate a (mono or bimolecular) random network with the given numbers of species and reactions (num_S and num_R)
num_total_R = num_total_C * (num_total_C - 1)
num_total_net = binomial(num_total_R, num_R)

num_repeat_net = num_total_net # the number of random networks that you want to search/check.
num_repeat_par = 5 # the number of parameter sets that you want to check ACR property.
num_repeat_init = 20 # the number of initial condition sets for a given fixed parameter set to check ACR property.

matrix_R_id = fill(0, num_R, num_repeat_net)
list_acr_id = fill(0, num_S, num_repeat_net)
list_unbnd_id = fill(0, num_S, num_repeat_net)
## Beginning of the random network search
rnd_search = false
for iter_network in 1:num_repeat_net
    if mod(iter_network, 10) == 1
        @printf("======= Analyzing network %4d =======\n", iter_network)
    end
    if rnd_search == true # search networks randomly.
        list_R_id = sample(1:num_total_R, num_R)
    else # search networks in order.
        list_R_id = acr.net_id_to_r_id_list(iter_network, num_total_R, num_R)
        # The value of num_repeat_net cannot exceed the value of num_total_net in this case.
    end
    list_source_id = fill(0, num_R)
    list_product_id = fill(0, num_R)
    for i in 1:num_R
        s_id, p_id = acr.reaction_to_complex(list_R_id[i], num_total_C)
        list_source_id[i] = s_id
        list_product_id[i] = p_id
    end
    source_mat = total_complex[:, list_source_id]
    product_mat = total_complex[:, list_product_id]
    stoi_mat = product_mat - source_mat # stoichiometry matrix

    ## simulation settings: initial conditions, parameters, and the time interval.
    kappa1 = zeros(num_R, 1) # vector of rate constants. It must be a column vector.
    x_init = zeros(num_S, 1) # vector of initial conditions. It must be a column vector.
    tspan1 = (0.0, 50.0) # time interval

    ub_param = 10 * ones(num_R, 1) # vector of the upper bounds for (randomly generated) rate constants. 
    lb_param = zeros(num_R, 1) # vector of the lower bounds for (randomly generated) rate constants. 

    ub_init = 20 * ones(num_S, 1) # vector of the upper bounds for (randomly generated) initial conditions. 
    lb_init = 1 * ones(num_S, 1) # vector of the lower bounds for (randomly generated) initial conditions. 

    list_acr_id_par = fill(0, num_S, num_repeat_par) # (i,j)-entry is 1 if the i-th species shows the dynamic ACR under the j-th parameter set.
    list_unbnd_id_par = fill(0, num_S, num_repeat_par) # (i,j)-entry is 1 if the i-th species shows an unbounded trajectory under the j-th parameter set.
    for iter_par in 1:num_repeat_par
        for i in 1:num_R
            kappa1[i] = lb_param[i] + (ub_param[i] - lb_param[i]) * rand()
        end

        final_val_mat = zeros(num_S, num_repeat_init)
        latter_val_mat = zeros(num_S, num_repeat_init)
        for iter_init in 1:num_repeat_init
            for i in 1:num_S
                x_init[i] = lb_init[i] + (ub_init[i] - lb_init[i]) * rand()
            end

            prob1 = ODEProblem(MAK_rsc, x_init, tspan1, kappa1) # Use the rescaled mass-action kinetics
            #prob1 = ODEProblem(MAK, x_init, tspan1, kappa1) # Use the original mass-action kinetics

            abstol = [Inf, 1e-1, Inf]
            # reltol = [1e-1, Inf, Inf]

            sol1 = solve(prob1, Tsit5(), saveat=1) # save the solution with the interval 'saveat'. saveat would be 1 or 0.1. 
            sol_mat1 = reduce(hcat, sol1.u)'
            length_tspan = length(sol1.t)
            final_val_mat[:, iter_init] = sol_mat1[length_tspan, :]
            if (Int(floor(0.8 * length_tspan)) >= 1)
                latter_val_mat[:, iter_init] = sol_mat1[Int(floor(0.8 * length_tspan)), :]
            end
        end
        
        # Find quantiles
        upp_val = zeros(num_S)
        low_val = zeros(num_S)
        for i in 1:num_S
            upp_val[i] = quantile(final_val_mat[i, :], 0.6) # 40th out of 100 for the i-th species
            low_val[i] = quantile(final_val_mat[i, :], 0.4) # 60th out of 100 for the i-th species      
        end

        for i in 1:num_S
            if (upp_val[i] - low_val[i] < eps_acr && low_val[i] > thres_positive) # Do we need a positivity criterion for steady states?
                list_acr_id_par[i, iter_par] = 1
            end
            if mean(latter_val_mat[i, :]) > ub_init[i] && mean(final_val_mat[i, :]) > mean(latter_val_mat[i, :]) + 2
                list_unbnd_id_par[i, iter_par] = 1
            end
        end
    end

    for i in 1:num_S
        if (sum(list_acr_id_par[i, :]) > 0)
            list_acr_id[i, iter_network] = list_acr_id[i, iter_network] + 1
            # partial ACR (ACR for a subset of parameter sets)
        end
        if (prod(list_acr_id_par[i, :]) > 0)
            list_acr_id[i, iter_network] = list_acr_id[i, iter_network] + 1
            # complete ACR (ACR for all parameter sets)
        end
        if (sum(list_unbnd_id_par[i, :]) > 0)
            list_unbnd_id[i, iter_network] = list_acr_id[i, iter_network] + 1
            # partial unboundedness (unboundedness for a subset of parameter sets)
        end
        if (prod(list_unbnd_id_par[i, :]) > 0)
            list_unbnd_id[i, iter_network] = list_acr_id[i, iter_network] + 1
            # complete unboundedness (unboundedness for all parameter sets)
        end
    end
    matrix_R_id[:, iter_network] = list_R_id
end

list_acr_id
list_unbnd_id
matrix_R_id
path = "/Users/hyukpyohong/Dropbox/CRN_dynamicACR/Codes/data/"
CSV.write(path * "list_acr_id_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * ".csv", Tables.table(list_acr_id), writeheader=false)
CSV.write(path * "list_unbnd_id_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * ".csv", Tables.table(list_unbnd_id), writeheader=false)
CSV.write(path * "matrix_R_id_S" * string(num_S) * "R" * string(num_R) * "_max_ord" * string(max_order) * ".csv", Tables.table(matrix_R_id), writeheader=false)
