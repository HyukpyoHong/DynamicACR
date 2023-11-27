## Initialization
# pakages
# Pkg.add()
using Pkg
using CSV
using DataFrames
using DifferentialEquations
using Interpolations
using Plots
using StatsBase, Statistics

eps_acr = 0.01 # the threshold value for checking ACR. 
# If 0.4 and 0.6 quantiles of steady-state values differ by less than eps_acr, 
# then we say the network has an ACR for a given parameter set.
thres_positive = 0.01 # the threshold value for checking positivity of a steady-state value. 

num_S = 3 # number of species
max_order = 2 # the maximum order of reaction. e.g, 1: monomolecular, 2:bimolecular, ...
num_R = 3 # the number of reactions that a random network contains.

num_repeat_net = 100 # the number of random networks that you want to search/check.
num_repeat_par = 10 # the number of parameter sets that you want to check ACR property.
num_repeat_init = 100 # the number of initial condition sets for a given fixed parameter set to check ACR property.

source_mat = nothing # Initialization of a variable that stores source complexes of a network.
product_mat = nothing # Initialization of a variable that stores product complexes of a network. 
stoi_mat = nothing  # Initialization of a variable for a stoichiometric matrix. 
# These initializations is necessary to use it as a global variable outside the for loops.  

# Define functions at the top of the code. 
# We have to check if this is valid: 
# e.g., the global variable stoi_mat functions properly in this code flow?
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
    end
end

num_C = binomial(max_order + num_S, num_S)

tmp_cmplx = fill(0, num_S)
total_complex = fill(0, num_S, num_C)

for i in 1:num_C
    for j in 1:num_S
        if sum(tmp_cmplx) > max_order
            tmp_cmplx[j] = 0
            tmp_cmplx[j+1] = tmp_cmplx[j+1] + 1
        end
    end
    total_complex[:, i] = tmp_cmplx
    tmp_cmplx[1] = tmp_cmplx[1] + 1
end
# total_complex

## Generate a (mono or bimolecular) random network with the given numbers of species and reactions (num_S and num_R)
num_total_R = num_C * (num_C - 1)
matrix_R_id = fill(0, num_R, num_repeat_net)
list_acr_id = fill(0, num_repeat_net)



## Beginning of the random network search
for iter_network in 1:num_repeat_net
    list_R_id = sample(1:num_total_R, num_R)
    list_source_id = fill(0, num_R)
    list_product_id = fill(0, num_R)
    for i in 1:num_R
        sr_i = Int(floor((list_R_id[i] - 1) / (num_C - 1))) + 1
        pd_i = mod((list_R_id[i] - 1), (num_C - 1)) + 1
        if sr_i <= pd_i
            pd_i = pd_i + 1
        end
        list_source_id[i] = sr_i #Int(sr_i)
        list_product_id[i] = pd_i
    end

    source_mat = total_complex[:, list_source_id]
    product_mat = total_complex[:, list_product_id]

    ## One can manually specify the reaction network using this part. 

    # start comment - examples, use #= (multiline comments) =#
    # example 1: A+B -> 2B, B -> A
    #source_mat = [1 1; 0 1]' # source complex matrix
    #product_mat = [0 2; 1 0]' # 
    # example 2: 2A+B -> 2B, B -> A
    #source_mat = [2 1; 0 1]'
    #product_mat = [0 2; 1 0]'
    # example 3: A+B -> 2B, B -> A, 3A+B -> 2A+2B, 2A+B -> 3A.
    #source_mat = [1 1; 0 1; 3 1; 2 1]'
    #product_mat = [0 2; 1 0; 2 2; 3 0]'
    # end comment - examples 

    stoi_mat = product_mat - source_mat # stoichiometry matrix
    # num_S = size(source_mat)[1]
    # num_R = size(source_mat)[2]

    ## simulation settings: initial conditions, parameters, and the time interval.
    kappa1 = zeros(num_R, 1) # vector of rate constants. It must be a column vector.
    x_init = zeros(num_S, 1) # vector of initial conditions. It must be a column vector.
    tspan1 = (0.0, 30.0) # time interval

    ub_param = 10 * ones(num_R, 1) # vector of the upper bounds for (randomly generated) rate constants. 
    lb_param = zeros(num_R, 1) # vector of the lower bounds for (randomly generated) rate constants. 

    ub_init = 10 * ones(num_S, 1) # vector of the upper bounds for (randomly generated) initial conditions. 
    lb_init = zeros(num_S, 1) # vector of the lower bounds for (randomly generated) initial conditions. 

    list_acr_id_par = fill(0, num_repeat_par)
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

            #add_callback!(prob1, PositiveDomain())
            abstol = [Inf, 1e-1, Inf]
            # reltol = [1e-1, Inf, Inf]

            sol1 = solve(prob1, Tsit5(), saveat=1) # save the solution with the interval 'saveat'. saveat would be 1 or 0.1. 
            # sol1 = solve(prob1);
            sol_mat1 = reduce(hcat, sol1.u)'
            length_tspan = length(sol1.t)
            final_val_mat[:, iter_init] = sol_mat1[length_tspan, :]
            if (Int(floor(0.8 * length_tspan)) >= 1)
                latter_val_mat[:, iter_init] = sol_mat1[Int(floor(0.8 * length_tspan)), :]
            end
        end

        #plot(sol1.t, sol_mat1)
        #plot(sol_mat1[:, 1], sol_mat1[:, 2])

        # Find quantiles
        upp_val = quantile(final_val_mat[1, :], 0.6) # 40th out of 100 for the 1st species
        low_val = quantile(final_val_mat[1, :], 0.4) # 60th out of 100 for the 1st species

        if (upp_val - low_val < eps_acr && low_val > thres_positive)
            list_acr_id_par[iter_par] = 1
        end
    end
    if (sum(list_acr_id_par) > 0)
        list_acr_id[iter_network] = list_acr_id[iter_network] + 1
        # partial ACR (ACR for a subset of parameter sets)
    end
    if (prod(list_acr_id_par) > 0)
        list_acr_id[iter_network] = list_acr_id[iter_network] + 1
        # complete ACR (ACR for a subset of parameter sets)
    end
    matrix_R_id[:, iter_network] = list_R_id
end

list_acr_id
matrix_R_id
path = "/Users/hyukpyohong/Dropbox/CRN_dynamicACR/Codes/"
CSV.write(path * "list_acr_id_S" * string(num_S) * "R" * string(num_R) * ".csv", Tables.table(list_acr_id), writeheader=false)
CSV.write(path * "matrix_R_id_S" * string(num_S) * "R" * string(num_R) * ".csv", Tables.table(matrix_R_id), writeheader=false)


# These lines below are for loading saved data
#xx = CSV.read(path * "list_acr_id_S" * string(num_S) * "R" * string(num_R) * ".csv", DataFrame, header = false)
#mm = CSV.read(path * "matrix_R_id_S" * string(num_S) * "R" * string(num_R) * ".csv", DataFrame, header = false)
#list_acr_id = xx[:,1]
#matrix_R_id = Matrix(mm)

net_list_with_acr = findall(list_acr_id .> 0);

#for check_net_id in net_list_with_acr ## the code below is now designed for one instance of reaction network not for loop.


check_net_id = 3
list_R_id = matrix_R_id[:, check_net_id]
list_source_id = fill(0, num_R)
list_product_id = fill(0, num_R)

source_txt = fill("", num_R);
product_txt = fill("", num_R);
source_txt_short = fill("", num_R);
product_txt_short = fill("", num_R);
network_txt = fill("", num_R);
network_txt_short = fill("", num_R);

for i in 1:num_R
    sr_i = Int(floor((list_R_id[i] - 1) / (num_C - 1))) + 1
    pd_i = mod((list_R_id[i] - 1), (num_C - 1)) + 1
    if sr_i <= pd_i
        pd_i = pd_i + 1
    end
    list_source_id[i] = sr_i #Int(sr_i)
    list_product_id[i] = pd_i
end
println("source:")
source_mat = total_complex[:, list_source_id]
println("product:")
product_mat = total_complex[:, list_product_id]
for i in 1:num_R
    plus_flag_source = 0 # flag variable for source complexes to decide adding "+" symbol in strings
    plus_flag_product = 0 # flag variable for product complexes to decide adding "+" symbol in strings
    for j in 1:num_S
        # create a long version of strings, e.g., 0A+3B+1C
        source_txt[i] = source_txt[i] * string(source_mat[j, i]) * string(Char(64 + j))
        product_txt[i] = product_txt[i] * string(product_mat[j, i]) * string(Char(64 + j))
        if j < num_S
            source_txt[i] = source_txt[i] * "+"
            product_txt[i] = product_txt[i] * "+"
        end

        # create a short version of strings, e.g., 3B+C instead of 0A+3B+1C 
        if source_mat[j, i] > 1
            source_txt_short[i] = source_txt_short[i] * string(source_mat[j, i]) * string(Char(64 + j))
            plus_flag_source = 1
        elseif source_mat[j, i] == 1
            source_txt_short[i] = source_txt_short[i] * string(Char(64 + j))
            plus_flag_source = 1
            # omit "1" for the coefficient, i.e., use "A" instead of "1A"
        end

        if product_mat[j, i] > 1
            product_txt_short[i] = product_txt_short[i] * string(product_mat[j, i]) * string(Char(64 + j))
            plus_flag_product = 1
        elseif product_mat[j, i] == 1
            product_txt_short[i] = product_txt_short[i] * string(Char(64 + j))
            plus_flag_product = 1
            # omit "1" for the coefficient, i.e., use "A" instead of "1A"
        end
        if sum(source_mat[(j+1):end, i]) > 0 && plus_flag_source > 0
            source_txt_short[i] = source_txt_short[i] * "+" # add "=" symbol if there is anything else to add. 
            plus_flag_source = 0
        end
        if sum(product_mat[(j+1):end, i]) > 0 && plus_flag_product > 0
            product_txt_short[i] = product_txt_short[i] * "+" # add "=" symbol if there is anything else to add. 
            plus_flag_product = 0
        end
    end
    if source_txt_short[i] == ""
        source_txt_short[i] = "0" # add "0" is the source complex of a reaction is the zero complex.
    end
    if product_txt_short[i] == ""
        product_txt_short[i] = "0" # add "0" is the product complex of a reaction is the zero complex.
    end
end

network_txt = source_txt .* fill(" -> ", num_R) .* product_txt
network_txt_short = source_txt_short .* fill(" -> ", num_R) .* product_txt_short
#end

source_txt
product_txt
source_txt_short
product_txt_short
network_txt
network_txt_short

## Should set a convergence criterion later.
# Solve the ODE
plot(1:num_repeat_init, final_val_mat[1, :],
    xtickfont=12, ytickfont=12, linewidth=2)
plot!(1:num_repeat_init, latter_val_mat[1, :], linewidth=2)
plot(1:num_repeat_init, final_val_mat[2, :],
    xtickfont=12, ytickfont=12, linewidth=2)
plot!(1:num_repeat_init, latter_val_mat[2, :], linewidth=2)
scatter(final_val_mat[1, :], final_val_mat[2, :], alpha=0.3)
scatter!(final_val_mat[1, :], final_val_mat[2, :], alpha=0.3)

# plot(1:num_repeat_init, final_val_mat[3,:])
# plot!(1:num_repeat_init, latter_val_mat[3,:])
