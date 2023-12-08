module source_ACR
using Plots
using LinearAlgebra

function crn_writing(source_mat, product_mat)
    num_S, num_R = size(source_mat)

    source_txt = fill("", num_R)
    product_txt = fill("", num_R)
    source_txt_short = fill("", num_R)
    product_txt_short = fill("", num_R)
    network_txt = fill("", num_R)
    network_txt_short = fill("", num_R)

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
                source_txt_short[i] = source_txt_short[i] * "+" # add "+" symbol if there is anything else to add. 
                plus_flag_source = 0
            end
            if sum(product_mat[(j+1):end, i]) > 0 && plus_flag_product > 0
                product_txt_short[i] = product_txt_short[i] * "+" # add "+" symbol if there is anything else to add. 
                plus_flag_product = 0
            end
        end
        if source_txt_short[i] == ""
            source_txt_short[i] = "0" # add "0" if the source complex of a reaction is the zero complex.
        end
        if product_txt_short[i] == ""
            product_txt_short[i] = "0" # add "0" if the product complex of a reaction is the zero complex.
        end
    end

    network_txt = source_txt .* fill(" -> ", num_R) .* product_txt
    network_txt_short = source_txt_short .* fill(" -> ", num_R) .* product_txt_short

    return network_txt, network_txt_short

end

function crn_embedding(source_mat, product_mat; shake=true)
    num_S, num_R = size(source_mat)
    maxlim = 2 #float(maximum(hcat(source_mat, product_mat)))
    if num_S == 2
        p = plot([0.0 maxlim], [0.0 maxlim], xlabel="A", ylabel="B", legend=false)
        for i in 1:num_R
            if shake == true
                move = rand(num_S) * 0.1 .- 0.05
                #move = randn(num_S) * 0.05
            else
                move = fill(0.0, num_S)
            end
            plot!([source_mat[1, i] + move[1], product_mat[1, i] + move[1]],
                [source_mat[2, i] + move[2], product_mat[2, i] + move[2]],
                arrow=true, color=:black, linewidth=1.5)
        end
    elseif num_S == 3
        p = plot([0.0 maxlim], [0.0 maxlim], [0.0 maxlim], xlabel="A", ylabel="B", zlabel="C", legend=false)
        for i in 1:num_R
            if shake == true
                move = rand(num_S) * 0.1 .- 0.05
                #move = randn(num_S) * 0.05
            else
                move = fill(0.0, num_S)
            end
            plot!([source_mat[1, i] + move[1], product_mat[1, i] + move[1]],
                [source_mat[2, i] + move[2], product_mat[2, i] + move[2]],
                [source_mat[3, i] + move[3], product_mat[3, i] + move[3]],
                arrow=true, color=:black, linewidth=1.5) # Warning: arrows do not appear for a 3D drawing. 
        end
    else
        @warn "Visualization is possible only when the number of species is either 2 or 3. 'nothing' is returned"
        return nothing
    end
    return p
end

function crn_embedding_info(source_mat, product_mat, acr_id, unbnd_id; shake=true)
    num_S, num_R = size(source_mat)
    maxlim = 2 #float(maximum(hcat(source_mat, product_mat)))
    acr_id_tf = acr_id .> 0
    unbnd_id_tf = unbnd_id .> 0
    r = rank(product_mat - source_mat)
    if num_S == 2
        text_id = acr_id_tf[1] * 1 + acr_id_tf[2] * 2 + 1
        acr_sp = ["∅" "A" "B" "A,B"][text_id]
        text_id = unbnd_id_tf[1] * 1 + unbnd_id_tf[2] * 2 + 1
        unbnd_sp = ["∅" "A" "B" "A,B"][text_id]
        #p = plot([0.0 maxlim], [0.0 maxlim], xlabel="A", ylabel="B", legend=false, title="ACR: " * acr_sp * "\ndim(S)=" * string(r))
        p = plot([0.0 maxlim], [0.0 maxlim], xlabel="A", ylabel="B", legend=false)
        for i in 1:num_R
            if shake == true
                move = rand(num_S) * 0.1 .- 0.05
                #move = randn(num_S) * 0.05
            else
                move = fill(0.0, num_S)
            end
            plot!([source_mat[1, i] + move[1], product_mat[1, i] + move[1]],
                [source_mat[2, i] + move[2], product_mat[2, i] + move[2]],
                arrow=true, color=:black, linewidth=1.5)
        end
        #annotate!(1.7,1.7,"ACR: " * acr_sp * "\ndim(S)=" * string(r))
        annotate!((1.25, 1.7, text("ACR: " * acr_sp * "\nUnb: " * unbnd_sp * "\ndim(S)=" * string(r), 9)))
    elseif num_S == 3
        text_id = acr_id_tf[1] * 1 + acr_id_tf[2] * 2 + acr_id_tf[3] * 4 + 1
        acr_sp = ["∅" "A" "B" "A,B" "C" "A,C" "B,C" "A,B,C"][text_id]
        text_id = unbnd_id_tf[1] * 1 + unbnd_id_tf[2] * 2 + unbnd_id_tf[3] * 4 + 1
        unbnd_sp = ["∅" "A" "B" "A,B" "C" "A,C" "B,C" "A,B,C"][text_id]
        #p = plot([0.0 maxlim], [0.0 maxlim], [0.0 maxlim], xlabel="A", ylabel="B", zlabel="C", legend=false, title="ACR: " * acr_sp * "\ndim(S)=" * string(r))
        p = plot([0.0 maxlim], [0.0 maxlim], [0.0 maxlim], xlabel="A", ylabel="B", zlabel="C", legend=false)
        for i in 1:num_R
            if shake == true
                move = rand(num_S) * 0.1 .- 0.05
                #move = randn(num_S) * 0.05
            else
                move = fill(0.0, num_S)
            end
            plot!([source_mat[1, i] + move[1], product_mat[1, i] + move[1]],
                [source_mat[2, i] + move[2], product_mat[2, i] + move[2]],
                [source_mat[3, i] + move[3], product_mat[3, i] + move[3]],
                arrow=true, color=:black, linewidth=1.5) # Warning: arrows do not appear for a 3D drawing. 
        end
        #annotate!(1.7,1.7,1.7,"ACR: " * acr_sp * "\ndim(S)=" * string(r))
        annotate!((1.7, 1.7, 1.7, text("ACR: " * acr_sp * "\nUnb: " * unbnd_sp * "\ndim(S)=" * string(r), 9)))
    else
        @warn "Visualization is possible only when the number of species is either 2 or 3. 'nothing' is returned"
        return nothing
    end
    return p
end

function grouping_network(source_mat, product_mat, acr_id, unbnd_id)
    num_S = size(source_mat)[1]
    acr_id_tf = acr_id .> 0
    unbnd_id_tf = unbnd_id .> 0
    r = rank(product_mat - source_mat)
    gp = nothing
    if num_S == 2
        text_id_acr = acr_id_tf[1] * 1 + acr_id_tf[2] * 2 + 1
        acr_sp = ["∅" "A" "B" "A,B"][text_id_acr]
        text_id_unbnd = unbnd_id_tf[1] * 1 + unbnd_id_tf[2] * 2 + 1
        unbnd_sp = ["∅" "A" "B" "A,B"][text_id_unbnd]
        if text_id_acr == 1 # no ACR
            gp = 1
        elseif text_id_acr == 2 # ACR species: A
            gp = 4*(r-1) + text_id_unbnd + 1
        elseif text_id_acr == 3 # ACR species: B
            gp = 4*(r-1) + text_id_unbnd + 9
        else # text_id_acr == 4 # ACR species: A, B
            gp = 4*(r-1) + text_id_unbnd + 17
        end
    else
        @warn "Grouping network is currently possible only when the number of species is either 2. 'nothing' is returnded"
        return nothing
    end
    return gp
end

function make_total_complexes(num_S, max_order)
    num_total_C = binomial(max_order + num_S, num_S)
    tmp_cmplx = fill(0, num_S)
    total_complex = fill(0, num_S, num_total_C)

    for i in 1:num_total_C
        for j in 1:num_S
            if sum(tmp_cmplx) > max_order
                tmp_cmplx[j] = 0
                tmp_cmplx[j+1] = tmp_cmplx[j+1] + 1
            end
        end
        total_complex[:, i] = tmp_cmplx
        tmp_cmplx[1] = tmp_cmplx[1] + 1
    end
    return total_complex
end

function reaction_to_complex(r_id, num_total_C)
    source_id = Int(floor((r_id - 1) / (num_total_C - 1))) + 1
    product_id = mod((r_id - 1), (num_total_C - 1)) + 1
    if source_id <= product_id
        product_id = product_id + 1
    end
    return source_id, product_id
end

function net_id_to_r_id_list(net_id, num_total_R, num_R)
    tmp_id = net_id
    reaction_id_list = fill(0, num_R) # the list contains the id for reaction r_1, r_2, ..., r_{num_R}.
    # Those id's have to satisfy 1 <= r_1 < r_2 < ... < r_{num_R} <= num_total_R.
    if net_id > binomial(num_total_R, num_R) || net_id < 1
        @warn "The given network id must be between 1 and binomial(num_total_R, num_R). 'nothing' is returned."
        return nothing
    end
    # Determine the first reaction id, r_1.
    for j in 1:(num_total_R-num_R+1)
        tmp_id = tmp_id - binomial(num_total_R - j, num_R - 1)
        if tmp_id <= 0
            tmp_id = tmp_id + binomial(num_total_R - j, num_R - 1)
            reaction_id_list[1] = j
            break
        end
    end

    # Determine the second to last reactions id, r_2, r_3, ... , r_{num_R}.
    for i in 2:num_R
        for j in (reaction_id_list[i-1]+1):(num_total_R-num_R+i)
            tmp_id = tmp_id - binomial(num_total_R - j, num_R - i)
            if tmp_id <= 0
                tmp_id = tmp_id + binomial(num_total_R - j, num_R - i)
                reaction_id_list[i] = j
                break
            end
        end
    end
    return reaction_id_list
end
end

