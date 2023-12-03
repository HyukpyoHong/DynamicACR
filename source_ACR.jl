module source_ACR
using Plots: plot, plot!

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

    return network_txt, network_txt_short

end

function crn_embedding(source_mat, product_mat)
    num_S, num_R = size(source_mat)
    if num_S == 2
        p = plot([0.0 3.0], [0.0 3.0])
        for i in 1:num_R
            plot!([source_mat[1, i], product_mat[1, i]],
                [source_mat[2, i], product_mat[2, i]],
                arrow=true, color=:black, linewidth=2, label="")
        end
    elseif num_S == 3
        p = plot([0.0 3.0], [0.0 3.0], [0.0 3.0])
        for i in 1:num_R
            plot!([source_mat[1, i], product_mat[1, i]],
                [source_mat[2, i], product_mat[2, i]],
                [source_mat[3, i], product_mat[3, i]],
                arrow=true, color=:black, linewidth=2, label="") # Warning: arrows do not appear for a 3D drawing. 
        end
    else
        println("Visualization is possible only when the number of species is either 2 or 3")
        return nothing
    end
    return p
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

end