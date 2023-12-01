module source_ACR

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

end