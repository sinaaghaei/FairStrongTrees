using DataFrames;
using JuMP;
using Gurobi;
using CSV;
using Combinatorics;
file_path = "./../../DataSets/";
using Printf;

# In this code we save the prediction value as a new column called "class_pred" in the data_train and data_test;
# So if we have already a column with this name we may get in trouble



#####################################################
# Arguments that we need from the user
#####################################################
arg = ARGS;
# This is how we should pass the arguments in the terminal
# julia data_group kamiran_version sample depth lambda time_limit


data_group = arg[1]; # or compas, adult, german

# kamiran_version = 1 if we use the kamiran version of the data;
# In this case SP is just the difference between positive outcome probability between non_deprived and deprived group
kamiran_version = parse(Int,arg[2]);
sample = parse(Int,arg[3]);
depth = parse(Int,arg[4]);
lambda = parse(Float64,arg[5]);
time_limit = parse(Int,arg[6]);

train_file_name = data_group*"_train_"*string(sample)*".csv"
test_file_name = data_group*"_test_"*string(sample)*".csv"
calibration_file_name = data_group*"_calibration_"*string(sample)*".csv"







#####################################################
# Reading data
#####################################################
if kamiran_version == 1
    file_path = "./../../DataSets/KamiranVersion/";
else
    file_path = "./../../DataSets/";
end

data_train = CSV.read(file_path*train_file_name ,DataFrame);
N_train = size(data_train,1);

data_test = CSV.read(file_path*test_file_name ,DataFrame);
N_test = size(data_test,1);

data_calibration = CSV.read(file_path*calibration_file_name ,DataFrame);


if data_group == "compas"
    # Need to specify name of the column containing the class label and protected feature
    class = :target; # name of the class label in the dataset
    B = :race; # protected feautre

    # Need to specify which column are categorical and which columns are non-categorical (quantitative)
    # categorical features
    F_c = [:sex,:c_charge_degree];
    nf_c=size(F_c,1);

    # quantitative features
    F_q=[:age_cat,:priors_count,:length_of_stay];
    nf_q=size(F_q,1);

    # We need the class label to be binary (0,1). In this data, the class lables are (1,2)
    data_train[!,class] .-= 1;
    data_test[!,class] .-= 1;


    deprvied_group = 1
    positive_class = 0
end

if data_group == "german_binary"
    # Need to specify name of the column containing the class label and protected feature
    class = :target; # name of the class label in the dataset
    B = :age; # protected feautre

    # Need to specify which column are categorical and which columns are non-categorical (quantitative)
    # categorical features
    F_c = [:check_acc, :credit_history, :purpose, :saving_amo, :present_employment, :p_status, :guatan, :property, :installment, :Housing, :job, :telephn, :foreign_worker];
    nf_c=size(F_c,1);

    # quantitative features
    F_q=[:month_duration, :Credit_amo, :instalrate, :present_resident, :existing_cards, :no_people];
    nf_q=size(F_q,1);

    # We need the class label to be binary (0,1). In this data, the class lables are (1,2)
    data_train[!,class] .-= 1;
    data_test[!,class] .-= 1;

    positive_class = 1
    deprvied_group = 1
end


if data_group == "adult"
    # Need to specify name of the column containing the class label and protected feature
    class = :target; # name of the class label in the dataset
    B = :sex; # protected feautre

    # Need to specify which column are categorical and which columns are non-categorical (quantitative)
    # categorical features
    F_c = [:workclass, :education, :marital_status, :occupation, :relationship, :race, :native_country];
    nf_c=size(F_c,1);

    # quantitative features
    F_q=[:hours_per_week, :age_group, :fnlwgt, :capital];
    nf_q=size(F_q,1);

    # We need the class label to be binary (0,1). In this data, the class lables are (1,2)
    data_train[!,class] .-= 1;
    data_test[!,class] .-= 1;

    deprvied_group = 1
    positive_class = 1
end


#####################################################
#index function
#####################################################
function ind(x)
    if x==true
  1
    else
  0
    end
end




#####################################################
#Tree structures
#####################################################
function get_tree(d)
    nn = 2^d-1
    nl = 2^d

    left=Vector{Array{Int64}}();
    right = Vector{Array{Int64}}();
    for n in 1:nn
        crnt_depth = Int(floor(log(2,n)))
        number_nodes_crnt_depth = 2^crnt_depth
        num_leafs_under_n = 2^(d - crnt_depth)

        first_leaf_idx = (n-number_nodes_crnt_depth)*num_leafs_under_n+1
        last_leaf_idx = first_leaf_idx+ num_leafs_under_n -1
        mid_leaf_idx = Int(floor((first_leaf_idx+last_leaf_idx)/2))
        n_lef = [first_leaf_idx:1:mid_leaf_idx;]
        n_right = [mid_leaf_idx+1:1:last_leaf_idx;]
        push!(left, n_lef);
        push!(right, n_right);
    end

    nn, nl, left, right
end



function get_MIP_model(nn, nl, left, right, data_train, B, class, F_c, F_q, lambda)
    # In this model we assume the class labels in data_train are binary (0 and 1)
    data = deepcopy(data_train);
    N = size(data,1)

    # Parameters
    M=200;
    M2=200;
    ep=0.1;

    fair = 1; # whether we have fairness penalty (fair=1) or not (fair=0)
    if lambda ==0
        fair = 0
    end

    ############################################
    # Defining the decision variables
    ############################################
    model = Model()

    # r[i,l] = |y_i - z_l|*x[i,l] which is the prediction error if i is assigned to l
    # sum(r[i,l] for l=1:nl)=|y_i - yhat_i|
    @variable(model, r[1:N,1:nl]>=0);

    # x[i,l] denotes if datapoint i is assigned to leaf l
    @variable(model, x[1:N,1:nl],Bin);

    # ac[n,j]=1 means that we split on categorical feature j at node n
    # in the paper we call this variable p[n,j]
    @variable(model, ac[1:nn,F_c],Bin);

    # aq[n,j]=1 means that we split on quantitative feature j at node n
    @variable(model, aq[1:nn,F_q],Bin);

    # b[n] is the cut-off value at node n if we split on quantitative features
    @variable(model, b[1:nn]);

    @variable(model, gp[1:N,1:nn]>=0);
    @variable(model, gn[1:N,1:nn]>=0);

    # if wq[i,n]=1 datapoint i should go left at node n; Also this means that we have splitted on a quantitative feature
    @variable(model, wq[1:N,1:nn],Bin);

    # if wc[i,n]=1 datapoint i should go left at node n; Also this means that we have splitted on a categorical feature
    @variable(model, wc[1:N,1:nn],Bin);


    @variable(model, s[1:nn,f in F_c, k in levels(data[!,f])],Bin);


    # v[i,l] = |y_i- z_l|
    @variable(model, v[1:N,1:nl]>=0);

    #z_l is the binary prediction that we make at leaf node l
    @variable(model, z[1:nl], Bin);


    if fair == 1
        # to linearize the prediction in the penalty function
        # rp[i,l] = x[i,l]*z[l]
        # sum(rp[i,l] for l=1:nl) = yhat_i
        @variable(model, rp[1:N,1:nl]>=0);

        #to linearize the absolute value in the penalty function
        # rpp[y,xp] = |P(y) - P(y|xp)|
        @variable(model, rpp[y in levels(data[!,class]), xp in levels(data[!,B])] >=0);
    end;

    ############################################
    # objective
    ############################################
    if fair == 1
        # 1/N*sum(|y_i - yhat_i| over i) + lambda* sum(|P(y) - P(y|xp)| over y and xp)
        @objective(model, Min , (1-lambda)*(1/N)*sum(r[i,l] for i = 1:N , l=1:nl) +
            lambda*(sum(rpp[y,xp] for y in levels(data[!,class]), xp in levels(data[!,B])) ) );
    else
        # 1/N*sum(|y_i - yhat_i| over i)
        @objective(model, Min , (1/N)*sum(r[i,l] for i = 1:N , l=1:nl));
    end;


    ############################################
    #Fairness constraints
    ############################################
    if fair == 1
        # rpp[y,xp] = |P(y) - P(y|xp)|
        for y in levels(data[!,class]), xp in levels(data[!,B])
            # Let's |i: data[i,B]=xp|
            N_xp = size(data[(data[!,B] .== xp),:],1)
            if (N_xp!= 0)
            @constraint(model, sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N)/ N
                             - sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N if (data[i,B]==xp) )/N_xp<= rpp[y,xp]);
            @constraint(model, -sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N)/ N
                               +sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N if (data[i,B]==xp) )/N_xp<= rpp[y,xp]);

            end
        end

        # rp[i,l] = x[i,l]*z[l] so sum(rp[i,l] over l) = yhat_i
        for i in 1:N, l in 1:nl
            @constraint(model,rp[i,l]<=M2*x[i,l]);
            @constraint(model,rp[i,l]<=z[l]);
            @constraint(model,rp[i,l]>=z[l]-M2*(1-x[i,l]));
        end
    end;

    ##############################################
    # nodes constraints
    ##############################################
    for n in 1:nn
        @constraint(model,sum(ac[n,f] for f in F_c)+sum(aq[n,f] for f in F_q)==1);
    end


    ##############################################
    # categoricals
    ##############################################
    for n in 1:nn, f in F_c, k in levels(data[!,f])
        @constraint(model,s[n,f,k] <= ac[n,f]);
    end

    for i in 1:N, n in 1:nn
        @constraint(model,wc[i,n] == sum(sum(s[n,f,k]*ind(data[i,f]==k) for k in levels(data[!,f])) for f in F_c) );
        for l in left[n]
            @constraint(model,x[i,l] <= wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
        for l in right[n]
            @constraint(model,x[i,l]<=1-wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
    end
    ##############################################
    # non categoricals
    ##############################################
    for i in 1:N, n in 1:nn
        @constraint(model,b[n]-sum(aq[n,f]*data[i,f] for f in F_q) == gp[i,n]-gn[i,n]);
        @constraint(model,gp[i,n]<=M*wq[i,n]);
        @constraint(model,gn[i,n]<=M*(1-wq[i,n]));
        @constraint(model,gp[i,n]+gn[i,n]>=ep*(1-wq[i,n]));
        for l in right[n]
            @constraint(model,x[i,l]<=1-wq[i,n]+1-sum(aq[n,f] for f in F_q));
        end
        for l in left[n]
            @constraint(model,x[i,l]<=wq[i,n]+1-sum(aq[n,f] for f in F_q));
        end
    end
    ##############################################
    # linearize prediction error
    ##############################################
    for i in 1:N
        @constraint(model,sum(x[i,l] for l=1:nl)==1);
        for l in 1:nl
            @constraint(model,r[i,l]<=M2*x[i,l]);
            @constraint(model,r[i,l]<=v[i,l]);
            @constraint(model,r[i,l]>=v[i,l]-M2*(1-x[i,l]));
            @constraint(model,v[i,l]>=  data[i,class]-z[l]);
            @constraint(model,v[i,l]>= -data[i,class]+z[l]);
        end
    end


    model

end




function get_predictions(nn,nl,left,right, ac, aq, b, s, z, time_limit, data_org, F_c, F_q)
    data = deepcopy(data_org);
    N = size(data,1)

    #####################################################################
    model_pred = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model_pred, "TimeLimit", time_limit)
    M=200;
    M2=200;
    ep=0.1;
    #####################################################################
    @variable(model_pred, x[1:N,1:nl],Bin);
    @variable(model_pred, gp[1:N,1:nn]>=0);
    @variable(model_pred, gn[1:N,1:nn]>=0);
    @variable(model_pred, wq[1:N,1:nn],Bin);
    @variable(model_pred, wc[1:N,1:nn],Bin);

    # routing constraints for categorical features
    for i in 1:N,n in 1:nn
        @constraint(model_pred,wc[i,n] == sum(sum(s[n,f,k]*ind(data[i,f]==k) for k in levels(data[!,f])) for f in F_c) );
        for l in left[n]
            @constraint(model_pred,x[i,l] <= wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
        for l in right[n]
            @constraint(model_pred,x[i,l]<=1-wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
    end

    # routing constraints for non categoricals features
    for i in 1:N, n in 1:nn
        @constraint(model_pred,b[n]-sum(aq[n,f]*data[i,f] for f in F_q) == gp[i,n]-gn[i,n]);
        @constraint(model_pred,gp[i,n]<=M*wq[i,n]);
        @constraint(model_pred,gn[i,n]<=M*(1-wq[i,n]));
        @constraint(model_pred,gp[i,n]+gn[i,n]>=ep*(1-wq[i,n])); #if b=criteria then w is 1 and go left
        for l in right[n]
            @constraint(model_pred,x[i,l]<=1-wq[i,n]+1-sum(aq[n,f] for f in F_q));
        end
        for l in left[n]
            @constraint(model_pred,x[i,l]<=wq[i,n]+1-sum(aq[n,f] for f in F_q));
        end
    end

    for i in 1:N
        @constraint(model_pred,sum(x[i,l] for l=1:nl)==1);
    end
    #####################################################################
    set_silent(model_pred)
    optimize!(model_pred)
    x = round.(JuMP.value.(x));
    pred= x*z;
    acc = sum(pred .== data[!,class])/N


    pred, acc
end


function get_SP(data, label_souce, B, deprvied_group, positive_class, kamiran_version)
    #data = deepcopy(data_org);
    disc = 0
    if kamiran_version==1
        sp_non_deprived = size(data[(data[!,B] .!= deprvied_group) .& (data[!,label_souce] .== positive_class) ,:],1)/
        size(data[(data[!,B] .!= deprvied_group) ,:],1)

        sp_deprived = size(data[(data[!,B] .== deprvied_group) .& (data[!,label_souce] .== positive_class) ,:],1)/
        size(data[(data[!,B] .== deprvied_group) ,:],1)

        disc = sp_non_deprived - sp_deprived
    else

        for (p, p_prime) in combinations(levels(data[!,B]),2)
            N_p = size(data[(data[!,B] .== p) ,:],1)
            N_p_prime = size(data[(data[!,B] .== p_prime) ,:],1)

            if N_p!=0 && N_p_prime!=0
                sp_p = size(data[(data[!,B] .== p) .& (data[!,label_souce] .== positive_class) ,:],1)/ N_p

                sp_p_prime = size(data[(data[!,B] .== p_prime) .& (data[!,label_souce] .== positive_class) ,:],1)/N_p_prime

                disc = max(disc, abs(sp_p-sp_p_prime))
            end

        end
    end


    disc
end


function get_DIDI(data, class, label_souce, B)
    #data = deepcopy(data_org);
    disc = 0
    N = size(data,1)
    for y in levels(data[!,class]), xp in levels(data[!,B])
        N_xp = size(data[(data[!,B] .== xp),:],1)
        if (N_xp!= 0)
            P_y = size(data[(data[!,label_souce] .== y) ,:],1)/N
            P_y_given_xp = size(data[(data[!,B] .== xp) .& (data[!,label_souce] .== y) ,:],1)/N_xp
            disc+= abs(P_y-P_y_given_xp)
        end
    end

    disc
end

function print_tree(nn, nl, data, F_c, F_q, ac, aq, b, s, z)
    for n in 1:nn
        for j in F_c
            if ac[n,j]==1
                left_levels = []
                for k in levels(data[!,j])
                    if s[n,j,k]==1
                        append!(left_levels,k)
                    end
                end
                @printf("######## Node %d: Go to left if %s is in %s \n", n,j,string(left_levels))
            end
        end

        for j in F_q
            if aq[n,j]==1
                @printf("######## Node %d: Go to left if %s <= %f \n", n,j,b[n])
            end
        end
    end

    for n in 1:nl
        @printf("######## Leaf %d: %d \n",n ,z[n])
    end

end



#####################################################
#
#####################################################

# Let's build the model
nn, nl, left, right = get_tree(depth);
DT_model = get_MIP_model(nn, nl, left, right, data_train, B, class, F_c, F_q, lambda);

# Let's solve the model
set_optimizer_attribute(DT_model, "TimeLimit", time_limit);
set_optimizer(DT_model, Gurobi.Optimizer);
JuMP.optimize!(DT_model);


# Get the values of each decision variable
# We also round the values of binary decision variable to make sure we have sharp 0 and 1s
ac = round.(JuMP.value.(DT_model[:ac]));
aq = round.(JuMP.value.(DT_model[:aq]));
b = JuMP.value.(DT_model[:b]);
s = round.(JuMP.value.(DT_model[:s]));
z = round.(JuMP.value.(DT_model[:z]));


print_tree(nn, nl, data_train, F_c, F_q, ac, aq, b, s, z)


train_pred,train_acc = get_predictions(nn,nl,left,right, ac, aq, b, s, z, time_limit, data_train, F_c, F_q);
data_train[!,:class_pred] = train_pred;
println("train_acc=\t",train_acc)


test_pred,test_acc = get_predictions(nn,nl,left,right, ac, aq, b, s, z, time_limit, data_test, F_c, F_q);
data_test[!,:class_pred] = test_pred;
println("test_acc=\t",test_acc)


calibration_pred,calibration_acc = get_predictions(nn,nl,left,right, ac, aq, b, s, z, time_limit, data_calibration, F_c, F_q);
data_calibration[!,:class_pred] = calibration_pred;
println("calibration_acc=\t",calibration_acc)



train_data_didi = get_DIDI(data_train, class, class, B);
train_pred_didi = get_DIDI(data_train, class, :class_pred, B);

train_data_sp = get_SP(data_train, class,       B, deprvied_group, positive_class, kamiran_version);
train_pred_sp = get_SP(data_train, :class_pred, B, deprvied_group, positive_class, kamiran_version);

println("train_data_didi=\t",train_data_didi)
println("train_pred_didi=\t",train_pred_didi)
println("train_data_sp=\t",train_data_sp)
println("train_pred_sp=\t",train_pred_sp)



test_data_didi = get_DIDI(data_test, class, class, B);
test_pred_didi = get_DIDI(data_test, class, :class_pred, B);

test_data_sp = get_SP(data_test, class,       B, deprvied_group, positive_class, kamiran_version);
test_pred_sp = get_SP(data_test, :class_pred, B, deprvied_group, positive_class, kamiran_version);

println("test_data_didi=\t",test_data_didi)
println("test_pred_didi=\t",test_pred_didi)
println("test_data_sp=\t",test_data_sp)
println("test_pred_sp=\t",test_pred_sp)


calibration_data_didi = get_DIDI(data_calibration, class, class, B);
calibration_pred_didi = get_DIDI(data_calibration, class, :class_pred, B);

calibration_data_sp = get_SP(data_calibration, class,       B, deprvied_group, positive_class, kamiran_version);
calibration_pred_sp = get_SP(data_calibration, :class_pred, B, deprvied_group, positive_class, kamiran_version);

println("calibration_data_didi=\t",calibration_data_didi)
println("calibration_pred_didi=\t",calibration_pred_didi)
println("calibration_data_sp=\t",calibration_data_sp)
println("calibration_pred_sp=\t",calibration_pred_sp)



####################################################################################
#Saving results
####################################################################################
filePath = "./../../Results/"*"MIP_DIDI_"*train_file_name*"_depth_"* string(depth)*"_lambda_"*string(lambda)*".csv";

header=["Approach", "train_file_name", "test_file_name", "kamiran_version", "N_train", "depth", "lambda", "time_limit",
           "obj_val", "gap",
           "train_acc", "train_data_didi", "train_pred_didi", "train_data_sp", "train_pred_sp",
           "test_acc", "test_data_didi", "test_pred_didi", "test_data_sp", "test_pred_sp",
           "calibration_acc", "calibration_data_didi", "calibration_pred_didi", "calibration_data_sp", "calibration_pred_sp"];

row_input=["MIP_DIDI", train_file_name, test_file_name, kamiran_version, N_train, depth, lambda, time_limit,
           objective_value(DT_model), relative_gap(DT_model::Model),
           train_acc, train_data_didi, train_pred_didi, train_data_sp, train_pred_sp,
           test_acc, test_data_didi, test_pred_didi, test_data_sp, test_pred_sp,
           calibration_acc, calibration_data_didi, calibration_pred_didi, calibration_data_sp, calibration_pred_sp];

CSV.write(filePath,  Tables.table(row_input), header = header)
