## approach 1
using DataFrames;
using JuMP
using Gurobi
using CSV
using JLD
arg = ARGS;
#########################################
#julia file.jl sample(s1 s2 ...) N_tr start_tr fair lambda nothing N_tes start_tes time_limit MIP_GAP lambda_denominator data_set depth
# julia app1.jl s0 200 1 1 1 00 100 900 34000 5 1 df none

# arg[1]= sample name
# arg[2]= training size
# arg[3]=training start
# arg[4]= fairness
# arg[5]= lambda numerator
# arg[6]= nothing
# arg[7]= test size
# arg[8]= test start
# arg[9]=time limit
# arg[10]=mip gap numerator
# arg[11]=lambda denominator
# arg[12]=data set name
# arg[13]=depth of tree
#########################################

#########################################
#Choosing data set and loading data
#########################################
# default
if arg[12] == "df"
    tr_file = "defa.csv"
    tes_file = "defa.csv"
    tr = CSV.read(tr_file); # training data
    tr_save = tr # save the training set here
    class = :class; #name of the class feature in the dataset
    tes = CSV.read(tes_file); # training data
    tes_save = tes; # save the training set here
end
#########################################
#Census (adult)
if arg[12] == "cen"
    tr_file = "census_income_tr.csv"
    tes_file = "census_income_tes.csv"
    tr = CSV.read(tr_file); # training data
    tr_save = tr; # save the training set here
    class = :class; #name of the class feature in the dataset
    tes = CSV.read(tes_file); # training data
    tes_save = tes; # save the training set here
end
################################
#compas
if arg[12] == "com"
    tr_file = "com.csv"
    tes_file = "com.csv"
    tr = CSV.read(tr_file); # training data
    tr_save = tr; # save the training set here
    class = :class; #name of the class feature in the dataset
    tes = CSV.read(tes_file); # training data
    tes_save = tes; # save the training set here
end
#####################################################
#Sampling Process: making problem smaller
#####################################################
tr = tr_save; # restore the training if needed
N_o = size(tr,1) #original number of observations
### make problem smaller
sample=arg[1];
print("sample is " , arg[1], "\n");
N = parse(Int,arg[2]);
start = parse(Int,arg[3])
tr=tr[start:start+N-1,:];
N = size(tr,1) # make sure N is size of the training
########################################
# tes = tes_save; # restore the training if needed
N_o_tes = size(tes,1) #original number of observations
### make problem smaller
N_tes = parse(Int,arg[7]);
start_tes = parse(Int,arg[8])
tes=tes[start_tes:start_tes+N_tes-1,:];
#####################################################
#Parameter setting
#####################################################
# fairness on off
classes = 0:1;
fair = parse(Int,arg[4]);
lambda = parse(Int,arg[5])/parse(Int,arg[11]);
depth=arg[13];
time_lim=parse(Int,ARGS[9]);
if arg[12] == "df"
B = :SEX; #protected feautre
end
if arg[12] == "cen" || arg[12] == "com"
B = :race; #protected feautre
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
#defining categorical and non-categorical features
#####################################################
# default
if arg[12] == "df"
    # features default dataset
    #categorical features
    F_c = [:SEX, :EDUCATION, :MARRIAGE];
    F_c = F_c[F_c .!= B,]; # remove B from prediction and learning
    nf_c=size(F_c,1);

    # non categorical
    F_nc=[:LIMIT_BAL,:AGE, :PAY_0, :PAY_2, :PAY_3, :PAY_4, :PAY_5, :PAY_6, :BILL_AMT1,
     :BILL_AMT2, :BILL_AMT3, :BILL_AMT4, :BILL_AMT5, :BILL_AMT6, :PAY_AMT1, :PAY_AMT2, :PAY_AMT3, :PAY_AMT4, :PAY_AMT5, :PAY_AMT6];
    F_nc = F_nc[F_nc.!= B,]
    nf_nc=size(F_nc,1)
end
########################################
# Census
    if arg[12] == "cen"
    #categorical features
    F_c = [:workclass,:education, :marital_status, :occupation, :relationship,:race, :sex];
    nf_c=size(F_c,1);

    # non categorical
    F_nc=[:age,:fnlwgt , :education_num, :capital_gain ,:capital_lossh ,:hours_per_week ];
    nf_nc=size(F_nc,1);
end
#####################
#COMPAS
if arg[12] == "com"
    #categorical features
    F_c = [:sex,:race, :c_charge_degree,:is_recid, :r_charge_degree, :v_score_text,:score_text];
    nf_c=size(F_c,1);

    # non categorical
    F_nc=[:age,:juv_fel_count,:decile_score,:juv_misd_count,:juv_other_count,:priors_count,:days_b_screening_arrest,:c_days_from_compas,:v_decile_score ];
    nf_nc=size(F_nc,1);
end
#####################################################
#Tree structures
#####################################################
if arg[12] == "df"
    #tree structure (S1 S2 S4 S5)
    if arg[1] == "s1" || arg[1] == "s2" || arg[1] == "s4" || arg[1] == "s5"
        nn=3; # number of nodes
        nl=4;   #number of leaves
        left=[[1],[2 3],[2]];
        right=[[2 3 4],[4],[3]];
    end
    #tree structure (S3)
    if arg[1] == "s3"
        nn=4; # number of nodes
        nl=5;   #number of leaves
        left=[[1 2 3],[1],[2],[4]];
        right=[[4 5],[2 3],[3],[5]];
    end;
    # if arg[13] == "dp5"
    #     nn=6; # number of nodes
    #     nl=7;   #number of leaves
    #     left=[[1 2 3],[1],[2],[4 5 6],[4],[5]];
    #     right=[[4 5 6 7],[2 3],[3],[7],[5 6],[6]];
    # end;
    # if arg[13] == "dp6"
    #     nn=12; # number of nodes
    #     nl=13;   #number of leaves
    #     left=[[1 2 3 4 5 6 7],[1 2 3 4 5],[1 2],[1],[3],[4],[6],[8 9 10 11],[8],[9],[10],[12]];
    #     right=[[8 9 10 11 12 13],[6 7],[3 4 5],[2],[4 5],[5],[7],[12 13],[9 10 11],[10 11],[11],[13]];
    # end;
    # if arg[13] == "dp3"
    #     nn=3; # number of nodes
    #     nl=4;   #number of leaves
    #     left=[[1 2],[1],[3]];
    #     right=[[3 4],[2],[4]];
    # end;
    # if arg[13] == "dp4"
    #     nn=7; # number of nodes
    #     nl=8;   #number of leaves
    #     left=[[1 2 3 4],[1 2],[1],[3],[5 6],[5],[7]];
    #     right=[[5 6 7 8],[3 4],[2],[4],[7 8],[6],[8]];
    # end;
    # if arg[13] == "dp2"
    #     nn=1; # number of nodes
    #     nl=2;   #number of leaves
    #     left=[[1]];
    #     right=[[2]];
    # end;
end;
################
#Census (all)
if arg[12] == "cen"
    #tree structure
    nn=4; # number of nodes
    nl=5;   #number of leaves
    left=[[1 2],[1],[3 4],[3]];
    right=[[3 4 5],[2],[5],[4]];
    # if arg[13] == "dp5"
        # nn=6; # number of nodes
        # nl=7;   #number of leaves
        # left=[[1 2 3],[1],[2],[4 5 6],[4],[5]];
        # right=[[4 5 6 7],[2 3],[3],[7],[5 6],[6]];
    # end;
    # if arg[13] == "dp6"
        # nn=12; # number of nodes
        # nl=13;   #number of leaves
        # left=[[1 2 3 4 5 6 7],[1 2 3 4 5],[1 2],[1],[3],[4],[6],[8 9 10 11],[8],[9],[10],[12]];
        # right=[[8 9 10 11 12 13],[6 7],[3 4 5],[2],[4 5],[5],[7],[12 13],[9 10 11],[10 11],[11],[13]];
    # end;
    # if arg[13] == "dp3"
        # nn=3; # number of nodes
        # nl=4;   #number of leaves
        # left=[[1 2],[1],[3]];
        # right=[[3 4],[2],[4]];
    # end;
    # if arg[13] == "dp4"
        # nn=7; # number of nodes
        # nl=8;   #number of leaves
        # left=[[1 2 3 4],[1 2],[1],[3],[5 6],[5],[7]];
        # right=[[5 6 7 8],[3 4],[2],[4],[7 8],[6],[8]];
    # end;
    # if arg[13] == "dp2"
        # nn=1; # number of nodes
        # nl=2;   #number of leaves
        # left=[[1]];
        # right=[[2]];
    # end;
end;
###############
#COMPAS
if arg[12] == "com"
    if arg[1] == "s1"
        nn=4; # number of nodes
        nl=5;   #number of leaves
        left=[[1],[2],[3],[4]];
        right=[[2,3,4,5],[3,4,5],[4,5],[5]];
    end
    if arg[1] != "s1"
        nn=3; # number of nodes
        nl=4;   #number of leaves
        left=[[1],[2],[3]];
        right=[[2,3,4],[3,4],[4]];
    end
    # if arg[13] == "dp5"
        # nn=6; # number of nodes
        # nl=7;   #number of leaves
        # left=[[1 2 3],[1],[2],[4 5 6],[4],[5]];
        # right=[[4 5 6 7],[2 3],[3],[7],[5 6],[6]];
    # end;
    # if arg[13] == "dp6"
        # nn=12; # number of nodes
        # nl=13;   #number of leaves
        # left=[[1 2 3 4 5 6 7],[1 2 3 4 5],[1 2],[1],[3],[4],[6],[8 9 10 11],[8],[9],[10],[12]];
        # right=[[8 9 10 11 12 13],[6 7],[3 4 5],[2],[4 5],[5],[7],[12 13],[9 10 11],[10 11],[11],[13]];
    # end;
    # if arg[13] == "dp3"
        # nn=3; # number of nodes
        # nl=4;   #number of leaves
        # left=[[1 2],[1],[3]];
        # right=[[3 4],[2],[4]];
    # end;
    # if arg[13] == "dp4"
        # nn=7; # number of nodes
        # nl=8;   #number of leaves
        # left=[[1 2 3 4],[1 2],[1],[3],[5 6],[5],[7]];
        # right=[[5 6 7 8],[3 4],[2],[4],[7 8],[6],[8]];
    # end;
    # if arg[13] == "dp2"
        # nn=1; # number of nodes
        # nl=2;   #number of leaves
        # left=[[1]];
        # right=[[2]];
    # end;
end;
####################################################################################
#counting number of data points with their protected feature being equal to xp and having similar label to data point j
####################################################################################
#Sina
# sum_nd = Dict()
# for j=1:N, xp in levels(tr[B])
#     sum_nd[j,xp] = 0;
#     # size(tr[(tr[B] .== xp) && (tr[:class] == xp) ,:],1)
#     for i=1:N
#         if tr[i,B]==xp
#             if tr[j,:class]==tr[i,:class]
#                 sum_nd[j,xp]=sum_nd[j,xp]+1;
#             end
#         end
#     end
# end
sum_nd = Dict()
for j=1:N, xp in levels(tr[B])
    y=tr[j,:class]
    sum_nd[j,xp]=size(tr[(tr[B] .== xp) .& (tr[:class] .== y) ,:],1)
end
####################################################################################
#The MIP model
####################################################################################
cart = Model(solver=GurobiSolver(MIPGap=parse(Int,ARGS[10])/100,TimeLimit=parse(Int,ARGS[9])));
M=100;
M2=100;
M3=100;
M4 =100;
ep=0.1;

@variable(cart, r[1:N,1:nl]>=0);
@variable(cart, x[1:N,1:nl],Bin);
@variable(cart, ac[1:nn,F_c],Bin);
@variable(cart, anc[1:nn,F_nc],Bin);
@variable(cart, b[1:nn]);
@variable(cart, gp[1:N,1:nn]>=0);
@variable(cart, gn[1:N,1:nn]>=0);
@variable(cart, w[1:N,1:nn],Bin);
@variable(cart, s[1:nn,f in F_c, k in levels(tr[f])],Bin);
@variable(cart, wc[1:N,1:nn],Bin);
@variable(cart, v[1:N,1:nl]);
@variable(cart, z[1:nl], Bin); # constant
if fair == 1
    @variable(cart, rp[1:N,1:nl]>=0); # to linearize the prediction in the penalty function
    @variable(cart, rpp[y=classes, xp in levels(tr[B]),j=1:N] >=0);#Sina # to linearize the absolute value in the penalty function
end;
############################################
#objective
############################################
if fair == 1
   @objective(cart, Min , (1/size(tr,1))*sum(r[i,l] for i = 1:N , l=1:nl) +
        fair*lambda*(sum(rpp[y,xp,j] for y=classes, xp in levels(tr[B]),j=1:N ) ) );#Sina
else
    @objective(cart, Min , (1/size(tr,1))*sum(r[i,l] for i = 1:N , l=1:nl));
end;
if fair == 1
    #Sina
    for y=classes, xp in levels(tr[B]),j=1:N
            if (sum_nd[j,xp]!= 0)
            @constraint(cart, sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N if (tr[i,B]==xp) && (tr[i,:class]==tr[j,:class]))/
                            sum_nd[j,xp]
                            - sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N if (tr[i,:class]==tr[j,:class]))/size(tr[(tr[:class] .== tr[j,:class]) ,:],1)<= rpp[y,xp,j]);
            @constraint(cart, -sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N if (tr[i,B]==xp) && (tr[i,:class]==tr[j,:class]))/
                            sum_nd[j,xp]
                            +sum(ind(sum(rp[i,l] for l=1:nl)==y) for i=1:N if (tr[i,:class]==tr[j,:class]))/size(tr[(tr[:class] .== tr[j,:class]) ,:],1)<= rpp[y,xp,j]);

            end
    end
##########################################################
##### linearize prediction in the penalty function
    for i in 1:N
        for l in 1:nl
            @constraint(cart,rp[i,l]<=M2*x[i,l]);
            @constraint(cart,rp[i,l]<=z[l]);
            @constraint(cart,rp[i,l]>=z[l]-M2*(1-x[i,l]));
        end
    end
end;
##############################################
# nodes constraints
for n in 1:nn
    @constraint(cart,sum(ac[n,f] for f in F_c)+sum(anc[n,f] for f in F_nc)==1);
end
#######################################################################
# categoricals
for n in 1:nn
    for f in F_c
        for k in levels(tr[f])
            #@constraint(cart,sl[n,f,k]+sr[n,f,k]==ac[n,f]);
            @constraint(cart,s[n,f,k] <= ac[n,f]);
        end
    end
end
for i in 1:N
    for n in 1:nn
        #for f in F_c
            #@constraint(cart,wl[i,n] <= sum(sl[n,f,k]*ind(tr[i,f]==levels(tr[f])[k]) for k=1:nk_f[f] )+ 1-ac[n,f]);
            #@constraint(cart,wr[i,n] <= sum(sr[n,f,k]*ind(tr[i,f]==levels(tr[f])[k]) for k=1:nk_f[f] )+ 1-ac[n,f]);
            @constraint(cart,wc[i,n] == sum(sum(s[n,f,k]*ind(tr[i,f]==k) for k in levels(tr[f])) for f in F_c) );
        #end
        for l in left[n]
            #@constraint(cart,x[i,l] <= wr[i,n]+1-sum(ac[n,f] for f in F_c));
            @constraint(cart,x[i,l] <= wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
        for l in right[n]
            #@constraint(cart,x[i,l]<=wl[i,n]+1-sum(ac[n,f] for f in F_c));
            @constraint(cart,x[i,l]<=1-wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
    end
end
############################################################################################
# non categoricals
for i in 1:N
    for n in 1:nn
        @constraint(cart,b[n]-sum(anc[n,f]*tr[i,f] for f in F_nc) == gp[i,n]-gn[i,n]);
        @constraint(cart,gp[i,n]<=M*w[i,n]);
        @constraint(cart,gn[i,n]<=M*(1-w[i,n]));
        @constraint(cart,gp[i,n]+gn[i,n]>=ep*(1-w[i,n])); #if b=criteria then w is 1 and go left
        for l in right[n]
            @constraint(cart,x[i,l]<=1-w[i,n]+1-sum(anc[n,f] for f in F_nc));
        end
        for l in left[n]
            @constraint(cart,x[i,l]<=w[i,n]+1-sum(anc[n,f] for f in F_nc));
        end
    end
end
##########################################################
##### linearize prediction error
for i in 1:N
    @constraint(cart,sum(x[i,l] for l=1:nl)==1);
    for l in 1:nl
        @constraint(cart,r[i,l]<=M2*x[i,l]);
        @constraint(cart,r[i,l]<=v[i,l]);
        @constraint(cart,r[i,l]>=v[i,l]-M2*(1-x[i,l]));
        @constraint(cart,v[i,l]>= tr[i,class]-z[l]); #CONSTANT
        @constraint(cart,v[i,l]>= -tr[i,class]+z[l]); #CONSTANT
    end
end
####################################################################################
#Solving Model
####################################################################################
status = solve(cart)
#################
#results
getvalue(z)
println("z=\t",getvalue(z))
getvalue(anc)
println("anc=\t",getvalue(anc))
getvalue(ac)
println("ac=\t",getvalue(ac))
getvalue(b)
println("b=\t",getvalue(b))
getvalue(s)
println("s=\t",getvalue(s))
#constant
z1 = round.(getvalue(z))
x1 = round.(getvalue(x));
########
# training accuracy
tr_acc = 0;
for l in 1:nl
    tr_acc = tr_acc + (z1[l]*(*(transpose(tr[:,class]),x1[:,l])) + (1-z1[l])*(*(transpose(-tr[:,class]+1),x1[:,l]))) / N;
end
println("tr_acc=\t",tr_acc)
# prediction on train
pred = x1*z1;
############################
#d function
temp_d_fnc = convert(Array, tr[:,setdiff(names(tr),[B])]);
half_range = (maximum(temp_d_fnc,1)-minimum(temp_d_fnc,1))/2;
temp_d_fnc = (temp_d_fnc-repmat(mean(temp_d_fnc,1),size(temp_d_fnc,1)))./half_range;
#######
k_nn = 10;
knn = Dict();
for i=1:N
    knn_dis = sqrt.(sum((repmat(transpose(temp_d_fnc[i,:]),N,1)-temp_d_fnc).^2,2));
    knn[i] = sortperm(knn_dis[:,1])[1:k_nn];
end
################################
# training disc
temp = tr;
disc = 0
M = 1;
for up=classes, Bj in levels(temp[B]) , Bk in levels(temp[B])
    if Bj != Bk && (size(temp[(temp[B] .== Bj) .& (temp[class] .== up),:],1) != 0) && (size(temp[(temp[B] .== Bk) .& (temp[class] .== up),:],1) != 0)
	denom = 0;
        M =1;
        for j=1:N, jp in knn[j]
            if (temp[j,B]==Bj) && (temp[jp,B]==Bk) && (temp[j,class]==up) && (temp[jp,class]==up)
                denom = denom + abs(pred[j]-pred[jp]);
                M = M+1;
            end
        end
        disc = disc + (denom/M);
    end
end
tr_disc=disc;
println("train_disc=\t ", tr_disc)

####################################
#Sina
#training DTDI index
DT_disc=0
for y=classes ,xp in levels(tr[B]), j=1:N
        if ( count((tr[i,B]==xp) for i in knn[j])!= 0)
            DT_disc=DT_disc+abs((count((tr[i,B]==xp) && (pred[i]==y) for i in knn[j] ))/(count((tr[i,B]==xp) for i in knn[j]))-
                                (count((pred[i]==y) for i in knn[j]) )/k_nn)
        end
end
tr_DTDI=DT_disc;
tr_DTDI_norm=tr_DTDI/size(levels(tr[B]),1)/size(levels(tr[:class]),1)/N;
println("train_DTDI=\t ", tr_DTDI)
println("train_DTDI_norm=\t ", tr_DTDI_norm)
################################################################################################
#test
################################################################################################
# test
# to find the prediction fit the parameters and then find the feasible solution
ac = getvalue(ac);
anc = getvalue(anc);
b = getvalue(b);
s = getvalue(s);
z = round.(getvalue(z));
########
for n in 1:nn, f in F_c,
    ac[n,f] = round.(ac[n,f])
    for  k in levels(tr[f])
        s[n,f,k] = round.(s[n,f,k])
    end
end
for n in 1:nn, f in F_nc,
    anc[n,f] = round.(anc[n,f])
end
########################
N = size(tes,1) # make sure N is size of the training
####################################################################################
cart_pred = Model(solver=GurobiSolver(MIPGap=1,TimeLimit=3600));
M=100;
M2=100;
M3=100;
M4=100;
ep=0.1;
##########
@variable(cart_pred, x[1:N,1:nl],Bin);
@variable(cart_pred, gp[1:N,1:nn]>=0);
@variable(cart_pred, gn[1:N,1:nn]>=0);
@variable(cart_pred, w[1:N,1:nn],Bin);
@variable(cart_pred, wc[1:N,1:nn],Bin);
############
for i in 1:N
    for n in 1:nn
            @constraint(cart_pred,wc[i,n] == sum(sum(s[n,f,k]*ind(tes[i,f]==k) for k in levels(tes[f])) for f in F_c) );
        for l in left[n]
            @constraint(cart_pred,x[i,l] <= wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
        for l in right[n]
            @constraint(cart_pred,x[i,l]<=1-wc[i,n]+1-sum(ac[n,f] for f in F_c));
        end
    end
end
############################################################################################
# non categoricals
for i in 1:N
    for n in 1:nn
        @constraint(cart_pred,b[n]-sum(anc[n,f]*tes[i,f] for f in F_nc) == gp[i,n]-gn[i,n]);
        @constraint(cart_pred,gp[i,n]<=M*w[i,n]);
        @constraint(cart_pred,gn[i,n]<=M*(1-w[i,n]));
        @constraint(cart_pred,gp[i,n]+gn[i,n]>=ep*(1-w[i,n])); #if b=criteria then w is 1 and go left
        for l in right[n]
            @constraint(cart_pred,x[i,l]<=1-w[i,n]+1-sum(anc[n,f] for f in F_nc));
        end
        for l in left[n]
            @constraint(cart_pred,x[i,l]<=w[i,n]+1-sum(anc[n,f] for f in F_nc));
        end
    end
end
for i in 1:N
     @constraint(cart_pred,sum(x[i,l] for l=1:nl)==1);
end
###################
status = solve(cart_pred)
x1 = round.(getvalue(x));
########
# test accuracy
tes_acc = 0;
for l in 1:nl
    tes_acc = tes_acc + (z1[l]*(*(transpose(tes[:,class]),x1[:,l])) + (1-z1[l])*(*(transpose(-tes[:,class]+1),x1[:,l]))) / N;
end
println("tes_acc=\t",tes_acc)
#############
temp_d_fnc = convert(Array, tes[:,setdiff(names(tes),[B])]);
half_range = (maximum(temp_d_fnc,1)-minimum(temp_d_fnc,1))/2;
temp_d_fnc = (temp_d_fnc-repmat(mean(temp_d_fnc,1),size(temp_d_fnc,1)))./half_range;
for i=1:N
    knn_dis = sqrt.(sum((repmat(transpose(temp_d_fnc[i,:]),N,1)-temp_d_fnc).^2,2));
    knn[i] = sortperm(knn_dis[:,1])[1:20];
end
############
# prediction on test
pred_tes = x1*z1;
########
# test disc
temp = tes;
disc = 0
for up=classes, Bj in levels(temp[B]) , Bk in levels(temp[B])
    if Bj != Bk && (size(temp[(temp[B] .== Bj) .& (temp[class] .== up),:],1) != 0) && (size(temp[(temp[B] .== Bk) .& (temp[class] .== up),:],1) != 0)
	denom = 0;
        M = 1;
        for j=1:N, jp in knn[j]
            if (temp[j,B]==Bj) && (temp[jp,B]==Bk) && (temp[j,class]==up) && (temp[jp,class]==up)
                denom = denom + abs(pred_tes[j]-pred_tes[jp]);
                M = M+1
            end
        end
        disc = disc + (denom/M);
    end
end
tes_disc=disc;
println("tes_disc=\t",tes_disc)
############
#Sina
#test DTDI index
DT_disc=0
for y=classes ,xp in levels(tes[B]), j=1:N
        if ( count((tes[i,B]==xp) for i in knn[j])!= 0)
            DT_disc=DT_disc+abs((count((tes[i,B]==xp) && (pred_tes[i]==y) for i in knn[j] ))/(count((tes[i,B]==xp) for i in knn[j]))-
                                (count((pred_tes[i]==y) for i in knn[j]) )/k_nn)
        end
end
tes_DTDI=DT_disc;
tes_DTDI_norm=tes_DTDI/size(levels(tes[B]),1)/size(levels(tes[:class]),1)/N;
println("test_DTDI=\t ", tes_DTDI)
println("test_DTDI_norm=\t ", tes_DTDI_norm)
############

# save the prediction defalt sata set
if arg[12] == "df"
    gap = abs( cart.objBound - cart.objVal )/abs(cart.objVal);
    jld_name = join(["df_ap1",arg[1],string("lam",lambda)],"_","_")
    jld_name = join([jld_name,"jld"],".")
    save(jld_name,
        "pred_on_train", pred,
        "pred_on_test", pred_tes,
        "objVal", cart.objVal,
        "optimality gap" , gap,
        "disc_test", disc,
        "acc_test", tes_acc,
        "lambda" , lambda,
        "tr_DTDI",tr_DTDI,
        "tr_DTDI_norm",tr_DTDI_norm,
        "tes_DTDI",tes_DTDI,
        "tes_DTDI_norm",tes_DTDI_norm)
end

# save the prediction census sata set
if arg[12] == "cen"
    #Census
    # save the prediction s1 defalt sata set
    gap = abs( cart.objBound - cart.objVal )/abs(cart.objVal);
    jld_name = join(["census_ap1",arg[1],string("lam",lambda)],"_","_")
    jld_name = join([jld_name,"jld"],".")
    save(jld_name,
        "pred_on_train", pred,
        "pred_on_test", pred_tes,
        "objVal", cart.objVal,
        "optimality gap" , gap,
        "disc_test", disc,
        "acc_test", tes_acc,
        "lambda" , lambda,
        "tr_DTDI",tr_DTDI,
        "tr_DTDI_norm",tr_DTDI_norm,
        "tes_DTDI",tes_DTDI,
        "tes_DTDI_norm",tes_DTDI_norm)
end

# save the prediction COMPAS sata set
if arg[12] == "com"
    gap = abs( cart.objBound - cart.objVal )/abs(cart.objVal);
    jld_name = join(["com_ap1",arg[1],string("lam",lambda)],"_","_")
    jld_name = join([jld_name,"jld"],".")
    save(jld_name,
        "pred_on_train", pred,
        "pred_on_test", pred_tes,
        "objVal", cart.objVal,
        "optimality gap" , gap,
        "disc_test", disc,
        "acc_test", tes_acc,
        "lambda" , lambda,
        "tr_DTDI",tr_DTDI,
        "tr_DTDI_norm",tr_DTDI_norm,
        "tes_DTDI",tes_DTDI,
        "tes_DTDI_norm",tes_DTDI_norm)
end


#Sina
####################################################################################
#Saving results
####################################################################################
filePath = "class_app1_DT_results.csv";
# header=["app" "sample" "fair" "lambda" "depth" "time_lim" "tr_acc" "tr_disc" "tr_DTDI" "tr_DTDI_norm" "tes_acc" "tes_disc" "tes_DTDI" "tes_DTDI_norm"]
# file = open(filePath, "a");
# writecsv(file,header );
# close(file);

#printing results into a csv file
# header=["app" "sample" "fair" "lambda" "depth" "time_lim" "tr_acc" "tr_disc" "tr_DTDI" "tr_DTDI_norm" "tes_acc" "tes_disc" "tes_DTDI" "tes_DTDI_norm"]
row_input=["app1" arg[1] fair lambda depth time_lim tr_acc tr_disc tr_DTDI tr_DTDI_norm tes_acc tes_disc tes_DTDI tes_DTDI_norm];
file = open(filePath, "a");
writecsv(file,row_input);
close(file);
