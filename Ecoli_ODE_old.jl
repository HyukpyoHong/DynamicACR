## Initialization
#pakages

using Pkg
using CSV
using DataFrames
using DifferentialEquations
using Interpolations
using Plots

# stoi_mat = CSV.read("/Users/hyukpyohong/Dropbox/CRN_model_reduction/Code/stoi-ex2.csv", DataFrame, header = 0)
# source_mat = CSV.read("/Users/hyukpyohong/Dropbox/CRN_model_reduction/Code/source-ex2.csv", DataFrame, header = 0)

stoi_mat = [1 -1 0; 0 1 -1]
source_mat = [0 1 0; 0 0 1]

function ex1(dx, x, kappa, t)
  # f: vector of the rate functions WITHOUT rate constants.
  # kappa: vector of the rate constants
  f = zeros(length(kappa), 1);
  for i in 1:length(kappa)
    f[i] = prod(x .^ source_mat[:, i])
  end
  # f = [1, x[1], x[2]];
  r = kappa .* f; #r: vector of the rate functions WITH rate constants.
  dx_vec = stoi_mat * r
  # dx = dx_vec : does not work
  # dx = copy(dx_vec): does not work
  # dx[1] = dx_vec[1]
  # dx[2] = dx_vec[2]
  for i in 1:length(x)
    dx[i] = dx_vec[i];
  end
end

kappa1 = [10,2,4];
x_init = [0,0];
tspan1 = (0.0, 20.0);
# tspan = [0.0, 10.0]

prob1 = ODEProblem(ex1, x_init, tspan1, kappa1);
sol1 = solve(prob1, Vern9());
sol_mat1 = reduce(hcat,sol1.u)';
A = sol_mat1[:,1]
B = sol_mat1[:,2]
plot(sol1.t, [A B], label = ["A" "B"])
# plot(sol1, label = "")
