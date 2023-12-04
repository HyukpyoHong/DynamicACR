using DifferentialEquations
using Plots

function simul_dif(dc_dt, c, Q, t) # Here, check whether the order of the arguments works or not. 
  # Define the differential equations
  # dc_dt = zeros(length(c), 1)
  dc_dt[1] = Q * R * c[2] - c[1] * c[3]
  dc_dt[2] = -Q * c[2] + c[1] * c[3]
  dc_dt[3] = -c[1] * c[3]
  #return dc_dt
end

simul_dif
methods(simul_dif)
# Define the initial conditions
Cin = [1; 1; 1]

# plot([1,2], [1,2])

# Define the time range
trange = (0.0, 10.0)

# Define the parameter vector
Q = 1.0
R = 2

# Solve the differential equations
prob = ODEProblem(simul_dif, Cin, trange, Q)
#tspan = (0.0, 10.0)
sol2 = solve(prob)

# Extract the time and concentration values from the solution
t = sol.t
c = sol.u
# Plot the results
# plot(sol)
#plot([1,2], [2,3])


exit()
function example_function(x; y, z)
    println("x:$x, y: $y, z: $z")
end

# Call the function with keyword arguments
example_function(10, z=5)
methods(example_function)

x = nothing
y = 0 
for i in 1:5
  x = 2*i
  y = fill(0, 2,3)
  println(x)
  println(y)
end
println(x)
println(y)
