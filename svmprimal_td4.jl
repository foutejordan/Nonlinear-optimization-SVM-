using Ipopt
using JuMP
using Pandas

using DelimitedFiles
#question 1, lire le fichier csv

function init(filename,delim)
	res = readdlm(filename,delim)
	N = size(res)[1]
	n = size(res)[2]

	X = zeros(N, n-1)
	T  = zeros(N)

	for i = 1:N
		for j = 1:n-1
			X[i,j] = res[i,j]
		end
		T[i] = res[i,n]
	end		
	return([X,T])
end

data = init("train_200_40.csv", ',')
#println(data)

# Le dossier contient l'ensemble des corpus pour permettre de tester facilement
# 
#1) Cas Lineairement separables
function svmprimal()

    #model

    x = data[1]
    t = data[2]

    m = Model(Ipopt.Optimizer)
  
    N = size(x)[1]
    n = size(x)[2]
    
    @variable(m, w[1:n])
    @variable(m, b)
    
    #fonction objective
    @NLobjective(m, Min, sum(w[i]^2 for i=1:n))

    #contraintes
    @constraint(m,[i=1:N], t[i] * (sum(w[j]*x[i,j] for j=1:n) + b) >= 1 )

    status = optimize!(m)
    objective_value(m)

    return value.(w), value(b)

end

result_params = svmprimal()

#Split du dataset en donnees de test et de train
function split(filename,delim)
    train = open("train_200_0_quad.csv", "w")
    test = open("test_200_0_quad.csv", "w")
    cpt = 0
    open(filename, "r") do f
        for i=1:150
            s = readline(f)
            write(train,"$s\n")
        end
        for i=151:199
            s = readline(f)
            write(test,"$s\n")
        end
    end
    close(train)
    close(test)
end

split("shuffled_data200_0_quad.csv", "")

# Fonction de test et calcul de l'accuracy
function test(w,b,M)
    x = M[1]
    t = M[2]
    
    N = size(x)[1]
    n = size(x)[2]
    y = zeros(N)
    e = 0
    for i=1:N
        y[i] = sum(w[j] * x[i,j] for j=1:n) + b
        println(y[i])
        if(y[i] > 0)
            e += (1 - t[i]) /2
        else
            e += (1 + t[i])/2
        end
    end
    accuracy = (e / N) * 100
    println(accuracy)
end

M = init("test_200_40.csv", ',')
w = result_params[1]


b = result_params[2]

#Decommenter la fonction test() suivante pour pouvoir tester le cas lineairement separable
#test(w, b, M)


# Cas Non linéairement séparable..

function svmdual(data_D)
    x = data_D[1]
    t = data_D[2]

    m = Model(Ipopt.Optimizer)
  
    N = size(x)[1]
    n = size(x)[2]

    e = [1 for i=1:N]

    Q = zeros(N, N)


    for i=1:N
        for j = 1:N
            Q[i,j] =  t[i] * t[j] * (sum(x[i,k] * x[j,k] for k=1:n) + 0.5).^2     
        end
    end
    println(Q)

 
    @variable(m, λ[1:N] >= 0)
    #sum(e[i] * λ[i] for i=1:N)
    #fonction objective
   @expression(m, expr, e' * λ  - 0.5 * λ' * Q * λ)
   
   @NLobjective(m, Max, expr)

    #contraintes
    #
    @constraint(m, sum(λ[j]*t[j] for j=1:N) == 0 )

    status = optimize!(m)
    objective_value(m)

    lamda = value.(λ)
    # calcul de b
    S = length(λ)
    println(S)
    b = 1/S * sum(t[i] - sum(lamda[j] * t[i] * (sum(x[i,k] * x[j,k] for k=1:n) + 0.5).^2  for j=1:N) for i=1:N)
    
    println(b)
    return lamda,b
end

result_dual = svmdual(init("train_200_0_quad.csv", ','))

λ = result_dual[1]
b = result_dual[2]    

#println(λ)
#test de prediction et accuracy

function prediction(data_D)

    x = data_D[1]
    t = data_D[2]

    N = size(x)[1]
    n = size(x)[2]
    y = zeros(N)
    e = 0

    for i=1:N
        y[i] = 0
        for j=1:N 
            y[i] += λ[i] * t[i] *  (sum(x[i,k] * x[j,k] for k=1:n) + 0.5).^2
        end
        y[i] += b
        if(y[i] > 0)
            e += (1 - t[i]) /2
        else
            e += (1 + t[i])/2
        end
    end
    accuracy = (e / N) * 100
    println(accuracy)

end

#Decommenter la fonction prediction suivante pour tester le cas non lineaire
#prediction(init("test_200_0_quad.csv", ','))







