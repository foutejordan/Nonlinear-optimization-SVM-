####################################
# Auteur : S. Gueye
# Exemple de lecture d'un csv en julia
####################################

using DelimitedFiles

####################################
# filename : nom du fichier csv à lire
# delim : délimiteur dans le fichier csv
####################################
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


