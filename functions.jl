"""
    standardize!(X::AbstractMatrix)

Standardisation du vecteur colonne X ou de chacune des colonnes de la matrice X.

### Arguments
- `X::Matrix{Real}` : Vecteur ou matrice à standardiser.

### Détails

La fonction centre la matrice et modifie l'échelle de l'argument X.

### Exemples

\```
 julia> standardize!(X)
\```

"""
function standardize!(x::Vector{Float64})
    
    x̄ = mean(x)
    s = std(x)
    
    for i=1:length(x)
        x[i] = (x[i] - x̄)/s
    end
    
    return x
    
end

function standardize!(X::Matrix{Float64})
    
    for j=1:size(X,2)
       X[:,j] = standardize!(X[:,j])
    end
    
    return X
    
end

"""
    splitdataframe(df::DataFrame, p::Real)

Partitionne en un ensemble d'entraînement et un ensemble de validation un DataFrame.

### Arguments
- `df::DataFrame` : Un DataFrame
- `p::Real` : La proportion (entre 0 et 1) de données dans l'ensemble d'entraînement.

### Détails

La fonction renvoie deux DataFrames, un pour l'ensemble d'entraînement et l'autre pour l'ensemble de validation.

### Exemple

\```
 julia> splitdataframe(df, p.7)
\```

"""
function splitdataframe(df::DataFrame, p::Real)
   @assert 0 <= p <= 1 
    
    n = size(df,1)
    
    ind = shuffle(1:n)
    
    threshold = Int64(round(n*p))
    
    indTrain = sort(ind[1:threshold])
    
    indTest = setdiff(1:n,indTrain)
    
    dfTrain = df[indTrain,:]
    dfTest = df[indTest,:]
    
    return dfTrain, dfTest
    
end

"""
    CoeffRidge(data::DataFrame, y::Symbol, colonnesExplicatives:: Array{Any,1})

Calcule et retourne le meilleur estimateur Beta pour un y et des variables explicatives données.

@params
data: ensemble de données
y: variable à prédire
colonnesExplicatives: variables explicatives

B_meilleurs: estimateur Beta donnant le meilleur résultat

"""

function CoeffRidge(data::DataFrame, y::Symbol, colonnesExplicatives::Array{Any, 1})
    train, test = splitdataframe(data, .8)

    ỹ = convert(Vector{Float64}, test[:, y])
    X̃ = convert(Matrix{Float64}, test[:, filter(x -> (x in colonnesExplicatives), names(data))])

    y = convert(Vector{Float64}, train[:, y])
    X = convert(Matrix{Float64}, train[:, filter(x -> (x in colonnesExplicatives), names(data))])

    standardize!(X̃)
    standardize!(X)
    standardize!(ỹ)
    standardize!(y)

    λ = 0:1:100

    β_meilleurs = []
    λ_meilleur = 0
    RMSE_meilleur = 9999999999999

    for i in 1:length(λ)
        β_r = (X'X + λ[i] * I)\X'y
        ŷ = X̃*β_r
        ẽ = ỹ - ŷ
        RMSE = sqrt(dot(ẽ,ẽ)/length(ẽ))
        if RMSE < RMSE_meilleur
            RMSE_meilleur = RMSE
            β_meilleurs = β_r
            λ_meilleur = i
        end
    end

    return β_meilleurs

end