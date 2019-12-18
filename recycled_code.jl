##################################################################
#### VÉRIFIER SI IL PEUT Y AVOIR SURVERSE LORSDES JOURS SANS PLUIE
##################################################################
dates_with_zero_mm = []

for i=1:length(X_pcp_sum[:,1])
    if all(x -> x == 0, X_pcp_sum[i, 2:6])
        push!(dates_with_zero_mm, X_pcp_sum[i, 1])
    end
end

surv_to_check = surverse_ahuntsic
df_tt = DataFrame(DATE = surv_to_check[:, 1], SURVERSE = surv_to_check[:, 2], PLUIE = ones(length(surv_to_check[:, 2])))

for i=1:length(df_tt[:,1])
    if df_tt[i, 1] in dates_with_zero_mm
        df_tt[i, 3] = 0 
        
        if df_tt[i, 2] == 1
            print(df_tt[i, :])
        end
    end
    
end
plot(df_tt, x=:PLUIE, y=:SURVERSE, Geom.beeswarm)

df_zero_2019 = filter(x -> year(x) == 2019, dates_with_zero_mm)

####################################################################
#utilisation du score F1 pour calculer le meilleur seuil
seuils = LinRange(0, 1, 200)
meilleur_seuil = 0
meilleur_f1 = 0

for i = 1:length(seuils)
    Ŷ = zeros(Int64,length(test_set[:,1]))
    Ŷ[θ̂.>seuils[i]] .= 1;
    f1 = computeF1score(Ŷ, test_set[:, :SURVERSE])
    if f1 > meilleur_f1
        meilleur_f1 = f1
        meilleur_seuil = seuils[i]
    end
end

[meilleur_seuil meilleur_f1]


"""
getForestPCA permet d'obtenir pour chaque ouvrage un modèle de prédiction basé sur l'analyse en composante principale et 
sur la classification par forêts aléatoires. Les paramètres pca_fit_params et rf_fit_params correspondent aux valeurs 
possibles des hyperparamètres des modèles PCA et RandomForestClassifier que nous utilisons.

@params
num_ouvrage:      le numéro de l'ouvrage à évaluer
pca_fit_param:    dictionnaire des valeurs possibles pour les hyperparamètres du modèle PCA
rf_fit_param:     dictionnaire des valeurs possibles pour les hyperparamètres du modèle RF
n_folds:          le nombre de plis à effectuer sur l'ensemble d'entrainement lors de la validation croisée
grid:             false par défaut, si à true, la fonction effectue une GridSearch (exhaustive) sur l'ensemble des
                  hyperparamètres fournis, sinon elle effectue une RandomizedSearch
n_iter:           10 par défaut, le nombre de modèles à essayer lors d'une RandomizedSearch

"""


function getForestPCA(num_ouvrage::String; pca_fit_params::Dict, rf_fit_params::Dict, n_folds = 5, grid=false, n_iter = 10, verbose=1)
    train = CreateDataframe(num_ouvrage, surverse_df)
    y = train[!, :SURVERSE]
    X = convert(Matrix{Float64}, train[:, 4:28])
    
    # Analyse en composantes principales.
    # Le nombre de composantes principales est également testé dans la GridSearch/RandomizedSearch
    pca = PCA()
    fit!(pca, X)

    # Préparation de la classification par RandomForests.
    # Les HyperParamètres seront évalués dans la GridSearch/RandomizedSearch
    rf = RandomForestClassifier()
    
    # Préparation du Pipeline à tester.
    # Le Pipeline permet de lier l'obtention des composantes principales à la création d'une RandomForest les utilisant.
    pipe = Pipelines.Pipeline([("pca", pca), ("rf", rf)])
    
    n_models = 1

    fit_params = Dict()
    for (k,v) in pca_fit_params
        fit_params["pca__$(k)"] = v
        n_models *= length(v)
    end
    
    for (k,v) in rf_fit_params
        fit_params["rf__$(k)"] = v
        n_models *= length(v)
    end

    println("Nombre de modèles à parcourir: $(n_models)\n")

    if grid
        M = GridSearch.GridSearchCV(estimator=pipe,
                                    param_grid=fit_params,
                                    cv=n_folds, 
                                    verbose=verbose)
    else
        M = GridSearch.RandomizedSearchCV(estimator=pipe,
                                          param_distributions=fit_params,
                                          cv=n_folds,
                                          n_iter=n_iter, 
                                          verbose=verbose)
    end

    return fit!(M, X, y)
end

ouvrage = ["3260-01D", "3350-07D", "4240-01D", "4350-01D", "4380-01D"]
best_models = Dict()

#Préparation d'un dictionnaire de paramètres pour l'optimisation des hyperparamètres du modèle
#On déclare les valeurs jugées possibles pour le modèle

pca_fit_params = Dict()
pca_fit_params["n_components"] = convert(Array{Int64}, 13)

rf_fit_params = Dict()
rf_fit_params["n_subfeatures"] = convert(Array{Int64}, 3:1:5)#sans pca, 3:1:10
rf_fit_params["n_trees"] = convert(Array{Int64}, 80:5:200)#300
#rf_fit_params["partial_sampling"] = [0.6, 0.7, 0.8]
rf_fit_params["max_depth"] = convert(Array{Int64}, 4:1:20)#50
#rf_fit_params["min_samples_leaf"] = convert(Array{Int64}, 5:2:10)
#rf_fit_params["min_samples_split"] = convert(Array{Int64}, 2:2:16)
#rf_fit_params["min_purity_increase"] = [0.0, 0.1, 0.2]

for i=1:length(ouvrage)
    @time best_models[ouvrage[i]] = getForestPCA(ouvrage[i],
                                                 pca_fit_params = pca_fit_params,
                                                 rf_fit_params = rf_fit_params,
                                                 n_iter=1000)
    println(ouvrage[i], " done")
    println(best_models[ouvrage[i]].best_score_, "\n")
end
