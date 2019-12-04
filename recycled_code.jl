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