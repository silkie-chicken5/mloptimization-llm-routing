"""Make baselines to compare policy trees against + helper methods to score policy trees and baselines."""

using DataFrames, Statistics

# remove slashes to match the reward column names
short_name(model) = String(split(model, "/")[end])

# -------- Reward calculation -------
# Upper bound: pick the model with the highest actual reward for each prompt (infeasible and perfect)
oracle_reward(R) = mean(maximum.(eachrow(Matrix(R))))

# Mean reward of a fitted tree's picks (reward associated with the model picked by the tree per prompt)
avg_reward(lnr, X, R) = mean(IAI.predict_outcomes(lnr, X, R))

# Mean reward of sending every prompt to one fixed model
fixed_model_reward(R, model) = mean(R[!, model])

# ------- Model picking -------
# Model with the highest mean reward on the training rows
best_single_model(R_train) = argmax(name -> mean(R_train[!, name]), names(R_train))

# Model with the lowest mean dollar + carbon cost on the training prompts
# This does not depend on lambda, so it is the same model in every scenario.
function cheapest_model(pairs, train_ids)
    ids = Set(train_ids)
    p = filter(row -> row.sample_id in ids, pairs)
    costs = combine(groupby(p, :model), [:cost, :carbon_cost] => ((c, cc) -> mean(c .+ cc)) => :total_cost)
    return short_name(costs.model[argmin(costs.total_cost)])
end


# ------- Outcome lookups (for quality / dollars / CO2 of each method's picks) -------
# One pairs column as a prompts x models matrix, rows in the same order as prompts.csv
function outcome_matrix(pairs, prompt_ids, column, model_names)
    wide = unstack(pairs, :sample_id, :model_short, column)
    wide = leftjoin(DataFrame(sample_id = prompt_ids), wide, on = :sample_id)
    M = Matrix(wide[:, model_names])
    @assert !any(ismissing, M) "missing outcomes for column $column"
    return Float64.(M)
end

# Average of one outcome when each prompt goes to the model that was picked for it
function picked_mean(outcomes, rows, picks, model_names)
    col = Dict(name => j for (j, name) in enumerate(model_names))
    return mean(outcomes[row, col[model]] for (row, model) in zip(rows, picks))
end

# Models a fitted tree recommends for some rows (predict may return a tuple; picks come first)
function tree_picks(lnr, X)
    p = IAI.predict(lnr, X)
    p = p isa Tuple ? p[1] : p
    return String.(p)
end
