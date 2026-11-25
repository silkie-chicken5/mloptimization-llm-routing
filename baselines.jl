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


# ------- Tree tuning -------
# Grid search over max_depth and minbucket (values from config.toml): fit on train & keep tree w best val reward.
function tune_tree(X, R, train, val, cfg; localsearch, depths = cfg["tree"]["max_depth"], verbose = true)
    """
    X: the data
    R: the reward matrix
    train, val: row masks
    localsearch: true gives the optimal tree, false grows it greedy CART tree
    depths: defaults to config file max depth grid
    """
    best = (score = -Inf, lnr = nothing, depth = 0, minbucket = 0) # best tree so far
    for depth in depths, minbucket in cfg["tree"]["minbucket"]
        lnr = IAI.OptimalTreePolicyMaximizer(
            max_depth = depth,
            minbucket = minbucket,
            cp = 0, # control max_depth and minbucket, so alpha/cp is useless
            localsearch = localsearch,
            random_seed = cfg["seed"],
        )
        IAI.fit!(lnr, X[train, :], R[train, :])

        val_score = avg_reward(lnr, X[val, :], R[val, :])
        if verbose
            train_score = avg_reward(lnr, X[train, :], R[train, :])
            println("  depth=$depth minbucket=$minbucket  train=$(round(train_score, digits=3))  val=$(round(val_score, digits=3))")
        end

        if val_score > best.score # keeps best tree
            best = (score = val_score, lnr = lnr, depth = depth, minbucket = minbucket)
        end
    end
    return best
end
