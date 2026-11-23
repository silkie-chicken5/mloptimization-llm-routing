# Baselines that the policy trees are compared against.
# Every function takes a reward table R (one row per prompt, one column per model)
# that has already been subset to the rows of one split (train, val, or test).

using DataFrames, Statistics

# "mistralai/mistral-7b-chat" -> "mistral-7b-chat", to match the reward column names
short_name(model) = String(split(model, "/")[end])

# Upper bound: pick the model with the highest actual reward for each prompt
oracle_reward(R) = mean(maximum.(eachrow(Matrix(R))))

# Model with the highest mean reward on the given rows (pass the training rows)
best_single_model(R_train) = argmax(name -> mean(R_train[!, name]), names(R_train))

# Model with the lowest mean dollar + carbon cost on the given prompts (pass the training ids).
# This does not depend on lambda, so it is the same model in every scenario.
function cheapest_model(pairs, train_ids)
    ids = Set(train_ids)
    p = filter(row -> row.sample_id in ids, pairs)
    costs = combine(groupby(p, :model), [:cost, :carbon_cost] => ((c, cc) -> mean(c .+ cc)) => :total_cost)
    return short_name(costs.model[argmin(costs.total_cost)])
end

# Mean reward of sending every prompt to one fixed model
fixed_model_reward(R, model) = mean(R[!, model])
