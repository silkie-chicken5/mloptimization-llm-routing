"""Sweep lambda and record quality, dollars and CO2 of each method's test-set picks, for the tradeoff plots."""

# ------------------- Setup -------------------
# run with julia +1.11.6/IAI/3.2.2 --project=. sweep_lambda.jl
# Lambdas come from [sweep] in config.toml and are fixed inputs (not tuned).
# For each lambda, the trees' depth/minbucket are tuned on val exactly as in policy_trees.jl.
using CSV, DataFrames, TOML, Statistics, CategoricalArrays

root = @__DIR__
include(joinpath(root, "baselines.jl"))
cfg = TOML.parsefile(joinpath(root, "config.toml"))

prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
pairs = CSV.read(joinpath(root, "data", "processed", "pairs.csv"), DataFrame;
                 select = [:sample_id, :model, :quality, :cost, :wh, :carbon_cost])

train, val = prompts.split .== "train", prompts.split .== "val"
test_rows = findall(prompts.split .== "test")

X = select(prompts, Not([:sample_id, :split]))
X.task = categorical(X.task)

# prompts x models outcome matrices, rows in prompts.csv order
pairs.model_short = short_name.(pairs.model)
model_names = sort(unique(pairs.model_short))
quality_of = outcome_matrix(pairs, prompts.sample_id, :quality, model_names)
dollars_of = outcome_matrix(pairs, prompts.sample_id, :cost, model_names)
carbon_of  = outcome_matrix(pairs, prompts.sample_id, :carbon_cost, model_names)
wh_of      = outcome_matrix(pairs, prompts.sample_id, :wh, model_names)

# Same reward as process_data.py: quality - lambda_dollar * dollars - lambda_carbon * carbon
reward_table(ld, lc) = DataFrame(quality_of .- ld .* dollars_of .- lc .* carbon_of, model_names)

# Quality, $ per 1k queries and kg CO2 per 1k queries of a set of test-row picks
function outcomes(picks)
    wh = picked_mean(wh_of, test_rows, picks, model_names)
    return (quality = picked_mean(quality_of, test_rows, picks, model_names),
            dollars_per_1k = 1000 * picked_mean(dollars_of, test_rows, picks, model_names),
            kg_co2_per_1k = wh / 1000 * cfg["grid_intensity"] * 1000)  # Wh -> kWh -> kg CO2, per 1,000 queries
end

rows = DataFrame()

# Reference points that do not depend on lambda: always sending every prompt to one model
for m in model_names
    push!(rows, (sweep = "reference", lambda_dollar = missing, lambda_carbon = missing,
                 method = "always_model", model = m, depth = missing, minbucket = missing,
                 test_reward = missing, outcomes(fill(m, length(test_rows)))...); promote = true)
end

sweeps = vcat([("joint", l, l) for l in cfg["sweep"]["joint"]],
              [("carbon_only", 0, l) for l in cfg["sweep"]["carbon_only"]])

for (sweep, ld, lc) in sweeps
    println("$sweep: lambda_dollar = $ld, lambda_carbon = $lc")
    R = reward_table(ld, lc)
    test_reward(picks) = picked_mean(Matrix(R), test_rows, picks, model_names)

    opt = tune_tree(X, R, train, val, cfg; localsearch = true, verbose = false)
    greedy = tune_tree(X, R, train, val, cfg; localsearch = false, depths = [opt.depth], verbose = false)
    best_single = best_single_model(R[train, :])
    oracle = [model_names[argmax(collect(R[r, :]))] for r in test_rows]  # best model per test prompt, in hindsight

    for (method, model, depth, minbucket, picks) in [
            ("optimal_tree", "", opt.depth, opt.minbucket, tree_picks(opt.lnr, X[test_rows, :])),
            ("greedy_tree", "", greedy.depth, greedy.minbucket, tree_picks(greedy.lnr, X[test_rows, :])),
            ("best_single", best_single, missing, missing, fill(best_single, length(test_rows))),
            ("oracle", "", missing, missing, oracle)]
        push!(rows, (sweep = sweep, lambda_dollar = ld, lambda_carbon = lc, method = method, model = model,
                     depth = depth, minbucket = minbucket, test_reward = test_reward(picks),
                     outcomes(picks)...); promote = true)
    end

    # sanity check: tree picks reproduce IAI's own test reward
    @assert abs(test_reward(tree_picks(opt.lnr, X[test_rows, :])) - avg_reward(opt.lnr, X[test_rows, :], R[test_rows, :])) < 1e-9
end

CSV.write(joinpath(root, "results", "lambda_sweep.csv"), rows)
show(transform(rows, names(rows, Union{Missing, Float64}) .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()
