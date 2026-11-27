"""Sweep lambda and record quality, dollars and CO2 of each method's picks on val and test, for the tradeoff plots."""

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

# Rows each split's metrics are reported on.
# val: use these curves to CHOOSE lambdas. test: report results for lambdas chosen ahead of time.
eval_rows = Dict("val" => findall(val), "test" => test_rows)

# Quality, $ per 1k queries and kg CO2 per 1k queries of a set of picks for the given rows
function outcomes(picks, rows)
    wh = picked_mean(wh_of, rows, picks, model_names)
    return (quality = picked_mean(quality_of, rows, picks, model_names),
            dollars_per_1k = 1000 * picked_mean(dollars_of, rows, picks, model_names),
            kg_co2_per_1k = wh / 1000 * cfg["grid_intensity"] * 1000)  # Wh -> kWh -> kg CO2, per 1,000 queries
end

# Named scenario (from [lambdas] in config.toml) at a given lambda pair, or "" if none
scenario_at(ld, lc) = something(findfirst(l -> l["dollar"] == ld && l["carbon"] == lc, cfg["lambdas"]), "")

out = DataFrame()

# Reference points that do not depend on lambda: always sending every prompt to one model
for (split, rows) in eval_rows, m in model_names
    push!(out, (split = split, sweep = "reference", scenario = "", lambda_dollar = missing, lambda_carbon = missing,
                method = "always_model", model = m, depth = missing, minbucket = missing,
                reward = missing, outcomes(fill(m, length(rows)), rows)...); promote = true)
end

sweeps = vcat([("joint", l, l) for l in cfg["sweep"]["joint"]],
              [("carbon_only", 0, l) for l in cfg["sweep"]["carbon_only"]])
# named scenarios that fall on neither sweep (e.g. carbon_realistic) are run too, so they can be plotted
on_sweep = Set((ld, lc) for (_, ld, lc) in sweeps)
for (name, l) in sort(collect(cfg["lambdas"]), by = first)
    (l["dollar"], l["carbon"]) in on_sweep || push!(sweeps, ("off_sweep", l["dollar"], l["carbon"]))
end

for (sweep, ld, lc) in sweeps
    println("$sweep: lambda_dollar = $ld, lambda_carbon = $lc")
    R = reward_table(ld, lc)
    Rm = Matrix(R)

    # trees fit on train, depth/minbucket tuned on val (lambda itself is fixed, never tuned)
    opt = tune_tree(X, R, train, val, cfg; localsearch = true, verbose = false)
    greedy = tune_tree(X, R, train, val, cfg; localsearch = false, depths = [opt.depth], verbose = false)
    best_single = best_single_model(R[train, :])

    for (split, rows) in eval_rows
        oracle = [model_names[argmax(Rm[r, :])] for r in rows]  # best model per prompt, in hindsight
        for (method, model, depth, minbucket, picks) in [
                ("optimal_tree", "", opt.depth, opt.minbucket, tree_picks(opt.lnr, X[rows, :])),
                ("greedy_tree", "", greedy.depth, greedy.minbucket, tree_picks(greedy.lnr, X[rows, :])),
                ("best_single", best_single, missing, missing, fill(best_single, length(rows))),
                ("oracle", "", missing, missing, oracle)]
            push!(out, (split = split, sweep = sweep, scenario = scenario_at(ld, lc),
                        lambda_dollar = ld, lambda_carbon = lc, method = method, model = model,
                        depth = depth, minbucket = minbucket,
                        reward = picked_mean(Rm, rows, picks, model_names),
                        outcomes(picks, rows)...); promote = true)
        end
        # sanity check: tree picks reproduce IAI's own reward on these rows
        @assert abs(picked_mean(Rm, rows, tree_picks(opt.lnr, X[rows, :]), model_names) -
                    avg_reward(opt.lnr, X[rows, :], R[rows, :])) < 1e-9
    end
end

sort!(out, [:split, :sweep, :lambda_dollar, :lambda_carbon])
CSV.write(joinpath(root, "results", "lambda_sweep.csv"), out)
show(transform(out, names(out, Union{Missing, Float64}) .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()
