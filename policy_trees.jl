"""Fit and tune our policy tree models, and compare results to the baseline models."""

# ------------------- Setup -------------------
# run with julia +1.11.6/IAI/3.2.2 policy_trees.jl
using CSV, DataFrames, TOML, Statistics, CategoricalArrays

root = @__DIR__
include(joinpath(root, "baselines.jl"))
cfg = TOML.parsefile(joinpath(root, "config.toml"))

# read in the data
prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
pairs = CSV.read(joinpath(root, "data", "processed", "pairs.csv"), DataFrame;
                 select = [:sample_id, :model, :quality, :cost, :wh, :carbon_cost])

# train/val/test masks
splits = Dict(s => prompts.split .== s for s in ["train", "val", "test"])
train, val = splits["train"], splits["val"]

X = select(prompts, Not([:sample_id, :split])) # exclude id and split feat
X.task = categorical(X.task)

# Every (prompt, model) outcome as a prompts x models matrix, rows in prompts.csv order
pairs.model_short = short_name.(pairs.model)
model_names = sort(unique(pairs.model_short))
quality_of = outcome_matrix(pairs, prompts.sample_id, :quality, model_names)
dollars_of = outcome_matrix(pairs, prompts.sample_id, :cost, model_names)
carbon_of  = outcome_matrix(pairs, prompts.sample_id, :carbon_cost, model_names)
wh_of      = outcome_matrix(pairs, prompts.sample_id, :wh, model_names)
test_rows = findall(splits["test"])
outcome_tables = DataFrame[]

# BASELINE: cheapest model (only need to pick once/doesn't depend on lambda so pick outside of loop)
cheapest = cheapest_model(pairs, prompts.sample_id[train])
println("cheapest model (train): ", cheapest)


# ---------- Run every scenario ----------
results = DataFrame()
# order by total weight: quality_first, balanced, efficiency_first, carbon_first
scenario_names = collect(keys(cfg["lambdas"]))
scenarios = sort(scenario_names, by = name -> cfg["lambdas"][name]["dollar"] + cfg["lambdas"][name]["carbon"])

for scenario in scenarios
    lam = cfg["lambdas"][scenario]
    println("\n===== $scenario (lambda dollar = $(lam["dollar"]), carbon = $(lam["carbon"])) =====")
    rewards = CSV.read(joinpath(root, "data", "processed", "rewards_$scenario.csv"), DataFrame)
    @assert prompts.sample_id == rewards.sample_id # check row order then drop id
    R = select(rewards, Not(:sample_id))

    # MODEL
    println("optimal tree:")
    opt = tune_tree(X, R, train, val, cfg; localsearch = true)

    # BASELINE: greedy uses optimal tree depth to compare the two
    println("greedy CART, depth fixed at $(opt.depth):")
    greedy = tune_tree(X, R, train, val, cfg; localsearch = false, depths = [opt.depth])

    # BASELINE: best single model
    best_single = best_single_model(R[train, :])

    # compute score per model (avg reward)
    methods = [
        ("optimal_tree", "", opt.depth, opt.minbucket, rows -> avg_reward(opt.lnr, X[rows, :], R[rows, :])),
        ("greedy_tree", "", greedy.depth, greedy.minbucket, rows -> avg_reward(greedy.lnr, X[rows, :], R[rows, :])),
        ("best_single", best_single, missing, missing, rows -> fixed_model_reward(R[rows, :], best_single)),
        ("cheapest", cheapest, missing, missing, rows -> fixed_model_reward(R[rows, :], cheapest)),
        ("oracle", "", missing, missing, rows -> oracle_reward(R[rows, :])),
    ]

    for (method, model, depth, minbucket, score) in methods
        push!(results, (
            scenario = scenario, lambda_dollar = lam["dollar"], lambda_carbon = lam["carbon"],
            method = method, model = model,
            depth = depth, minbucket = minbucket,
            train = score(train), val = score(val), test = score(splits["test"]),
        ); promote = true)
    end

    # ---- Quality, dollars and CO2 of each method's picks on the test rows ----
    n_test = length(test_rows)
    picks_by_method = [
        "optimal_tree" => tree_picks(opt.lnr, X[test_rows, :]),
        "greedy_tree"  => tree_picks(greedy.lnr, X[test_rows, :]),
        "best_single"  => fill(best_single, n_test),
        "cheapest"     => fill(cheapest, n_test),
    ]

    outcome_rows = DataFrame(method = String[], quality = Float64[],
                             dollars_per_1k = Float64[], kg_co2_per_1k = Float64[])
    for (method, picks) in picks_by_method
        @assert all(in(model_names), picks) "unexpected model names from $method: $(unique(picks))"
        quality = picked_mean(quality_of, test_rows, picks, model_names)
        dollars = picked_mean(dollars_of, test_rows, picks, model_names)
        carbon_dollars = picked_mean(carbon_of, test_rows, picks, model_names)
        wh = picked_mean(wh_of, test_rows, picks, model_names)

        # sanity check: these pieces must rebuild the test reward Julia already computed
        julia_test = only(results[(results.scenario .== scenario) .& (results.method .== method), :test])
        rebuilt = quality - lam["dollar"] * dollars - lam["carbon"] * carbon_dollars
        @assert abs(rebuilt - julia_test) < 1e-6 "picks for $method do not reproduce its test reward"

        push!(outcome_rows, (method, quality, dollars * 1000,
                             wh / 1000 * cfg["grid_intensity"] * 1000))  # Wh -> kWh -> kg CO2, per 1,000 queries
    end

    # relative to the best single model (positive "saved" = cheaper / cleaner than best single)
    base = only(eachrow(outcome_rows[outcome_rows.method .== "best_single", :]))
    outcome_rows.quality_retained = outcome_rows.quality ./ base.quality
    outcome_rows.dollars_saved_per_1k = base.dollars_per_1k .- outcome_rows.dollars_per_1k
    outcome_rows.kg_co2_saved_per_1k = base.kg_co2_per_1k .- outcome_rows.kg_co2_per_1k
    insertcols!(outcome_rows, 1, :scenario => scenario)
    push!(outcome_tables, outcome_rows)

    # tree diagrams saved to files (optimal & greedy)
    IAI.write_html(joinpath(root, "results", "tree_$scenario.html"), opt.lnr)
    IAI.write_html(joinpath(root, "results", "greedy_tree_$scenario.html"), greedy.lnr)
end

# "share of oracle gain captured": how much better each model gets
# (tree − best single) / (oracle − best single)
transform!(groupby(results, :scenario),
    [:method, :test] => ((m, t) -> begin
        single, oracle = t[m .== "best_single"][1], t[m .== "oracle"][1]
        (t .- single) ./ (oracle - single)
    end) => :share_oracle_gain)

CSV.write(joinpath(root, "results", "comparison.csv"), results)

# Wide summary: one row per scenario, one test-reward column per method
# (lambda_quality = 1 since reward = quality - lambda_dollar * dollars - lambda_carbon * carbon)
scores = unstack(results, [:scenario, :lambda_dollar, :lambda_carbon], :method, :test)
insertcols!(scores, 2, :lambda_quality => 1)
CSV.write(joinpath(root, "results", "test_scores_by_scenario.csv"), scores)

all_outcomes = vcat(outcome_tables...)
CSV.write(joinpath(root, "results", "outcomes_by_method.csv"), all_outcomes)

println("\n===== Quality, dollars and CO2 on the test rows =====")
show(transform(all_outcomes, names(all_outcomes, Real) .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()

println("\n===== Comparison (fit on train, tuned on val, reported on test) =====")
show(transform(results, [:train, :val, :test, :share_oracle_gain] .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()
