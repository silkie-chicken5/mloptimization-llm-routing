"""Fit and tune our policy tree models, and compare results to the baseline models."""

# ------------------- Setup -------------------
# run with julia +1.11.6/IAI/3.2.2 --project=. policy_trees.jl
using CSV, DataFrames, TOML, Statistics, CategoricalArrays

root = @__DIR__
include(joinpath(root, "baselines.jl"))
cfg = TOML.parsefile(joinpath(root, "config.toml"))

# read in the data
prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
pairs = CSV.read(joinpath(root, "data", "processed", "pairs.csv"), DataFrame;
                 select = [:sample_id, :model, :cost, :carbon_cost])

# train/val/test masks
splits = Dict(s => prompts.split .== s for s in ["train", "val", "test"])
train, val = splits["train"], splits["val"]

X = select(prompts, Not([:sample_id, :split])) # exclude id and split feat
X.task = categorical(X.task)

# BASELINE: cheapest model (only need to pick once/doesn't depend on lambda so pick outside of loop)
cheapest = cheapest_model(pairs, prompts.sample_id[train])
println("cheapest model (train): ", cheapest)


# Grid search over max_depth and minbucket (values from config.toml): fit on train & keep tree w best val reward.
function tune_tree(X, R; localsearch, depths = cfg["tree"]["max_depth"])
    """
    X: the data
    R: the reward matrix
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
        train_score = avg_reward(lnr, X[train, :], R[train, :])
        println("  depth=$depth minbucket=$minbucket  train=$(round(train_score, digits=3))  val=$(round(val_score, digits=3))")

        if val_score > best.score # keeps best tree
            best = (score = val_score, lnr = lnr, depth = depth, minbucket = minbucket)
        end
    end
    return best
end

# ---------- Run every scenario ----------
results = DataFrame()
# order by total weight: quality_first, balanced, efficiency_first, carbon_first
scenario_names = collect(keys(cfg["lambdas"]))
scenarios = sort(scenario_names, by = cfg["lambdas"][name]["dollar"] + cfg["lambdas"][name]["carbon"])

for scenario in scenarios
    lam = cfg["lambdas"][scenario]
    println("\n===== $scenario (lambda dollar = $(lam["dollar"]), carbon = $(lam["carbon"])) =====")
    rewards = CSV.read(joinpath(root, "data", "processed", "rewards_$scenario.csv"), DataFrame)
    @assert prompts.sample_id == rewards.sample_id # check row order then drop id
    R = select(rewards, Not(:sample_id))

    # MODEL
    println("optimal tree:")
    opt = tune_tree(X, R; localsearch = true)

    # BASELINE: greedy uses optimal tree depth to compare the two
    println("greedy CART, depth fixed at $(opt.depth):")
    greedy = tune_tree(X, R; localsearch = false, depths = [opt.depth])

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

println("\n===== Comparison (fit on train, tuned on val, reported on test) =====")
show(transform(results, [:train, :val, :test, :share_oracle_gain] .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()
