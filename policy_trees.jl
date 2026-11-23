# Run from the repo root with the IAI Julia and this repo's environment:
#   julia +1.11.6/IAI/3.2.2 --project=. policy_trees.jl
using CSV, DataFrames, TOML, Statistics, CategoricalArrays

root = @__DIR__
include(joinpath(root, "baselines.jl"))
cfg = TOML.parsefile(joinpath(root, "config.toml"))

# ---------- Data shared by every scenario ----------
prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
pairs = CSV.read(joinpath(root, "data", "processed", "pairs.csv"), DataFrame;
                 select = [:sample_id, :model, :cost, :carbon_cost])

# train/val/test masks: fit on train, tune depth and minbucket on val, report on test
splits = Dict(s => prompts.split .== s for s in ["train", "val", "test"])
train, val = splits["train"], splits["val"]

X = select(prompts, Not([:sample_id, :split]))
X.task = categorical(X.task)

# Cheapest model depends only on cost, not lambda, so pick it once
cheapest = cheapest_model(pairs, prompts.sample_id[train])
println("cheapest model (train): ", cheapest)

# Average reward of the tree's picks, using the call from IAI's guides
function avg_reward(lnr, X, R)
    return mean(IAI.predict_outcomes(lnr, X, R))
end

# Grid search over depth and minbucket: fit on train, keep the tree with the best val reward.
# localsearch = true gives the optimal tree; localsearch = false grows it greedily, one split
# at a time (CART-style), with the same features and objective. Pass `depths` to fix the depth.
function tune_tree(X, R; localsearch, depths = cfg["tree"]["max_depth"])
    best = (score = -Inf, lnr = nothing, depth = 0, minbucket = 0)
    for depth in depths, minbucket in cfg["tree"]["minbucket"]
        lnr = IAI.OptimalTreePolicyMaximizer(
            max_depth = depth,
            minbucket = minbucket,
            cp = 0,
            localsearch = localsearch,
            random_seed = cfg["seed"],
        )
        IAI.fit!(lnr, X[train, :], R[train, :])

        val_score = avg_reward(lnr, X[val, :], R[val, :])
        train_score = avg_reward(lnr, X[train, :], R[train, :])
        println("  depth=$depth minbucket=$minbucket  train=$(round(train_score, digits=3))  val=$(round(val_score, digits=3))")

        if val_score > best.score
            best = (score = val_score, lnr = lnr, depth = depth, minbucket = minbucket)
        end
    end
    return best
end

# ---------- Run every scenario ----------
results = DataFrame()
# Order by total weight: quality_first, balanced, efficiency_first, carbon_first
scenarios = sort(collect(keys(cfg["lambdas"])), by = s -> sum(values(cfg["lambdas"][s])))

for scenario in scenarios
    lam = cfg["lambdas"][scenario]
    println("\n===== $scenario (lambda dollar = $(lam["dollar"]), carbon = $(lam["carbon"])) =====")
    rewards = CSV.read(joinpath(root, "data", "processed", "rewards_$scenario.csv"), DataFrame)
    @assert prompts.sample_id == rewards.sample_id
    R = select(rewards, Not(:sample_id))

    println("optimal tree:")
    opt = tune_tree(X, R; localsearch = true)
    # Greedy tree uses the optimal tree's depth so the two are comparable; only minbucket is tuned
    println("greedy (CART) tree, depth fixed at $(opt.depth):")
    greedy = tune_tree(X, R; localsearch = false, depths = [opt.depth])

    best_single = best_single_model(R[train, :])

    # One scoring function per method: rows of a split -> mean reward
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

    # Interactive pictures saved to files
    IAI.write_html(joinpath(root, "results", "tree_$scenario.html"), opt.lnr)
    IAI.write_html(joinpath(root, "results", "greedy_tree_$scenario.html"), greedy.lnr)
end

# Share of the best-single -> oracle gap each method closes on test (1 = oracle, 0 = best single)
transform!(groupby(results, :scenario),
    [:method, :test] => ((m, t) -> begin
        single, oracle = t[m .== "best_single"][1], t[m .== "oracle"][1]
        (t .- single) ./ (oracle - single)
    end) => :test_gap_closed)

CSV.write(joinpath(root, "results", "comparison.csv"), results)

println("\n===== Comparison (fit on train, tuned on val, reported on test) =====")
show(transform(results, [:train, :val, :test, :test_gap_closed] .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()
