#import Pkg; Pkg.add("CategoricalArrays")
using CSV, DataFrames, TOML, Statistics, CategoricalArrays

# data imports
root = @__DIR__
cfg = TOML.parsefile(joinpath(root, "config.toml"))
scenario = "quality_first"

prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
rewards = CSV.read(joinpath(root, "data", "processed", "rewards_$scenario.csv"), DataFrame)
@assert prompts.sample_id == rewards.sample_id

# train/val/test mask
train = prompts.split .== "train"
val = prompts.split .== "val"

X = select(prompts, Not([:sample_id, :split]))
X.task = categorical(X.task)
R = select(rewards, Not(:sample_id))

# Average reward of the tree's picks, using the call from IAI's guides
function avg_reward(lnr, X, R)
    return mean(IAI.predict_outcomes(lnr, X, R))
end

# Tune depth and minbucket on the validation rows
best_score = -Inf
best_lnr = nothing
for depth in cfg["tree"]["max_depth"], minbucket in cfg["tree"]["minbucket"]
    lnr = IAI.OptimalTreePolicyMaximizer(
        max_depth = depth,
        minbucket = minbucket,
        cp = 0,
        random_seed = cfg["seed"],
    )
    IAI.fit!(lnr, X[train, :], R[train, :])

    val_score = avg_reward(lnr, X[val, :], R[val, :])
    train_score = avg_reward(lnr, X[train, :], R[train, :])
    println("depth=$depth minbucket=$minbucket  train=$(round(train_score, digits=3))  val=$(round(val_score, digits=3))")

    if val_score > best_score
        global best_score = val_score
        global best_lnr = lnr
    end
end

# Baseline: always use the best single model (chosen on train)
best_model = argmax(Dict(name => mean(R[train, name]) for name in names(R)))
println("\nbest single model: ", best_model)
println("best single, val: ", mean(R[val, best_model]))
println("best tree, val:   ", best_score)

# Interactive picture saved to a file
IAI.write_html(joinpath(root, "results", "tree_$scenario.html"), best_lnr)