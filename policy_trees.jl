#import Pkg; Pkg.add("CategoricalArrays")
using CSV, DataFrames, TOML, Statistics, CategoricalArrays

# data imports
root = @__DIR__
cfg = TOML.parsefile(joinpath(root, "config.toml"))
scenario = "balanced"

prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
rewards = CSV.read(joinpath(root, "data", "processed", "rewards_$scenario.csv"), DataFrame)
@assert prompts.sample_id == rewards.sample_id

# train/val/test mask
train = prompts.split .== "train"
val = prompts.split .== "val"

X = select(prompts, Not([:sample_id, :split])) # exclude id and split feat
X.task = categorical(X.task)
R = select(rewards, Not(:sample_id))

# uses max_depth and minbucket (values from config.toml)
grid = IAI.GridSearch(
    IAI.OptimalTreePolicyMaximizer(random_seed = cfg["seed"]),
    max_depth = cfg["tree"]["max_depth"],
    minbucket = cfg["tree"]["minbucket"],
)

# fit the model
IAI.fit!(grid, X[train, :], R[train, :], X[val, :], R[val, :])
println(IAI.get_grid_result_summary(grid))
println(IAI.get_best_params(grid))

# winning tree
best_lnr = IAI.get_learner(grid)

train_score = mean(IAI.predict_outcomes(best_lnr, X[train, :], R[train, :]))
val_score = mean(IAI.predict_outcomes(best_lnr, X[val, :], R[val, :]))
println("best tree: train=$(round(train_score, digits=3))  val=$(round(val_score, digits=3))")

# Interactive picture saved to a file
IAI.write_html(joinpath(root, "results", "tree_$scenario.html"), best_lnr)