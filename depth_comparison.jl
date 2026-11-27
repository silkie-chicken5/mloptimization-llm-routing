# Optimal vs greedy policy trees at each fixed depth: test reward, with a paired bootstrap
# interval on the difference, plus tree size.
# Run from the repo root: julia +1.11.6/IAI/3.2.2 depth_comparison.jl
using CSV, DataFrames, TOML, Statistics, CategoricalArrays, Random

root = @__DIR__
include(joinpath(root, "baselines.jl"))
cfg = TOML.parsefile(joinpath(root, "config.toml"))

DEPTHS = 1:5
N_BOOT = 2000
SCENARIOS = ["balanced", "efficiency_first", "carbon_realistic"]   # add more if time allows

# ------------------- Data -------------------
prompts = CSV.read(joinpath(root, "data", "processed", "prompts.csv"), DataFrame)
pairs = CSV.read(joinpath(root, "data", "processed", "pairs.csv"), DataFrame;
                 select = [:sample_id, :model, :quality, :cost, :carbon_cost])

train, val = prompts.split .== "train", prompts.split .== "val"
test_rows = findall(prompts.split .== "test")

X = select(prompts, Not([:sample_id, :split]))
X.task = categorical(X.task)

pairs.model_short = short_name.(pairs.model)
model_names = sort(unique(pairs.model_short))
quality_of = outcome_matrix(pairs, prompts.sample_id, :quality, model_names)
dollars_of = outcome_matrix(pairs, prompts.sample_id, :cost, model_names)
carbon_of  = outcome_matrix(pairs, prompts.sample_id, :carbon_cost, model_names)

# ------------------- Bootstrap draws -------------------
# Resample test prompts with replacement within each task; the same draws are used everywhere
rng = MersenneTwister(cfg["seed"])
test_tasks = prompts.task[test_rows]
draws = Matrix{Int}(undef, N_BOOT, length(test_rows))
for task in unique(test_tasks)
    cols = findall(==(task), test_tasks)
    draws[:, cols] = rand(rng, cols, N_BOOT, length(cols))
end

# Reward of each test prompt under a set of picks (one pick per test row)
function per_prompt_reward(R, picks)
    col = Dict(name => j for (j, name) in enumerate(names(R)))
    M = Matrix(R)
    return [M[row, col[model]] for (row, model) in zip(test_rows, picks)]
end

# ------------------- Fit both trees at every depth -------------------
runs = DataFrame()
differences = DataFrame()

for scenario in SCENARIOS
    lam = cfg["lambdas"][scenario]
    R = DataFrame(quality_of .- lam["dollar"] .* dollars_of .- lam["carbon"] .* carbon_of, model_names)

    for depth in DEPTHS
        println("$scenario, depth $depth")
        test_rewards = Dict{String, Vector{Float64}}()

        for (method, localsearch) in (("optimal_tree", true), ("greedy_tree", false))
            best = tune_tree(X, R, train, val, cfg; localsearch = localsearch, depths = [depth], verbose = false)
            picks = tree_picks(best.lnr, X[test_rows, :])
            rewards = per_prompt_reward(R, picks)
            test_rewards[method] = rewards

            push!(runs, (
                scenario = scenario, depth = depth, method = method, minbucket = best.minbucket,
                train = avg_reward(best.lnr, X[train, :], R[train, :]),
                val = best.score,
                test = mean(rewards),
                leaves = (IAI.get_num_nodes(best.lnr) + 1) ÷ 2,          # a binary tree with L leaves has 2L - 1 nodes
                models_used_on_test = length(unique(picks)),
            ))
        end

        # paired bootstrap on the per-prompt difference
        diff = test_rewards["optimal_tree"] .- test_rewards["greedy_tree"]
        boot = [mean(diff[draws[b, :]]) for b in 1:N_BOOT]
        push!(differences, (
            scenario = scenario, depth = depth,
            optimal_minus_greedy = mean(diff),
            ci_low = quantile(boot, 0.025), ci_high = quantile(boot, 0.975),
        ))
    end
end

CSV.write(joinpath(root, "results", "depth_comparison.csv"), runs)
CSV.write(joinpath(root, "results", "depth_differences.csv"), differences)

println("\n===== Optimal vs greedy by depth =====")
show(transform(runs, [:train, :val, :test] .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println("\n\n===== Optimal minus greedy test reward (95% bootstrap interval) =====")
show(transform(differences, [:optimal_minus_greedy, :ci_low, :ci_high] .=> (c -> round.(c, digits = 4)), renamecols = false),
     allrows = true, allcols = true)
println()