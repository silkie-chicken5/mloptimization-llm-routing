SEED = 42
SPLIT = (0.6, 0.2, 0.2)   # train / validation / test
MODELS = ["Mistral-7B", "Mixtral-8x7B", "Yi-34B-chat", "Llama-70B-chat"]  # match RouterBench column names
GRID_INTENSITY = None     # kg CO2/kWh, ISO-NE, from Electricity Maps
CARBON_PRICE = None       # $/metric ton, RGGI