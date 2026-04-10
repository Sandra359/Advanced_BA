import json

# Read the notebook
with open('Sandra.ipynb', 'r') as f:
    nb = json.load(f)

# Define the correct cell order (by their IDs)
correct_order = [
    "#VSC-c80aeacb",      # Empty markdown
    "#VSC-8e44069c",      # "## Elering data"
    "#VSC-441c02c1",      # Old exploratory code
    "#VSC-88a8edc9",      # "## Turnover"
    "#VSC-a67d83fc",      # Turnover code
    "#VSC-3b968a6f",      # "empty"
    "#VSC-86e124e4",      # "## Print times interval"
    "#VSC-1ca1241b",      # Old print intervals code
    # NEW ORDER: ST-GNN section
    "#VSC-927b9749",      # "## ST-GNN Forecasting Model"
    "#VSC-60b0ca8e",      # 1. FETCH FLOWS
    "#VSC-331b6a3c",      # 2. FETCH PRICES
    "#VSC-1e1da3b0",      # 3. FETCH PRODUCTION
    "#VSC-85eee871",      # 4. BUILD GRAPH
    "#VSC-0d7d1db3",      # 5. PREPARE FEATURES
    "#VSC-987bd6da",      # 6. MODEL DEFINITION
    "#VSC-9635f121",      # 7. TRAINING
    "#VSC-3ac096e4",      # 8. SCENARIO SIMULATION
    "#VSC-e7d4e0f8",      # 9. VISUALIZATION
    "#VSC-05907a5e",      # 10. FEATURE IMPORTANCE
    "#VSC-7ae82005",      # Summary
]

# Create ordered cells list
cells_by_id = {cell['id']: cell for cell in nb['cells']}
ordered_cells = [cells_by_id[cid] for cid in correct_order if cid in cells_by_id]

# Update notebook
nb['cells'] = ordered_cells

# Write back
with open('Sandra.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f"Reordered {len(ordered_cells)} cells")
