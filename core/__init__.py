"""Paper 9 toolbox core modules.

Algorithm implementations (not the ArcGIS UI, which is LandUseOptimization_P9.pyt).

Module layout:
    prepare_data.py      -- Tool 1: DEM + DLTB + XZQ -> blocks.gpkg
    sample_transitions.py -- Tool 2: sample transitions + pairwise
    train_ensemble.py    -- Tool 3: contrastive ensemble training
    mpc_plan.py          -- Tool 4: MPC rollout + write shapefile
    ensemble_runner.py   -- ONNX ensemble inference helper (used by mpc_plan)
    transition_model.py  -- Vendored TransitionModel (for training only)
    shapefile_io.py      -- DLTB / blocks IO helpers
    adjacency.py         -- PolygonNeighbors-based block adjacency
"""
