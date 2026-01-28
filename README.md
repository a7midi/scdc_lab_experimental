SCDC Lab Patch (H1/H3)

This patch fixes a major H1 pocket-closure issue and improves the H3 depth-null swap randomization.

1) scdc_lab/pockets.py
   - Force(S) no longer forces vertices with empty EssIn(v). This prevents the
     trivial blow-up where closures swallow huge fractions of the graph.

2) scdc_lab/diagnostics.py
   - depth_preserving_double_edge_swap now performs *accepted* swaps (up to a try limit),
     instead of counting failed attempts as swaps.

3) scdc_lab/experiments/h1_glider_search.py
   - Upgraded experiment script with:
       * layered graph controls (p_forward, p_skip, knot_layer)
       * rule selection (threshold/xor/random lookup)
       * active & pocket centroid tracking
       * extra diagnostics columns in CSV

To apply:
   - Copy the 'scdc_lab/' folder contents into your repo, overwriting the same files.

After patch, recommended run:
   python -m scdc_lab.experiments.h1_glider_search --graph_type layered --n 240 --layers 12 --knot_k 12 --steps 200 --rule xor --fixed_schedule --out_csv h1_fixed.csv
