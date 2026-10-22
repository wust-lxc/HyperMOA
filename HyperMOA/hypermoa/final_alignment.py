"""Finalize fixed refinement without changing clustering or repeating a warp."""

import json
import numpy as np

from hypermoa.alignment_refinement import refine_anndata, refinement_protocol


def finalize_anndata(adata, batch_order, *, batch_key="batch", embedding_key="HyperMOA_bc",
                     seed=42, copy=True):
    """Keep alignment_input for reproduction; transform_fine is the final output."""
    output = adata.copy() if copy else adata
    if "alignment_input" not in output.obsm:
        output.obsm["alignment_input"] = np.asarray(output.obsm["transform_fine"]).copy()
    temporary = "_final_alignment_coordinates"
    output, diagnostics = refine_anndata(
        output, batch_order, batch_key=batch_key, embedding_key=embedding_key,
        spatial_key="alignment_input", key_added=temporary, seed=seed, inplace=True)
    output.obsm["transform_fine"] = output.obsm[temporary].copy()
    output.obsm["aligned_spatial"] = output.obsm[temporary].copy()
    del output.obsm[temporary]
    del output.uns[f"{temporary}_protocol"]
    output.uns["alignment_refinement_protocol"] = json.dumps({**refinement_protocol(),
        "seed": int(seed), "batch_order": list(map(str, batch_order)),
        "input_key": "alignment_input", "output_key": "transform_fine",
        "chain_accepted": diagnostics.chain_accepted}, sort_keys=True)
    return output, diagnostics
