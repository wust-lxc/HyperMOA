from __future__ import annotations

from dataclasses import dataclass


PANEL_FONT_PT = 8.0
MARKER_TOP_ROW_SHIFT_PT = 8.0


@dataclass(frozen=True)
class PanelLabel:
    letter: str
    description: str
    x_pt: float
    top_pt: float
    bounds_pt: tuple[float, float, float, float]


CLUSTERING_LABELS = (
    PanelLabel("a", "Ground truth and clustering maps", 6.0, 10.8, (12.756, 29.0, 502.583, 220.0)),
    PanelLabel("b", "Clustering metrics for the displayed section", 6.0, 224.1, (12.756, 231.0, 502.583, 300.898)),
)

ALIGNMENT_LABELS = (
    PanelLabel("a", "Human lymph node: registered sections", 6.0, 33.8, (12.756, 52.0, 502.583, 151.0)),
    PanelLabel("b", "Human lymph node: alignment metrics", 6.0, 150.2, (12.756, 157.0, 502.583, 222.0)),
    PanelLabel("c", "Simulation: registered sections", 6.0, 226.1, (12.756, 244.3, 502.583, 343.3)),
    PanelLabel("d", "Simulation: alignment metrics", 6.0, 342.5, (12.756, 349.3, 502.583, 414.3)),
    PanelLabel("e", "Mouse Brain: registered sections", 6.0, 418.4, (12.756, 436.6, 502.583, 535.6)),
    PanelLabel("f", "Mouse Brain: alignment metrics", 6.0, 534.8, (12.756, 541.6, 502.583, 611.0)),
)

MARKER_LABELS = (
    PanelLabel("a", "Anatomical annotation", 6.0, 3.1, (22.961, 21.465, 140.645, 158.193)),
    PanelLabel("b", "Individual anatomical regions", 156.0, 3.1, (172.861, 21.465, 476.591, 158.193)),
    PanelLabel("c", "HyperMOA domain map", 6.0, 166.6, (22.961, 185.108, 144.0, 296.679)),
    PanelLabel("d", "Anatomical composition of each domain", 166.0, 166.6, (182.127, 185.108, 470.2, 296.679)),
    PanelLabel("e", "Spatial RNA expression of the three representative markers", 6.0, 309.2, (22.961, 331.594, 95.0, 519.732)),
    PanelLabel("f", "Marker RNA expression distributions across domains", 102.0, 309.2, (111.7, 331.594, 264.907, 519.732)),
    PanelLabel("g", "Domain-associated RNA marker expression heatmap", 318.0, 309.2, (331.264, 331.594, 476.591, 519.732)),
)

EMBRYO_LABELS = (
    PanelLabel("a", "Original annotated sections from eight developmental stages", 6.0, 2.6, (20.409, 26.192, 496.971, 163.389)),
    PanelLabel("b", "Whole-embryo developmental-stage stack", 6.0, 177.4, (16.328, 201.43, 248.485, 396.0)),
    PanelLabel("c", "Brain, Heart and Liver locations across stages", 255.0, 177.4, (262.772, 201.43, 494.929, 396.0)),
    PanelLabel("d", "Adjacent developmental stages: original views and aligned overlays", 6.0, 380.0, (33.165, 422.0, 477.071, 595.56)),
)


def labels_for(figure_key: str) -> tuple[PanelLabel, ...]:
    if figure_key in ("Human_lymph_node", "Simulation", "Mouse_Brain"):
        return CLUSTERING_LABELS
    return {
        "alignment": ALIGNMENT_LABELS,
        "marker": MARKER_LABELS,
        "embryo": EMBRYO_LABELS,
    }[figure_key]


def add_panel_labels_to_figure(figure, figure_key: str) -> None:
    width_pt, height_pt = (float(value) * 72 for value in figure.get_size_inches())
    if figure_key == "marker" and not getattr(figure, "_manuscript_panel_layout_applied", False):
        # Use the existing gap below the first row to leave room above its headings.
        for axis in figure.axes:
            if (axis.get_gid() or "").startswith("anatomical_"):
                position = axis.get_position()
                axis.set_position((position.x0, position.y0 - MARKER_TOP_ROW_SHIFT_PT / height_pt,
                                   position.width, position.height))
        figure._manuscript_panel_layout_applied = True
    for panel in labels_for(figure_key):
        figure.text(
            panel.x_pt / width_pt,
            1 - panel.top_pt / height_pt,
            panel.letter,
            ha="left",
            va="top",
            fontsize=PANEL_FONT_PT,
            fontfamily="Arial",
            fontweight="bold",
            color="black",
            gid=f"manuscript_panel_label_{panel.letter}",
        )
