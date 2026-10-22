"""Shared typography settings for the six manuscript figures."""

from __future__ import annotations

import math
import re

from matplotlib.text import Text
from matplotlib.transforms import offset_copy


def manuscript_fontsize(figure_key: str, text: str, size: float, context: str = "") -> float:
    if "manuscript_panel_label_" in context:
        return 8.0
    if figure_key == "embryo" and text == "Adjacent sections: native views and model-aligned overlays":
        return 7.0
    size = round(float(size), 2)
    if "legend" in context.lower():
        return 6.0
    if figure_key in ("Simulation", "Human_lymph_node", "Mouse_Brain"):
        return 7.0 if size > 7.0 else max(6.0, min(size, 7.0))
    if figure_key == "alignment":
        if text.strip() == "Method" or math.isclose(size, 7.8):
            return 6.5
        if math.isclose(size, 7.2):
            return 6.0 if re.fullmatch(r"[−+\-]?\d+(?:\.\d+)?", text.strip()) else 6.5
        if math.isclose(size, 6.8):
            return 6.5
        if math.isclose(size, 6.5):
            return 6.0
        return max(6.0, min(size, 7.0))
    if figure_key == "marker":
        if text == "HyperMOA domain" or math.isclose(size, 5.8):
            return 6.0
        return max(5.5, min(size, 7.0))
    if figure_key == "embryo":
        if math.isclose(size, 6.1):
            return 6.0
        return max(6.0, min(size, 7.0))
    raise ValueError(figure_key)


def apply_manuscript_typography(figure, figure_key: str) -> None:
    """Apply the same font sizes when a figure is regenerated from source data."""
    if getattr(figure, "_manuscript_typography_applied", False):
        return
    owner_context = {
        id(text): axis.get_gid() or ""
        for axis in figure.axes
        for text in axis.findobj(match=Text)
    }
    for text in figure.findobj(match=Text):
        if not text.get_text():
            continue
        axis_gid = owner_context.get(id(text), text.axes.get_gid() if text.axes is not None else "")
        context = f"{axis_gid or ''} {text.get_gid() or ''}"
        text.set_fontfamily("Arial")
        text.set_fontsize(manuscript_fontsize(figure_key, text.get_text(), text.get_fontsize(), context))
        if "manuscript_panel_label_" in context:
            text.set_fontweight("bold")
            text.set_fontstyle("normal")
        if figure_key in ("Simulation", "Human_lymph_node", "Mouse_Brain") and re.fullmatch(r"[−+\-]?\d+\.\d{3}", text.get_text().strip()):
            text.set_transform(offset_copy(text.get_transform(), fig=figure, y=-0.38, units="points"))
        if figure_key == "marker":
            content = text.get_text()
            if "expression across domains" in content or "  |  D" in content:
                text.set_transform(offset_copy(text.get_transform(), fig=figure, y=5.0, units="points"))
            if content == "Domain-associated marker expression":
                text.set_transform(offset_copy(text.get_transform(), fig=figure, y=1.0, units="points"))
            if content == "Anatomical annotation":
                text.set_transform(offset_copy(text.get_transform(), fig=figure, y=-8.0, units="points"))
            if content == "HyperMOA domain":
                text.set_transform(offset_copy(text.get_transform(), fig=figure, y=4.5, units="points"))
            if axis_gid in ("anatomical_annotation", "hypermoa_domain_map") and text.get_path_effects():
                text.set_path_effects([])
                text.set_color("#20242A")
                text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.3})
    figure._manuscript_typography_applied = True
