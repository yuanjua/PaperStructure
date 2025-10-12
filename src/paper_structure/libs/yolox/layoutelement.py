"""Layout element representation for YOLOX detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .constants import Source
from .elements import Rectangle, TextRegion, TextRegions


@dataclass
class LayoutElements(TextRegions):
    element_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    element_class_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    element_class_id_map: dict[int, str] = field(default_factory=dict)
    text_as_html: np.ndarray = field(default_factory=lambda: np.array([]))
    table_as_cells: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        element_size = self.element_coords.shape[0]
        # NOTE: maybe we should create an attribute _optional_attributes: list[str] to store this
        # list
        for attr in (
            "element_probs",
            "element_class_ids",
            "texts",
            "text_as_html",
            "table_as_cells",
        ):
            if getattr(self, attr).size == 0 and element_size:
                setattr(self, attr, np.array([None] * element_size))

        # for backward compatibility; also allow to use one value to set sources for all regions
        if self.sources.size == 0 and self.element_coords.size > 0:
            self.sources = np.array([self.source] * self.element_coords.shape[0])
        elif self.source is None and self.sources.size:
            self.source = self.sources[0]

        self.element_probs = self.element_probs.astype(float)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayoutElements):
            return NotImplemented

        mask = ~np.isnan(self.element_probs)
        other_mask = ~np.isnan(other.element_probs)
        return (
            np.array_equal(self.element_coords, other.element_coords)
            and np.array_equal(self.texts, other.texts)
            and np.array_equal(mask, other_mask)
            and np.array_equal(self.element_probs[mask], other.element_probs[mask])
            and (
                [self.element_class_id_map[idx] for idx in self.element_class_ids]
                == [other.element_class_id_map[idx] for idx in other.element_class_ids]
            )
            and np.array_equal(self.sources[mask], other.sources[mask])
            and np.array_equal(self.text_as_html[mask], other.text_as_html[mask])
            and np.array_equal(self.table_as_cells[mask], other.table_as_cells[mask])
        )

    def __getitem__(self, indices):
        return self.slice(indices)

    def slice(self, indices) -> LayoutElements:
        """slice and return only selected indices"""
        return LayoutElements(
            element_coords=self.element_coords[indices],
            texts=self.texts[indices],
            sources=self.sources[indices],
            element_probs=self.element_probs[indices],
            element_class_ids=self.element_class_ids[indices],
            element_class_id_map=self.element_class_id_map,
            text_as_html=self.text_as_html[indices],
            table_as_cells=self.table_as_cells[indices],
        )

    @classmethod
    def concatenate(cls, groups: Iterable[LayoutElements]) -> LayoutElements:
        """concatenate a sequence of LayoutElements in order as one LayoutElements"""
        coords, texts, probs, class_ids, sources = [], [], [], [], []
        text_as_html, table_as_cells = [], []
        class_id_reverse_map: dict[str, int] = {}
        for group in groups:
            coords.append(group.element_coords)
            texts.append(group.texts)
            probs.append(group.element_probs)
            sources.append(group.sources)
            text_as_html.append(group.text_as_html)
            table_as_cells.append(group.table_as_cells)

            idx = group.element_class_ids.copy()
            if group.element_class_id_map:
                for class_id, class_name in group.element_class_id_map.items():
                    if class_name in class_id_reverse_map:
                        idx[group.element_class_ids == class_id] = class_id_reverse_map[class_name]
                        continue
                    new_id = len(class_id_reverse_map)
                    class_id_reverse_map[class_name] = new_id
                    idx[group.element_class_ids == class_id] = new_id
            class_ids.append(idx)

        return cls(
            element_coords=np.concatenate(coords),
            texts=np.concatenate(texts),
            element_probs=np.concatenate(probs),
            element_class_ids=np.concatenate(class_ids),
            element_class_id_map={v: k for k, v in class_id_reverse_map.items()},
            sources=np.concatenate(sources),
            text_as_html=np.concatenate(text_as_html),
            table_as_cells=np.concatenate(table_as_cells),
        )

    def iter_elements(self):
        """iter elements as one LayoutElement per iteration; this returns a generator and has less
        memory impact than the as_list method"""
        for (x1, y1, x2, y2), text, prob, class_id, source, text_as_html, table_as_cells in zip(
            self.element_coords,
            self.texts,
            self.element_probs,
            self.element_class_ids,
            self.sources,
            self.text_as_html,
            self.table_as_cells,
        ):
            yield LayoutElement.from_coords(
                x1,
                y1,
                x2,
                y2,
                text=text,
                type=(
                    self.element_class_id_map[class_id]
                    if class_id is not None and self.element_class_id_map
                    else None
                ),
                prob=None if np.isnan(prob) else prob,
                source=source,
                text_as_html=text_as_html,
                table_as_cells=table_as_cells,
            )

    def as_list(self):
        """return a list of LayoutElement objects"""
        return list(self.iter_elements())

    @classmethod
    def from_list(cls, elements: list):
        """create LayoutElements from a list of LayoutElement objects; the objects must have the
        same source"""
        len_ele = len(elements)
        coords = np.empty((len_ele, 4), dtype=float)
        # text and probs can be Nones so use lists first then convert into array to avoid them being
        # filled as nan
        texts, text_as_html, table_as_cells, sources, class_probs = [], [], [], [], []
        class_types = np.empty((len_ele,), dtype="object")

        for i, element in enumerate(elements):
            coords[i] = [element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2]
            texts.append(element.text)
            sources.append(element.source)
            text_as_html.append(element.text_as_html)
            table_as_cells.append(element.table_as_cells)
            class_probs.append(element.prob)
            class_types[i] = element.type or "None"

        unique_ids, class_ids = np.unique(class_types, return_inverse=True)
        unique_ids[unique_ids == "None"] = None

        return cls(
            element_coords=coords,
            texts=np.array(texts),
            element_probs=np.array(class_probs),
            element_class_ids=class_ids,
            element_class_id_map=dict(zip(range(len(unique_ids)), unique_ids)),
            sources=np.array(sources),
            text_as_html=np.array(text_as_html),
            table_as_cells=np.array(table_as_cells),
        )


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None
    prob: Optional[float] = None
    image_path: Optional[str] = None
    parent: Optional[LayoutElement] = None
    text_as_html: Optional[str] = None
    table_as_cells: Optional[str] = None

    def to_dict(self) -> dict:
        """Converts the class instance to dictionary form."""
        out_dict = {
            "coordinates": None if self.bbox is None else self.bbox.coordinates,
            "text": self.text,
            "type": self.type,
            "prob": self.prob,
            "source": self.source,
        }
        return out_dict

    @classmethod
    def from_region(cls, region: TextRegion):
        """Create LayoutElement from superclass."""
        text = region.text if hasattr(region, "text") else None
        type = region.type if hasattr(region, "type") else None
        prob = region.prob if hasattr(region, "prob") else None
        source = region.source if hasattr(region, "source") else None
        return cls(text=text, source=source, type=type, prob=prob, bbox=region.bbox)
