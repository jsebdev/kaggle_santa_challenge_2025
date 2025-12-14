
from dataclasses import dataclass, field
from decimal import Decimal

from utils.tree import ChristmasTree


@dataclass
class HighlightTreeData:
    has_collision: bool


@dataclass
class Snapshot:
    trees: list[ChristmasTree]
    side_length: Decimal
    selected_trees: dict[int, HighlightTreeData] = field(default_factory=dict)
    text: str = ""
    title: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
