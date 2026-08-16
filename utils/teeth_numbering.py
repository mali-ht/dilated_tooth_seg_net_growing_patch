import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import trimesh
from operator import itemgetter

_teeth_color = {
    8: (129, 175, 129),
    7: (241, 215, 145),
    6: (177, 122, 100),
    5: (111, 184, 210),
    4: (217, 101, 79),
    3: (221, 130, 101),
    2: (144, 239, 144),
    1: (7, 118, 7),
    9: (128, 175, 169),
    10: (242, 215, 185),
    11: (176, 122, 140),
    12: (112, 184, 250),
    13: (218, 101, 119),
    14: (222, 130, 141),
    15: (145, 239, 184),
    16: (8, 118, 47),
    # gum
    0: (204, 204, 204)
}

_teeth_labels = {
    0: 'gum',
    1: 'l_central_incisor',
    2: 'l_lateral_incisor',
    3: 'l_canine',
    4: 'l_1_st_premolar',
    5: 'l_2_nd premolar',
    6: 'l_1_st_molar',
    7: 'l_2_nd_molar',
    8: 'l_3_nd_molar',
    9: 'r_central_incisor',
    10: 'r_lateral_incisor',
    11: 'r_canine',
    12: 'r_1_st_premolar',
    13: 'r_2_nd premolar',
    14: 'r_1_st_molar',
    15: 'r_2_nd_molar',
    17: 'r_3_nd_molar'
}

_teeth_codes_lower = {
    11: (8, 'central_incisor'),
    12: (7, 'lateral_incisor'),
    13: (6, 'canine'),
    14: (5, '1_st_premolar'),
    15: (4, '2_nd premolar'),
    16: (3, '1_st_molar'),
    17: (2, '2_nd_molar'),
    18: (1, '3_nd_molar'),
    21: (9, 'central_incisor'),
    22: (10, 'lateral_incisor'),
    23: (11, 'canine'),
    24: (12, '1_st_premolar'),
    25: (13, '2_nd premolar'),
    26: (14, '1_st_molar'),
    27: (15, '2_nd_molar'),
    28: (16, '3_nd_molar'),
    0: (0, 'gum')
}

_teeth_codes_upper = {
    31: (8, 'central_incisor'),
    32: (7, 'lateral_incisor'),
    33: (6, 'canine'),
    34: (5, '1_st_premolar'),
    35: (4, '2_nd premolar'),
    36: (3, '1_st_molar'),
    37: (2, '2_nd_molar'),
    38: (1, '3_nd_molar'),
    41: (9, 'central_incisor'),
    42: (10, 'lateral_incisor'),
    43: (11, 'canine'),
    44: (12, '1_st_premolar'),
    45: (13, '2_nd premolar'),
    46: (14, '1_st_molar'),
    47: (15, '2_nd_molar'),
    48: (16, '3_nd_molar'),
    0: (0, 'gum')
}

_gum = (0, 'gum')

# Coarse 5-class remap of the 17-class (0-16) scheme: {gum, incisor, canine, premolar, molar}.
# Position-in-arch is what determines tooth type in this label scheme (see _teeth_codes_lower/
# upper above), so this mapping is the same regardless of upper/lower arch or L/R side.
_coarse_class_names = {
    0: 'gum',
    1: 'incisor',
    2: 'canine',
    3: 'premolar',
    4: 'molar',
}

_label_to_coarse_label = {
    0: 0,                              # gum
    7: 1, 8: 1, 9: 1, 10: 1,           # incisors (lateral + central, both sides)
    6: 2, 11: 2,                       # canines
    4: 3, 5: 3, 12: 3, 13: 3,          # premolars (1st + 2nd, both sides)
    1: 4, 2: 4, 3: 4, 14: 4, 15: 4, 16: 4,  # molars (1st/2nd/3rd, both sides)
}
_coarse_lookup_table = np.array([_label_to_coarse_label[i] for i in range(17)])

# distinct from _teeth_color (which is the model's real 17-class color scheme, used by
# color_mesh/colors_to_label) - this is only for visually distinguishing the 5 coarse classes.
_coarse_class_color = {
    0: (255, 192, 203),  # gum - light pink
    1: (70, 130, 180),   # incisor - steel blue
    2: (255, 140, 0),    # canine - dark orange
    3: (60, 179, 113),   # premolar - medium sea green
    4: (147, 112, 219),  # molar - medium purple
}
_coarse_color_lookup_table = np.array([_coarse_class_color[i] for i in range(5)])



def color_mesh(mesh: trimesh, labels: np.ndarray) -> trimesh:
    mesh = mesh.copy()
    colors = label_to_colors(labels)
    mesh.visual.face_colors = colors
    return mesh


def fdi_to_label(fdi_codes: np.ndarray) -> np.ndarray:
    teeth_codes = {**_teeth_codes_upper, **_teeth_codes_lower}
    labels = itemgetter(*list(fdi_codes))(teeth_codes)
    return np.array([l[0] for l in labels])


# Standard FDI -> Universal Numbering System (1-32) conversion, verified against known reference
# points (FDI 11 = Universal 8, upper right central incisor; FDI 48 = Universal 32, lower right
# 3rd molar). Independent of this codebase's internal label scheme - a fixed dental-notation fact.
_UNIVERSAL_FROM_FDI = {}
for _pos in range(1, 9):
    _UNIVERSAL_FROM_FDI[10 + _pos] = 9 - _pos    # FDI quadrant 1 -> Universal 8..1
    _UNIVERSAL_FROM_FDI[20 + _pos] = 8 + _pos    # FDI quadrant 2 -> Universal 9..16
    _UNIVERSAL_FROM_FDI[30 + _pos] = 25 - _pos   # FDI quadrant 3 -> Universal 24..17
    _UNIVERSAL_FROM_FDI[40 + _pos] = 24 + _pos   # FDI quadrant 4 -> Universal 25..32
del _pos


def label_to_universal_number(labels: np.ndarray, arch: str) -> np.ndarray:
    """Internal 1-16 label (0=gum) -> Universal Numbering System (1-32). Needs an explicit `arch`
    ('upper' or 'lower') because the internal label scheme mirrors L/R only, not upper/lower (see
    _teeth_codes_lower/_teeth_codes_upper - the SAME internal label means different real teeth on
    the two arches) - there's no way to recover which arch a label came from from the label alone.
    Returns 0 for gum (a sentinel; real Universal numbers are 1-32, never 0).

    This assumes the same patient left/right orientation convention the training data (Teeth3DS,
    via _teeth_codes_lower/_teeth_codes_upper's own FDI codes) used. Verify against a known
    ground-truth example before relying on this for anything beyond a live visual aid.

    NOTE: _teeth_codes_lower/_teeth_codes_upper are named backwards relative to what they
    actually contain - confirmed directly against a real Teeth3DS *_lower.json label file, whose
    FDI codes (31-47, quadrants 3&4 - real mandibular/lower per the FDI standard) match
    _teeth_codes_UPPER's keys, not _teeth_codes_lower's (11-28, quadrants 1&2 - real maxillary/
    upper). This never mattered before: fdi_to_label merges both dicts into one combined lookup
    (their key sets are disjoint) and never needed to pick one specifically. Left the original
    dict names alone (part of the reference implementation, and fdi_to_label's merge is correct
    regardless of the naming) - just selecting the right one here.
    """
    if arch not in ('upper', 'lower'):
        raise ValueError(f"arch must be 'upper' or 'lower', not {arch!r}")
    codes = _teeth_codes_upper if arch == 'lower' else _teeth_codes_lower
    label_to_fdi = {internal_label: fdi for fdi, (internal_label, _name) in codes.items()}
    lookup = np.zeros(17, dtype=np.int64)
    for internal_label, fdi in label_to_fdi.items():
        lookup[internal_label] = _UNIVERSAL_FROM_FDI.get(fdi, 0)
    return lookup[labels]


def label_to_coarse_label(labels: np.ndarray) -> np.ndarray:
    """Remap the 17-class (0-16) label scheme to the coarse 5-class scheme (see
    _coarse_class_names): {0: gum, 1: incisor, 2: canine, 3: premolar, 4: molar}."""
    return _coarse_lookup_table[labels]


def coarse_label_to_colors(coarse_labels: np.ndarray) -> np.ndarray:
    """RGB colors for the coarse 5-class scheme (see _coarse_class_color) - a visualization-only
    palette distinct from the model's real 17-class _teeth_color scheme."""
    return _coarse_color_lookup_table[coarse_labels]


def label_to_colors(labels: np.ndarray) -> np.ndarray:
    sorted_index = sorted(list(_teeth_color.items()), key=lambda tup: tup[0])
    # class_labels = np.array([i[0] for i in sorted_index])
    class_colors = np.array([i[1] for i in sorted_index])

    return class_colors[labels]


def colors_to_label(colors: np.ndarray) -> np.ndarray:
    # ignore alpha channel
    colors = colors[:, :3]
    colors = np.repeat(colors.reshape(-1, 1, 3), 17, axis=1)
    sorted_index = sorted(list(_teeth_color.items()), key=lambda tup: tup[0])
    class_labels = np.array([i[0] for i in sorted_index])
    class_colors = np.array([i[1] for i in sorted_index])
    class_colors = np.repeat(class_colors.reshape(1, -1, 3), colors.shape[0], axis=0)
    mask = (class_colors == colors)
    return class_labels[np.argmax(mask.all(axis=2), axis=1)]
