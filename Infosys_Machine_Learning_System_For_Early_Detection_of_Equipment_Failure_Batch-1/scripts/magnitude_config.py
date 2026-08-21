"""RSR-19: the magnitude head's label-aware constants, in ONE place.

Previously the forward-drop clip/scale and the "positive window" floor were
hardcoded in several magnitude scripts and could drift. The magnitude label maps
the forward min-drop-% into [0,1]:

    label = clip(fwd_drop_pct, 0, CLIP_PCT) / SCALE_PCT

and a window counts as positive when its (predicted or realized) forward drop
exceeds ``POSITIVE_FLOOR_PCT`` percent.
"""
CLIP_PCT = 5.0      # max realized forward-drop-% captured by the label
SCALE_PCT = 5.0     # normalize by this so labels are in [0,1]
POSITIVE_FLOOR_PCT = 0.6  # forward-drop >= this -> the window is "positive"