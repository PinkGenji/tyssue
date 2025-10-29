"""
This file contains all behaviour functions that models different cellular activities.

"""

import numpy as np
from ...geometry.planar_geometry import PlanarGeometry
from ...topology.sheet_topology import cell_division

"""cell proliferation behaviour function

A cell proliferation behaviour function does two thing on the selected cell.
First, it compares the current cell area with an area threshold to see if the cell large enough for division.
Secondly: if the cell is large enough, a cell division is performed on the cell; alternatively, if the cell area is 
smaller than the threshold value, then the target area is added by "growth_rate * dt" to expand the cell.
"""

# Note: still needs to improve the function, so it controls cell class change.
def proliferation(sheet, manager, geom, unique_id, crit_area, growth_rate, dt):
    idx = sheet.idx_lookup('face', unique_id)
    if sheet.face_df.loc[idx, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[idx, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geom)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        sgeom.update_all(sheet)
        print(f"cell n°{daughter} is born")
    else:
        sheet.face_df.loc[idx, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(division, geom = geom, unique_id = unique_id, crit_area = crit_area, growth_rate = growth_rate, dt = dt)




"""cell fusion behaviour function

A cell fusion behaviour function is used when a CT is fusing into the STB layer. The cell class of the selection cell 
should become "STB" at the end of the function.
First, the edges between the selected cell and its STB neighbours are disabled for edge tension term (coefficient = 0).
Secondly, we need to perform a T1 swap on the edge that connects a boundary vertex and the mutual vertex shared between all 
STB neighbours and the fusing cell; in this way, the newly fused cell would have an edge that is "open" to "outside".
Thirdly, new dynamic parameters need to be updated to ensure consistent physics rule.
"""
def fusion(sheet, manager, geom, unique_id):
    sheet.get_extra_indices()
    idx = sheet.idx_lookup('face', unique_id)
    # find the edge shared with STB units



"""cell extrusion behaviour function

A cell extrusion behaviour function has two components.
The first component is to let the STB to detach from the CT layer. This is done by a series of T1 transition on shared
edges between the selected STB and CTs.
The second component is to let the STB to shed from the layer. The shedding is modelled by cell removal but keep the
vertices shared between STB units that are still in the system.
"""


