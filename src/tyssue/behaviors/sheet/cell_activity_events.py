"""
This file contains all behaviour functions that models different cellular activities.

"""

import numpy as np
import pandas as pd
from ...geometry.planar_geometry import PlanarGeometry
from ...topology.sheet_topology import cell_division, type1_transition, remove_face

"""cell proliferation behaviour function

A cell proliferation behaviour function does two thing on the selected cell.
First, it compares the current cell area with an area threshold to see if the cell large enough for division.
Secondly: if the cell is large enough, a cell division is performed on the cell; alternatively, if the cell area is 
smaller than the threshold value, then the target area is added by "growth_rate * dt" to expand the cell.
"""
# Note: still needs to improve the function, so it controls cell class change.
def proliferation(sheet, manager, geom, unique_id, crit_area, growth_rate, dt):
    idx = sheet.idx_lookup(unique_id,'face') # get the current face index from unique_id.
    if sheet.face_df.loc[idx, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[idx, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, idx, geom)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geom.update_all(sheet)
        print(f"cell n°{daughter} is born")
    else:
        sheet.face_df.loc[idx, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(proliferation, geom = geom, unique_id = unique_id, crit_area = crit_area, growth_rate = growth_rate, dt = dt)


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
    # store the face index of STB neighbours
    STB_neighbours = sheet.get_neighbors(idx)
    STB_neighbours = list(STB_neighbours.intersection(set(sheet.face_df.loc[sheet.face_df['cell_class'] == 'STB'].index)))
    # Find the edges associated with the fusing face index, filter out the boundary edges.
    internal_edges = sheet.edge_df[(sheet.edge_df['face'] == idx) & (sheet.edge_df['opposite'] != -1)]
    # We have to update both the internal arrowed edge and its opposite arrowed edge.
    for ie in internal_edges.index:
        sheet.edge_df.loc[ie,'is_active'] = 0
        sheet.edge_df.loc[sheet.edge_df.loc[ie,'opposite'],'is_active'] = 0

    # Find all the boundary vertices in STB neighbours, based on the opposite == -1 value.
    STB_boundary_edges = sheet.edge_df[
        (sheet.edge_df['face'].isin(STB_neighbours)) &
        (sheet.edge_df['opposite'] == -1)
        ]
    STB_boundary_verts = pd.unique(STB_boundary_edges[['srce','trgt']].values.ravel())
    # Next, extract all the vertices belong to the fusing face, the edge connects a boundary vertex and the fusing face
    # is the edge that we should perform T1 swap on. We can utilise the variable internal_edges.
    fusing_face_verts = internal_edges['srce']
    # We only need to looping over the STB neighbours edges, then find edges connects a boundary vertex to a fusing face vertex.
    STB_edges = sheet.edge_df[sheet.edge_df['face'].isin(STB_neighbours)]
    matching_edge = STB_edges[
        (
                (STB_edges['srce'].isin(fusing_face_verts)) & (STB_edges['trgt'].isin(STB_boundary_verts))
        ) |
        (
                (STB_edges['trgt'].isin(fusing_face_verts)) & (STB_edges['srce'].isin(STB_boundary_verts))
        )
    ]
    if matching_edge.empty:
        raise ValueError("No matching edge found between fusing face vertices and STB boundary vertices.")
    first_edge_index = matching_edge.index[0]
    New_boundary_edge = type1_transition(sheet,first_edge_index,do_reindex=False, remove_tri_faces=False, multiplier=1.5)
    # Then make the new boundary edge to be active in tension (was dummy before T1).
    sheet.edge_df.loc[New_boundary_edge,'is_active'] = 1
    geom.update_all(sheet)


"""cell extrusion behaviour function

A cell extrusion behaviour function has two components.

The first component is to let the STB to detach from the CT layer. This is done by a series of T1 transition on shared
edges between the selected STB and CTs: Detach STB.

The second component is to let the STB to shed from the layer. The shedding is modelled by cell removal but keep the
vertices shared between STB units that are still in the system: STB removal.
"""
def extrude(sheet, manager, geom, unique_id):
    idx = sheet.idx_lookup('face', unique_id)
    remove.face(sheet,idx)
    geom.update_all(sheet)

def detach(sheet, manager, geom, unique_id):
    sheet.get_extra_indices()
    idx = sheet.idx_lookup('face', unique_id)
    # Identify CT neighbours (non-STB)
    CT_neighbours = sheet.get_neighbors(idx)
    CT_neighbours = list(CT_neighbours.intersection(
        set(sheet.face_df.loc[sheet.face_df['cell_class'] != 'STB'].index)
    ))
    # Find internal edges of the detaching face
    internal_edges = sheet.edge_df[(sheet.edge_df['face'] == idx) & (sheet.edge_df['opposite'] != -1)]
    # Filter internal edges that are shared with CT neighbours
    shared_edges = internal_edges[
        sheet.edge_df.loc[internal_edges['opposite'], 'face'].isin(CT_neighbours).values
    ]
    if shared_edges.empty:
        raise ValueError("No mutual edge found between detaching face and CT neighbours.")
    # Perform T1 transitions on each shared edge and update geometry
    for edge_idx in shared_edges.index:
        new_edge_idx = type1_transition(sheet, edge_idx, do_reindex=False, remove_tri_faces=False, multiplier=1.5)
        sheet.edge_df.loc[new_edge_idx, 'is_active'] = 1
        geom.update_all(sheet)
    # Optionally extrude the detached face
    manager.append(extrude, geom = geom, unique_id = unique_id)
