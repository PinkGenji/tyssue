"""
This file contains all behaviour functions that models different cellular activities.

"""

import numpy as np
import pandas as pd
from ...geometry.planar_geometry import PlanarGeometry
from ...topology.sheet_topology import cell_division, type1_transition, remove_face, add_vert
from ...draw.bilayer_drawing_tool import bilayer_draw_spec_update
from ...behaviors.sheet.bilayer_dummy_set import auto_dummy_edges
"""cell proliferation behaviour function

A cell proliferation behaviour function does two thing on the selected cell.
First, it compares the current cell area with an area threshold to see if the cell large enough for division.
Secondly: if the cell is large enough, a cell division is performed on the cell; alternatively, if the cell area is 
smaller than the threshold value, then the target area is added by "growth_rate * dt" to expand the cell.
"""
# Note: still needs to improve the function, so it controls cell class change.
def proliferation(sheet, manager, geom, unique_id, crit_area, growth_rate, dt):
    idx = sheet.idx_lookup(unique_id,'face') # get the current face index from unique_id.
    # cell performs a division if its area exceeds the critical area.
    if sheet.face_df.loc[idx, "area"] > crit_area:
        # restore preferred area of the dividing cell before division
        sheet.face_df.loc[idx, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, idx, geom, angle=np.pi) # The division tries to be a horizontal cut.
        daughter_id = sheet.face_df.loc[daughter, 'unique_id']
        print(f"cell #{daughter_id} is born")
        manager.append(proliferation, geom=geom, unique_id=unique_id, crit_area=crit_area, growth_rate=growth_rate,
                       dt=dt)
        manager.append(proliferation, geom=geom, unique_id=daughter_id, crit_area=crit_area,
                          growth_rate=growth_rate, dt=dt)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geom.update_all(sheet)
    # If cell area is less than critical area, then it keeps expanding by updating its prefered area.
    else:
        sheet.face_df.loc[idx, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(proliferation, geom = geom, unique_id = unique_id, crit_area = crit_area, growth_rate = growth_rate, dt = dt)


def face_vertices(sheet, face_id):
    """
    Given a face_id, return the list of vertex indices that are part of that face.
    """
    edges = sheet.edge_df[sheet.edge_df['face'] == face_id]
    verts = list(edges['srce']) + list(edges['trgt'])
    return list(set(verts))
def find_local_stb_stb_edge(sheet, F_cell):
    """
    Find the ONE STB–STB mutual edge such that:
    1. Both faces are STB neighbours of F_cell.
    2. At least one endpoint of the edge is a vertex of F_cell.
    Only loops over sheet.sgle_edges.
    Returns a single integer edge index, or None.
    """

    sheet.get_extra_indices()

    # Vertices of the F cell (force into Python ints)
    F_vertices = list(map(int, face_vertices(sheet, F_cell)))

    # STB neighbours of F_cell
    neighbours = sheet.get_neighbors(F_cell)
    stb_neigh = [n for n in neighbours if sheet.face_df.loc[n, 'cell_class'] == 'STB']

    # Loop ONLY over unique edges
    for e in sheet.sgle_edges:
        # f1 is the face index of the edge e belongs to.
        f1 = sheet.edge_df.loc[e, 'face']
        opp = sheet.edge_df.loc[e, 'opposite']
        if opp == -1:
            continue
        # f2 is the face index of the opposite edge of edge e belongs to.
        f2 = sheet.edge_df.loc[opp, 'face']

        # Condition 1: both faces are STB neighbours of F_cell
        if f1 not in stb_neigh or f2 not in stb_neigh:
            continue
        # Condition 2: edge touches the F cell
        v1 = sheet.edge_df.loc[e, 'srce']
        v2 = int(sheet.edge_df.loc[e, 'trgt'])

        if v1 in F_vertices or v2 in F_vertices:
            return e  # return immediately

    return None


def identify_edge_endpoints(sheet, F_cell, indirect_edge):
    """
    For each edge in local_edges, determine:
    - which endpoint belongs to the F cell
    - which endpoint belongs to the STB neighbour
    Returns a list: [STB_vertex, F_vertex]
    """

    F_vertices = face_vertices(sheet, F_cell)
    v1 = sheet.edge_df.loc[indirect_edge, 'srce']
    v2 = sheet.edge_df.loc[indirect_edge, 'trgt']

    # Determine which vertex belongs to the F cell
    if v1 in F_vertices and v2 not in F_vertices:
        return [v2, v1]

    elif v2 in F_vertices and v1 not in F_vertices:
        return [v1, v2]

    # Return None if neither vertex belongs to the F cell (should not happen if preconditions are met)
    return None

def fuse_single_cell(sheet, F_cell, extra_time = 1):
    """
    Attempt to fuse a CT cell (now in class 'F') into the STB layer.

    Fusion requires a specific geometric configuration:
    - The F cell must touch an STB–STB mutual edge.
    - That edge must share a vertex with the F cell.
    - Only then can the geometric fusion (vertex splitting + T1) proceed.

    If the geometry is NOT ready (e.g., due to T1/T2/T3 transitions or cell division),
    the fusion is postponed by extending the F timer. This prevents:
        - invalid topology operations,
        - isolated STB cells,
        - broken bilayer structure,
        - simulation crashes.

    Parameters
    ----------
    sheet : tyssue.Sheet
        The current tissue sheet.
    F_cell : int
        Index of the cell attempting to fuse.

    Returns
    -------
    new_edge : int or None
        The index of the newly created edge after fusion,
        or None if fusion was postponed.
    """
    sse = find_local_stb_stb_edge(sheet, F_cell)
    if sse is None:
        # Geometry not ready for fusion, postpone by extending the timer with a random extra time within F phase.
        sheet.face_df.loc[F_cell, 'timer'] += extra_time
        return None
    # If we reach here, it means the geometry is ready for fusion. Do full geometric operation to fuse the cell.
    stb_face = sheet.edge_df.loc[sse, 'face']
    stbv, fv = identify_edge_endpoints(sheet, F_cell, sse)
    base_split(sheet, stbv, stb_face, sheet.edge_df[sheet.edge_df['face'] == stb_face], epsilon=1, recenter=True)
    new_edge = sheet_split(sheet, fv, F_cell)[0]
    new_edge = type1_transition(sheet, new_edge, do_reindex=True, remove_tri_faces=False, multiplier=5)
    sheet.face_df.loc[F_cell, 'cell_class'] = 'STB'
    geom.update_all(sheet)
    return new_edge

"""cell fusion behaviour function

A cell fusion behaviour function is used when a CT is fusing into the STB layer. The cell class of the selection cell 
should become "STB" at the end of the function.
First, the shared STB edges that share a vertex on the F cells identified.

Secondly, split STB shared vertices that on the outer surface and create separated edges.

Thirdly, split vertex shared by STBs and CT, creating a new CT edge.

Fourthly, F class cell transition to STB class.

Lastly, new dynamic parameters need to be updated to ensure consistent physics rule.
"""
def fusion(sheet, manager, geom, unique_id, tau_F_min, tau_F_max):
    """
    Event‑manager‑compatible fusion behaviour for a single CT cell in class 'F'.

    A cell in fusion phase ('F') attempts to merge into the STB layer.
    Fusion only proceeds when the local geometry is ready.
    If not ready, the fusion is postponed by extending the cell's timer.
    """
    idx = sheet.idx_lookup(unique_id, 'face')
    # If the cell is no longer in class F, stop scheduling fusion.
    if sheet.face_df.loc[idx, "cell_class"] != "F":
        return

    # Check whether a valid STB–STB interface is available.
    sse = find_local_stb_stb_edge(sheet, idx)
    # If geometry is NOT ready, postpone fusion.
    if sse is None:
        extra_time = round(rng.uniform(tau_F_min, tau_F_max), 4)
        sheet.face_df.loc[idx, "timer"] += extra_time
        # Re‑schedule fusion for this cell.
        manager.append(
            fusion,
            geom=geom,
            unique_id=unique_id,
            tau_F_min=tau_F_min,
            tau_F_max=tau_F_max
        )
        return
    # Geometry is ready. Identify the correct vertices.
    stb_face = sheet.edge_df.loc[sse, "face"]
    stb_vertex, f_vertex = identify_edge_endpoints(sheet, idx, sse)
    # Perform the geometric merge.
    base_split(
        sheet,
        stb_vertex,
        stb_face,
        sheet.edge_df[sheet.edge_df["face"] == stb_face],
        epsilon=1,
        recenter=True
    )
    new_edge = sheet_split(sheet, f_vertex, idx)[0]
    type1_transition(
        sheet,
        new_edge,
        do_reindex=True,
        remove_tri_faces=False,
        multiplier=5
    )
    # Update the cell class.
    sheet.face_df.loc[idx, "cell_class"] = "STB"
    # Refresh geometry.
    geom.update_all(sheet)
    print(f"Cell #{unique_id} has fused into the STB layer.")


"""cell extrusion behaviour function

A cell extrusion behaviour function has two components.

The first component is to let the STB to detach from the CT layer. This is done by a series of T1 transition on shared
edges between the selected STB and CTs: Detach STB.

The second component is to let the STB to shed from the layer. The shedding is modelled by cell removal but keep the
vertices shared between STB units that are still in the system: STB removal.
"""

def stb_detach(sheet, geom, cell_id):
    if cell_id not in sheet.face_df.index:
        return
    sheet.get_extra_indices()
    while True:
        internal_edges = sheet.edge_df[(sheet.edge_df['face'] == cell_id) & (sheet.edge_df['opposite'] != -1)]
        did_t1 = False
        for edge_id in internal_edges.index:
            opposite_edge_id = internal_edges.loc[edge_id, 'opposite']
            opposite_cell = sheet.edge_df.loc[opposite_edge_id, 'face']
            if sheet.face_df.loc[opposite_cell, 'cell_class'] == 'STB' or sheet.face_df.loc[opposite_cell, 'cell_class'] == 'E':
                continue
            else:
                print(f'processing edge {edge_id} for detachment of cell {cell_id}. ')
                collapse_edge(sheet, edge_id, reindex=True)
                geom.update_all(sheet)
                sheet.reset_index(order=False)
                did_t1 = True
                break
        if not did_t1:
            break

def face_boundary_edges(sheet, face_id):
    """Return all boundary edges belonging to a given face."""
    return sheet.edge_df[
        (sheet.edge_df['face'] == face_id) &
        (sheet.edge_df['opposite'] == -1)
    ].index.tolist()


def stb_extrusion(sheet, cell_id):
    if cell_id not in sheet.face_df.index:
        return
    while True:
        boundary_edges = face_boundary_edges(sheet, cell_id)
        if len(boundary_edges) == 0:
            break
        edge_id = boundary_edges[0]
        if edge_id not in sheet.edge_df.index:
            break
        collapse_edge(sheet, edge_id, reindex=False)
    sheet.reset_index(order=False)


def extrude(sheet, manager, geom, unique_id):
    idx = sheet.idx_lookup(unique_id,'face')
    remove.face(sheet,idx)
    geom.update_all(sheet)

def detach(sheet, manager, geom, unique_id):
    sheet.get_extra_indices()
    idx = sheet.idx_lookup(unique_id,'face')
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
