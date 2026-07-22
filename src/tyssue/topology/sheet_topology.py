import logging
import warnings

import numpy as np
import pandas as pd

from .base_topology import add_vert, close_face, collapse_edge, remove_face, v_e_distance, v_v_distance, third_mutual_vertex
from .base_topology import split_vert as base_split_vert

logger = logging.getLogger(name=__name__)
MAX_ITER = 100


def split_vert(
    sheet, vert, face=None, multiplier=1.5, reindex=True, recenter=False, epsilon=None
):
    """Splits a vertex towards the center of the face.

    This operation removes the  face `face` from the neighborhood of the vertex.

    Returns a list of the new edge's indices in edge_df. This should be two:
        the edge with vert as the srce and the newly created vertex as the trgt
        and the reverse edge with vert as the trgt and the new one as the srce
    """
    # Get the value for the length of the new edge
    if epsilon is None:
        epsilon = sheet.settings.get("threshold_length", 0.1) * multiplier
    else:
        warnings.warn(
            "The epsilon argument is deprecated and will be removed"
            " in a future version. "
            "The length of the new edge should be set by "
            "`sheet.settings['threshold_length]*multiplier` "
        )
    if face is None:
        face = np.random.choice(sheet.edge_df[sheet.edge_df["srce"] == vert]["face"])

    face_edges = sheet.edge_df.query(f"face == {face}")     # A filerted view of the edge_df, contains only the edges belonging to the given face.
    (prev_v,) = face_edges[face_edges["trgt"] == vert]["srce"]  # The vertex connects into vert (the sources of the edges that whose target is vert).
    (next_v,) = face_edges[face_edges["srce"] == vert]["trgt"]  # The vertex connects out of vert (the targets of the edges whose source is vert).
    # A filtered view of the edge_df, contains all edges that touch either prev_v or next_v, regardless of face.
    connected = sheet.edge_df[
        sheet.edge_df["trgt"].isin((next_v, prev_v))
        | sheet.edge_df["srce"].isin((next_v, prev_v))
    ]
    # pass the filtered subset of edge_df as connected to rewire.
    base_split_vert(sheet, vert, face, connected, epsilon, recenter)
    new_edges = []
    for face_ in connected["face"]:
        new_edge = close_face(sheet, face_)
        if new_edge is not None:
            new_edges.append(new_edge)

    if reindex:
        sheet.reset_index()
        sheet.reset_topo()

    return new_edges


def type1_transition(sheet, edge01, *, do_reindex =True, remove_tri_faces=True, multiplier=1.5):
    """Performs a type 1 transition around the edge edge01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings
    See Finegan et al. for a description of the algotithm https://doi.org/10.1101/704932


    Parameters
    ----------
    sheet : a `Sheet` instance
    edge_01 : int
       index of the edge around which the transition takes place
    do_reindex : bool, optional
        whether or not to reindex  the sheet.
    remove_tri_faces : bool, optional
       if True (the default), will remove triangular cells
       after the T1 transition is performed
    multiplier : float, optional
       default 1.5, the multiplier to the threshold length, so that the
       length of the new edge is set to multiplier * threshold_length

    """

    srce, trgt, face = sheet.edge_df.loc[edge01, ["srce", "trgt", "face"]].astype(int)

    vert = min(srce, trgt)  # find the vertex that won't be reindexed
    ret_code = collapse_edge(sheet, edge01, reindex=do_reindex, allow_two_sided=True)
    if ret_code < 0:
        warnings.warn(f"Collapse of edge {edge01} failed")
        return ret_code

    split_vert(
        sheet,
        vert,
        face,
        multiplier=multiplier,
        reindex=do_reindex,
        recenter=True,
    )

    if not remove_tri_faces:
        return 0
    # Type 1 transitions might create 3 or 2 sided cells, we remove those
    tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
    i = 0
    while len(tri_faces):
        remove_face(sheet, tri_faces[0])
        tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
        i += 1
        if i > MAX_ITER:
            raise RecursionError
    return 0


def cell_division(sheet, mother, geom, angle=None):
    """Causes a cell to divide

    Parameters
    ----------

    sheet : a 'Sheet' instance
    mother : face index of target dividing cell
    geom : a 2D geometry
    angle : division angle for newly formed edge

    Returns
    -------
    daughter: face index of new cell

    Notes
    -----
    - Function checks for perodic boundaries if there are, it checks if dividing cell
      rests on an edge of the periodic boundaries if so, it displaces the boundaries
      by a half a period and moves the target cell in the bulk of the tissue. It then
      performs cell division normally and reverts the periodic boundaries
      to the original configuration
    """

    if sheet.settings.get("boundaries") is not None:
        mother_on_periodic_boundary = False
        if (
            sheet.face_df.loc[mother]["at_x_boundary"]
            or sheet.face_df.loc[mother]["at_y_boundary"]
        ):
            mother_on_periodic_boundary = True
            saved_boundary = sheet.specs["settings"]["boundaries"].copy()
            for u, boundary in sheet.settings["boundaries"].items():
                if sheet.face_df.loc[mother][f"at_{u}_boundary"]:
                    period = boundary[1] - boundary[0]
                    sheet.specs["settings"]["boundaries"][u] = [
                        boundary[0] + period / 2.0,
                        boundary[1] + period / 2.0,
                    ]
            geom.update_all(sheet)

    if not sheet.face_df.loc[mother, "is_alive"]:
        logger.warning("Cell %s is not alive and cannot devide", mother)
        return
    edge_a, edge_b = get_division_edges(sheet, mother, geom, angle=angle, axis="x")
    if edge_a is None:
        return

    vert_a, *_ = add_vert(sheet, edge_a)
    vert_b, *_ = add_vert(sheet, edge_b)
    sheet.vert_df.index.name = "vert"
    daughter = face_division(sheet, mother, vert_a, vert_b)

    if sheet.settings.get("boundaries") is not None and mother_on_periodic_boundary:
        sheet.specs["settings"]["boundaries"] = saved_boundary
        geom.update_all(sheet)
    return daughter


def get_division_edges(sheet, mother, geom, angle=None, axis="x"):

    if angle is None:
        angle = np.random.random() * np.pi

    m_data = sheet.edge_df[sheet.edge_df["face"] == mother]
    rot_pos = geom.face_projected_pos(sheet, mother, psi=angle)

    srce_pos = rot_pos.loc[m_data["srce"], axis]
    srce_pos.index = m_data.index
    trgt_pos = rot_pos.loc[m_data["trgt"], axis]
    trgt_pos.index = m_data.index
    try:
        edge_a = m_data[(srce_pos < 0) & (trgt_pos >= 0)].index[0]
        edge_b = m_data[(srce_pos >= 0) & (trgt_pos < 0)].index[0]
    except IndexError:
        print("Failed")
        logger.error("Division of Cell {} failed".format(mother))
        return None, None
    return edge_a, edge_b


def face_division(sheet, mother, vert_a, vert_b):
    """
    Divides the face associated with edges
    indexed by `edge_a` and `edge_b`, splitting it
    in the middle of those edes.
    """
    # mother = sheet.edge_df.loc[edge_a, 'face']

    face_cols = sheet.face_df.loc[mother:mother]

    daughter = int(sheet.add_element('face'))

    new_edge_m = sheet.add_element('edge')
    sheet.edge_df.loc[new_edge_m, "srce"] = vert_b
    sheet.edge_df.loc[new_edge_m, "trgt"] = vert_a

    new_edge_d = sheet.add_element('edge')
    sheet.edge_df.loc[new_edge_d, "srce"] = vert_a
    sheet.edge_df.loc[new_edge_d, "trgt"] = vert_b

    sheet.edge_df.loc[new_edge_m, "face"] = mother
    sheet.edge_df.loc[new_edge_d, "face"] = mother

    # ## Discover daughter edges
    m_data = sheet.edge_df[sheet.edge_df["face"] == mother]
    daughter_edges = [new_edge_d]
    srce, trgt = vert_a, vert_b
    srces, trgts = m_data[["srce", "trgt"]].values.T
    spins = 0

    while trgt != vert_a:
        srce, trgt = trgt, trgts[srces == trgt][0]

        daughter_edges.append(
            m_data[(m_data["srce"] == srce) & (m_data["trgt"] == trgt)].index[0]
        )
        spins += 1
        if spins > m_data.shape[0]:
            raise ValueError(f"The face {mother} has an invalid topology, \n")
    sheet.edge_df.loc[daughter_edges, "face"] = daughter
    sheet.edge_df.index.name = "edge"
    sheet.reset_topo()
    return daughter


def drop_face(eptm, face, geom, **kwargs):
    """
    Removes the face indexed by "face" and all associated edges to allow holes
    """
    edge = eptm.edge_df.loc[(eptm.edge_df['face'] == face)].index

    eptm.remove(edge, **kwargs)
    eptm.sanitize(trim_borders = True)
    geom.update_all(eptm)


def resolve_t1s(sheet, geom, model, solver, max_iter=60):

    l_th = sheet.settings["threshold_length"]
    i = 0
    while sheet.edge_df.length.min() < l_th:

        for edge in (
            sheet.edge_df[sheet.edge_df.length < l_th].sort_values("length").index
        ):
            try:
                type1_transition(sheet, edge)
            except KeyError:
                continue
            sheet.reset_index()
            sheet.reset_topo()
            geom.update_all(sheet)
        solver.find_energy_min(sheet, geom, model)
        i += 1
        if i > max_iter:
            break


def _cast_to_int(df_value):

    if len(df_value) == 1:
        return int(df_value)
    elif len(df_value) == 0:
        return -1
    else:
        raise ValueError("Trying to retrieve an integer from a more than length 1 df ")



def boundary_ids(sheet):
    """
    Takes an edge_df, creates a tuple of boundary edge indices and a tuple of boundary vertices indices.
    The function returns a tuple of the two indices tuples.
    First we create a set of boundary elements, then we convert them back to a list for operations in T3
    """
    boundary_edges = sheet.edge_df[sheet.edge_df['opposite'] == -1].index
    boundary_edge_ids = list(set(sheet.edge_df.loc[boundary_edges, "unique_id"].tolist()))
    boundary_verts = sheet.edge_df.loc[boundary_edges,'srce'].values
    boundary_vert_ids = list(set(sheet.vert_df.loc[boundary_verts,"unique_id"].tolist()))
    return boundary_edge_ids, boundary_vert_ids

def edge_uid_to_pos(sheet, uid):
    return sheet.edge_df.index[sheet.edge_df["unique_id"] == uid][0]

def vert_uid_to_pos(sheet, uid):
    return sheet.vert_df.index[sheet.vert_df["unique_id"] == uid][0]


def T3_transition(eptm,boundary_vertices, boundary_edges, length_threshold, multiplier):
    """
    This is a funtion that does T3 transition on boundary nodes and boundary edges.

    eptm: eptm object
    boundary_nodes: a tuple of boundary vertices
    boundary_edges: a tuple of boundary edges
    length_threshold: minimum length to triggers the transition
    multiplier: multiplier used to set the separation distance between edges and vertices
    """
    # Initiate a minimum separation distance variable for later use.
    d_sep = length_threshold * multiplier
    # Creates a df that acts as a table of each vertex: all the faces associate to that vertex.
    faces_by_srce = eptm.edge_df.groupby('srce')['face'].apply(list)

    # Compute the distance between each boundary edge and boundary vertices. Record them as two lists.
    # One list is the edge_vertex list that means we need to deal with a boundary vertex-edge case,
    # Another list is the vertex_vertex list that means we need to deal with a boundary vertex-vertex case.
    vertex_edge_pairs = []
    vertex_vertex_pairs = []
        # Compute vertex-edge distances, we use positional id (pid) for computing distance, but use uid to track items.
    for first_vertex_uid in boundary_vertices:
        first_vertex_pid = vert_uid_to_pos(eptm, first_vertex_uid)
        for first_edge_uid in boundary_edges:
            # convert to positional id in df.
            first_edge_pid = edge_uid_to_pos(eptm, first_edge_uid)
            # Utilize the faces_by_srce df, skip checking if the vertex and edge are from the same face.
            if eptm.edge_df.loc[first_edge_pid,'face'] in faces_by_srce[first_vertex_pid]:
                continue
            else: # Go ahead with the process if vertex and edge are not in the same face.
                distance, collision_point = v_e_distance(eptm, first_edge_pid, first_vertex_pid, d_sep)
                if distance < length_threshold:
                    vertex_edge_pairs.append((first_vertex_uid, first_edge_uid, collision_point))
                    boundary_vertices.remove(first_vertex_uid)
                else:
                    continue

        # Compute vertex-vertex distances for the rest of the boundary vertices.
    for i, first_vertex_uid in enumerate(boundary_vertices):
        first_vertex_pid = vert_uid_to_pos(eptm, first_vertex_uid)
        for second_vertex_uid in boundary_vertices[i + 1:]:
            # convert to positional id in the df
            second_vertex_pid = vert_uid_to_pos(eptm, second_vertex_uid)
            # Based on their positional id, and the faces_by_srce df, skip checking vertices from the same face.
            common_v = [x for x in faces_by_srce[first_vertex_pid] if x in faces_by_srce[second_vertex_pid]]
            if not common_v: # go ahead if the common_v is an empty list. 'an empty list has a False boolean value'
                distance = v_v_distance(eptm, first_vertex_pid, second_vertex_pid)
                if distance < length_threshold:
                    vertex_vertex_pairs.append((first_vertex_uid, second_vertex_uid))
            else: # if the common_v is not an empty list, then they are vertices of a face, skip checking.
                continue
    print('vertex_edge_pairs: ', vertex_edge_pairs)
    print('vertex_vertex_pairs: ', vertex_vertex_pairs)
    # We first look at the vertex_vertex list, there are two cases.
    # Case 1: if the pair shares mutual vertex, that means they are from two adjacent cells,
    # we need to extend the existing edge by creating a new vertex at the mid-point between vert-vert pair, and use
    # the new vertex to replace the two old one.
    # Case 2: when vert-vert pair doesn't share a mutual edge, we should do a preemptive T1 transition.
    for pairs in vertex_vertex_pairs:
        first_vertex_uid = pairs[0]
        second_vertex_uid = pairs[1]
        first_vertex_pid = vert_uid_to_pos(eptm, first_vertex_uid)
        second_vertex_pid = vert_uid_to_pos(eptm, second_vertex_uid)
        if third_mutual_vertex(eptm, first_vertex_pid, second_vertex_pid):
            # Case 1 situation, first add a new vertex in the vert_df with middle location between two vertices. Then update edge_df.
            new_vert = eptm.add_element('vert')
            eptm.vert_df.loc[new_vert, eptm.coords] = eptm.vert_df.loc[[first_vertex_pid, second_vertex_pid], eptm.coords].mean(numeric_only=True)
            # Then update edge_df.
            eptm.edge_df.replace({"srce": first_vertex_pid, "trgt": first_vertex_pid}, new_vert, inplace=True)
            eptm.edge_df.replace({"srce": second_vertex_pid, "trgt": second_vertex_pid}, new_vert, inplace=True)
        else:
            # Case 2 situation, perform preemptive T1 transition.
            imaginary_line = eptm.vert_df.loc[second_vertex_pid,eptm.coords] - eptm.vert_df.loc[first_vertex_pid,eptm.coords]
            imaginary_line_unit = imaginary_line/np.linalg.norm(imaginary_line)
            perpendicular_unit = pd.Series({'x': imaginary_line_unit['y'],'y': -imaginary_line_unit['x']})
                # Compute the coordinates of the midpoint.
            imaginary_line_midpoint = eptm.vert_df.loc[[first_vertex_pid, second_vertex_pid], eptm.coords].mean(numeric_only=True)
                # Add two new rows in the vert_df
            new_vert_1 = eptm.add_element('vert')
            eptm.vert_df.loc[new_vert_1, eptm.coords] = imaginary_line_midpoint - d_sep * perpendicular_unit

            # create a new vert 2.
            new_vert_2 = eptm.add_element('vert')
            eptm.vert_df.loc[new_vert_2, eptm.coords] = imaginary_line_midpoint + d_sep*perpendicular_unit
                # Update the edge dataframe.
            eptm.edge_df.replace({"srce": first_vertex_pid, "trgt": first_vertex_pid}, new_vert_1, inplace=True)
            eptm.edge_df.replace({"srce": second_vertex_pid, "trgt": second_vertex_pid}, new_vert_2, inplace=True)

    # Lastly we deal with the edge-vertex pairs. For each pair, we use the closest point as an imaginary point, then create
    # two vertices that are each d_sep away from the imaginary collision point.
    for pairs in vertex_edge_pairs:
        incoming_vertex_pid = vert_uid_to_pos(eptm, pairs[0])
        collide_edge_pid = edge_uid_to_pos(eptm, pairs[1])
        collision_coord = pairs[2]
        # From the collision point coordiates, computes the coordinates for the two new vertices.
        # Utilize the edge dataframe, for each edge, 'ux,uy' column is the unit vector from srce to trgt.
        srce_trgt_unit_vector = eptm.edge_df.loc[collide_edge_pid,["ux","uy"]]
        # Extract all rows of edges that has either srce or trgt as the incoming vertex.
        connected = eptm.edge_df[(eptm.edge_df["trgt"] == incoming_vertex_pid)|(eptm.edge_df["srce"] == incoming_vertex_pid)]
        connected_index = connected.index
        # Compute the correct coordinates to the new vertices.
        new_vert1_coord = collision_coord - d_sep * srce_trgt_unit_vector
        new_vert2_coord = collision_coord + d_sep * srce_trgt_unit_vector
            # Add two new vertices on the collding edge
        new_vert_1, new_edge1, new_oedge_1 = add_vert(eptm, collide_edge_pid, new_vert1_coord)
        new_vert_2, new_edge2, new_oedge_2 = add_vert(eptm, collide_edge_pid, new_vert2_coord)
        # Rewire the edges based on the extracted index previously.
        # The first asscoiated vertex is reconnected to the new vertex 1, all the rest reconnects to the new vertex 2.
        # Note: for loc and iloc, double square bracket returns a new dataframe, single bracket gives a series.
        first = connected.iloc[[0]].replace(
            {"srce": incoming_vertex_pid, "trgt": incoming_vertex_pid}, new_vert_1
        )
        rest = connected.iloc[1:].replace(
            {"srce": incoming_vertex_pid, "trgt": incoming_vertex_pid}, new_vert_2
        )
        eptm.edge_df.loc[connected_index[0]] = first.iloc[0]
        eptm.edge_df.loc[connected_index[1:]] = rest







