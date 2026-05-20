""" The function in this script auto-controls the edges are active or not for a bilayer tissue sheet."""

from ...config.draw import sheet_spec

def auto_dummy_edges(sheet):
    """
    Update edge activity based on cell classes.
    STB-like (STB or E) on both sides are set inactive (dummy edge).
    Boundary edges and mixed-class edges remain active.
    """
    sheet.get_extra_indices()
    # STB-like predicate
    def is_stb_like(cell_class):
        return cell_class in ("STB", "E")

    for i in sheet.edge_df.index:
        opp = sheet.edge_df.loc[i, "opposite"]
        # Boundary edges are always active
        if opp == -1 or opp not in sheet.edge_df.index:
            sheet.edge_df.loc[i, "is_active"] = 1
            continue
        # Faces on each side
        f1 = sheet.edge_df.loc[i, "face"]
        f2 = sheet.edge_df.loc[opp, "face"]
        # If faces disappeared during topology changes then keep active
        if f1 not in sheet.face_df.index or f2 not in sheet.face_df.index:
            sheet.edge_df.loc[i, "is_active"] = 1
            sheet.edge_df.loc[opp, "is_active"] = 1
            continue
        c1 = sheet.face_df.loc[f1, "cell_class"]
        c2 = sheet.face_df.loc[f2, "cell_class"]

        # STB-like on both sides then deactivate
        if is_stb_like(c1) and is_stb_like(c2):
            sheet.edge_df.loc[i, "is_active"] = 0
            sheet.edge_df.loc[opp, "is_active"] = 0
        else:
            sheet.edge_df.loc[i, "is_active"] = 1
            sheet.edge_df.loc[opp, "is_active"] = 1
    print("Dummy edges updated (STB and E treated identically).")

def update_draw_specs(sheet):
    """
    Update drawing specifications for faces and edges based on cell class and edge activity.

    Returns:
        draw_specs (dict): ready to pass into sheet_view(...)
    """

    draw_specs = sheet_spec()
    draw_specs['face']['visible'] = True
    # Assign color by cell class
    sheet.face_df['color'] = sheet.face_df['cell_class'].map(
        lambda c: 0.7 if c == 'STB' else 0.1
    )
    draw_specs['face']['color'] = sheet.face_df['color']
    draw_specs['face']['alpha'] = 0.2

    # Edge appearance
    draw_specs['edge']['visible'] = True
    sheet.edge_df['width'] = sheet.edge_df['is_active'].map(
        lambda active: 0.5 if active == 1 else 2
    )
    draw_specs['edge']['width'] = sheet.edge_df['width']

    return draw_specs

def deactivate_cells(sheet, id_list):
    """
    Deactivate specific cells (e.g., corner cells) so they do not
    participate in energy minimisation. Also deactivate their vertices.
    """
    for cell_index in id_list:
        # Skip if the face no longer exists (after topology changes)
        if cell_index not in sheet.face_df.index:
            continue

        # Mark the cell as dead
        sheet.face_df.loc[cell_index, "is_alive"] = 0
        # Find all vertices belonging to edges of this cell
        edges = sheet.edge_df[sheet.edge_df["face"] == cell_index]
        for v in edges["srce"].tolist():
            if v in sheet.vert_df.index:
                sheet.vert_df.loc[v, "is_active"] = 0
