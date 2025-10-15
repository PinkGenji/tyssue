"""
Event module for cell class transition rules, modify the details accordingly to your model.
=======================

"""

import numpy as np
from ...geometry.planar_geometry import PlanarGeometry
from ...topology.sheet_topology import cell_division

def cell_cycle_transition(sheet, manager, dt, p_recruit=0.1, G2_duration=0.4, G1_duration=0.11):
    """
    Controls cell class state transitions for cell cycle based on timers and probabilities.

    Parameters
    ----------
    sheet: tyssue.Sheet
        The tissue sheet.
    manager: EventManager
        The event manager scheduling the behaviour.
    face_id: Integer
        ID of the cell being controlled.
    p_recruit: float
        Probability for an 'S' cell to be recruited to 'G2'.
    dt: float
        Time step increment.
    G2_duration: float
        Fixed duration cells stay in G2 phase.
    G1_duration: float
        Fixed duration cells stay in G1 phase.
    """
    # Generate two df that are cells that need to change its cell class or stay in its current class.
    cells_stay_in_class = sheet.face_df.loc[sheet.face_df['timer'] > 0].index.tolist()
    cells_to_change = sheet.face_df.loc[sheet.face_df['timer'] <= 0]

    # For cells that still needs to elapse the timer, just reduce the timer by dt
    for cell in cells_stay_in_class:
        sheet.face_df.loc[cell,'timer'] -= dt

    # Then we change the cell type based on the cell cycle diagram.
    G1_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'G1'].index.tolist()
    S_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'S'].index.tolist()
    G2_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'G2'].index.tolist()
    M_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'M'].index.tolist()

    # For cells in G1_cells indices, we simply change the cell class to S
    sheet.face_df.loc[G1_cells, 'cell_class'] = 'S'

    # For cells in S_cells , we need to loop and decide based on the random number generated for if it moves into G2.
    for cell in S_cells:
        if np.random.rand() < p_recruit:
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell, 'timer'] = G2_duration

    # For cells in G2_cells, we move them into M class
    sheet.face_df.loc[G2_cells, 'cell_class'] = 'M'

    # For cells in M_cells, they need to be undergone cell division
    for cell in M_cells:
        daughter = cell_division(sheet, mother=cell, geom = PlanarGeometry )
        # Set parent and daughter to G1 with G1 timer, note that variable daughter is the index of the new row already.
        sheet.face_df.loc[cell, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[cell, 'timer'] = G1_duration
        sheet.face_df.loc[daughter, 'timer'] = G1_duration

    manager.append(cell_cycle_transition, dt = dt, p_recruit = p_recruit,G2_duration = G2_duration, G1_duration = G1_duration)
