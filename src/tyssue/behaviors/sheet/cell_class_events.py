"""
Event module for cell class transition rules, modify the details accordingly to your model.
=======================

"""

import numpy as np
from ...geometry.planar_geometry import PlanarGeometry
from ...topology.sheet_topology import cell_division

def cell_cycle_transition(sheet, manager, dt, face_id, p_recruit=0.1, G2_duration=0.4, G1_duration=0.11):
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
    # First, we need to look up the current index of the face with face ID.
    idx = sheet.idx_lookup(face_id, "face")
    # Record the current cell class
    current_class = sheet.face_df.loc[idx, 'cell_class']
    # (1) Recruit mature 'S' cells into G2 with probability p_recruit
    if current_class == 'S':
        if np.random.rand() < p_recruit:
            sheet.face_df.loc[idx, 'cell_class'] = 'G2'
            sheet.face_df.loc[idx, 'timer'] = G2_duration
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, face_id=face_id)

    # (2) Decrement timers for cells in G2; when timer ends, move to M
    elif current_class == 'G2':
        sheet.face_df.loc[idx, 'timer'] -= dt
        if sheet.face_df.loc[idx, 'timer'] <= 0:
            sheet.face_df.loc[idx, 'cell_class'] = 'M'
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, face_id=face_id)

    # (3) For cells in M, perform division and set daughters to G1 with timer
    elif current_class == 'M':
        # Make sure we pass the variable idx for cell division.
        daughter = cell_division(sheet, mother=idx, geom = PlanarGeometry )
        # Set parent and daughter to G1 with G1 timer, note that variable daughter is the index of the new row already.
        sheet.face_df.loc[idx, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[idx, 'timer'] = G1_duration
        sheet.face_df.loc[daughter, 'timer'] = G1_duration
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, face_id=face_id)
        # look up the unique id of daughter, then append.
        daughter_id = sheet.face_df.loc[daughter, 'unique_id']
        print(f'daughter index {daughter} with unique ID: {daughter_id} is generated....')
        manager.append(cell_cycle_transition, dt=dt, face_id=daughter_id)

    # (4) Decrement timers for G1 cells; when timer ends, move to S
    elif current_class == 'G1':
        sheet.face_df.loc[idx, 'timer'] -= dt
        if sheet.face_df.loc[idx, 'timer'] <= 0:
            sheet.face_df.loc[idx, 'cell_class'] = 'S'
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, face_id = face_id)






