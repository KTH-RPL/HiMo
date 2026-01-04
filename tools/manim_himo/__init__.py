import numpy as np
MAX_RAY_LENGTH = 10000
def get_line_intersection(p1, p2, p3, p4):
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None  # The lines are parallel or coincident

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    if (
        min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])
        and min(p1[1], p2[1]) <= y <= max(p1[1], p2[1])
        and min(p3[0], p4[0])-0.01 <= x <= max(p3[0], p4[0])+0.1
        and min(p3[1], p4[1])-0.01 <= y <= max(p3[1], p4[1])+0.1
    ):
        return np.array([x, y, 0])
    else:
        return None
    
def sort_angle(rays, dots, angles):
    # Assuming rays, dots, and angles are already populated
    combined = list(zip(angles, rays, dots))

    # Sort combined list based on the angle
    combined.sort(key=lambda x: x[0])

    # Unpack the sorted tuples back into individual lists
    angles, rays, dots = zip(*combined)
    
    return list(rays), list(dots), list(angles)