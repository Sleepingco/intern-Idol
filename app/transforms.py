import numpy as np
from utils import rebuild_hat_base_O


def calculate_hat_transform_matrix(M_smooth):
    if M_smooth is None:
        return None
    O_hat_base = rebuild_hat_base_O()
    return (M_smooth @ np.diag([1, 1, -1, 1]).astype(np.float32) @ O_hat_base).astype(np.float32)

