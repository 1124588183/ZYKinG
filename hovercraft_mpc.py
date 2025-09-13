import numpy as np


def hovercraft_matrices(dt: float):
    """Return discrete-time state-space matrices for a simple hovercraft model."""
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    B = np.array([[0.5 * dt ** 2, 0],
                  [0, 0.5 * dt ** 2],
                  [dt, 0],
                  [0, dt]])
    return A, B


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int):
    """Finite-horizon LQR to compute MPC feedback gains."""
    P = Q.copy()
    Ks = []
    for _ in range(N):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Ks.append(K)
        P = Q + A.T @ P @ (A - B @ K)
    Ks.reverse()
    return Ks


def simulate(path, dt=0.1, horizon=10):
    A, B = hovercraft_matrices(dt)
    Q = np.diag([10, 10, 1, 1])
    R = np.diag([0.1, 0.1])
    Ks = lqr(A, B, Q, R, horizon)

    x = np.zeros(4)
    states = [x.copy()]
    for i in range(len(path) - 1):
        refs = []
        for k in range(horizon + 1):
            idx = min(i + k, len(path) - 1)
            refs.append(np.array([path[idx][0], path[idx][1], 0, 0]))
        refs = np.array(refs)
        u_refs = []
        for k in range(horizon):
            u_ref = np.linalg.lstsq(B, refs[k + 1] - A @ refs[k], rcond=None)[0]
            u_refs.append(u_ref)
        u = u_refs[0] - Ks[0] @ (x - refs[0])
        x = A @ x + B @ u
        states.append(x.copy())
    return np.array(states)


def main():
    path = [(i * 0.5, i * 0.5) for i in range(21)]
    states = simulate(path)
    print("Final position:", states[-1][:2])


if __name__ == "__main__":
    main()
