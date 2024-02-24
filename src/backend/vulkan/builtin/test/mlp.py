import numpy as np
from pathlib import Path


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


if __name__ == "__main__":
    generator = np.random.Generator(np.random.PCG64(seed=0))

    width = 64
    hidden_layers = 0
    batch_size = 128

    input = generator.standard_normal(size=[width, batch_size]).astype(np.float16)

    win = generator.standard_normal(size=[width, width]).astype(np.float16)
    wout = generator.standard_normal(size=[width, width]).astype(np.float16)

    # weights = [win, wout]
    weights = [wout]

    H = input
    for W in weights[:-1]:
        H = H @ np.transpose(W, [1, 0])
        H = relu(H)

    H = weights[-1] @ H

    output = H

    weights = np.stack(weights)

    print(f"{input.reshape([-1]).shape=}")
    datapath = Path(__file__).parent / "data"
    datapath.mkdir(exist_ok=True, parents=True)
    weights.reshape([-1]).tofile(datapath / "weights.bin")
    input.reshape([-1]).tofile(datapath / "input.bin")
    output.reshape([-1]).tofile(datapath / "output.bin")

    print(f"{output.reshape([-1])}")
