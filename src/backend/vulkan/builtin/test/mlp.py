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
    # W = np.identity(width).astype(np.float16)
    # W[0][1] = 1
    # W[0][0] = 0
    # W = W

    weights = [wout]

    H = input

    for W in weights:
        H = W @ H

    output = H
    print(f"{input[:, 0]=}")
    print(f"{output[:, 0]=}")

    weights = np.stack(weights)

    input = np.transpose(input, [1, 0])
    input = input.reshape([-1])

    output = np.transpose(output, [1, 0])
    output = output.reshape([-1])

    weights = weights.reshape([-1])

    print(f"{input[:64]=}")
    print(f"{output[:64]=}")
    print(f"{weights[:2]=}")

    print(f"{input.reshape([-1]).shape=}")
    datapath = Path(__file__).parent / "data"
    datapath.mkdir(exist_ok=True, parents=True)
    weights.tofile(datapath / "weights.bin")
    input.tofile(datapath / "input.bin")
    output.tofile(datapath / "output.bin")

    print(f"{output.reshape([-1])}")
