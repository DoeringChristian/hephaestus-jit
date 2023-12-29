import numpy as np
import os

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")

dr.set_flag(dr.JitFlag.KernelHistory, True)


def compress(n: int) -> float:
    src = dr.ones(mi.Bool, n)
    dr.eval(src)

    dr.kernel_history_clear()

    indices = dr.compress(src)

    kernels = dr.kernel_history()
    duration = kernels[-1]["execution_time"]
    return duration


def prefix_sum(n: int) -> float:
    src = dr.ones(mi.UInt32, n)
    dr.eval(src)

    dr.kernel_history_clear()

    pfs = dr.prefix_sum(src)

    kernels = dr.kernel_history()
    duration = kernels[-1]["execution_time"]
    return duration


def test(f) -> tuple[np.ndarray, np.ndarray]:
    m = 100
    x = []
    y = []
    for i in range(10, 31):
        n = 2**i
        print(f"{n=}")
        x.append(n)
        duration = 0.0
        for i in range(m):
            duration += f(n)
            dr.flush_kernel_cache()
            dr.flush_malloc_cache()
        duration /= m
        y.append(duration)

    x = np.array(x)
    y = np.array(y) / 1000
    return (x, y)


def test_and_plot(name: str, f):
    x, y = test(f)

    throughput = x / y

    dir = f"out/{name}"
    os.makedirs(dir, exist_ok=True)

    plt.plot(x, y)
    plt.xlabel("n")
    plt.ylabel("s")
    plt.savefig(f"{dir}/timing.svg")
    plt.clf()
    plt.cla()

    plt.plot(x, throughput)
    plt.xlabel("n")
    plt.ylabel("throughput in n/s")
    plt.savefig(f"{dir}/throughput.svg")
    plt.clf()
    plt.cla()


if __name__ == "__main__":
    test_and_plot("compress", compress)
    test_and_plot("prefix_sum_u32", prefix_sum)
