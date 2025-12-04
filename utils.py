import time


def time_it(label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{label}: {end - start:.4f} seconds")
    return result

