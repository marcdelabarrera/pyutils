import time

def timeit(func):
    def wrapper(*args, **kwargs):
        function_call = f"{func.__name__}({', '.join([str(i) for i in args])})"
        print(f"Calling {function_call}", end='')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        if elapsed < 1:
            print(f"\r{function_call} took {elapsed*1000:.3f} milliseconds")
        elif elapsed < 60:
            print(f"\r{function_call} took {elapsed:.3f} seconds")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"\r{function_call} took {minutes} minutes and {seconds:.3f} seconds")
        return result
    return wrapper