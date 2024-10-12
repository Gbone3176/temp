import time
from functools import wraps

def time_it_withwrap(func):
    @wraps(func)  # 保留原函数的信息，比如函数名称和文档字符串
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result  # 返回原函数的返回值
    return wrapper

def time_it_withoutwrap(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result  # 返回原函数的返回值
    return wrapper

@time_it_withwrap
def example_function1():
    time.sleep(2)  # 模拟一个运行时间为 2 秒的操作

@time_it_withoutwrap
def example_function2():
    time.sleep(2)  # 模拟一个运行时间为 2 秒的操作

example_function1()
example_function2()

print(example_function1.__name__)
print(example_function2.__name__) # 如果不使用@wrap装饰器，被装饰的函数的名字就会变成wrapper
print(example_function1.__doc__)
print(example_function2.__doc__)
