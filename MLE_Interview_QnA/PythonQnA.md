# Python for DS/MLE Interview QnA
I'm preparing for Data Science & MLE interviews. I want to use this chat to go through Q&A flashcards only (no deep implementation). Focus on interview-style questions and concise, accurate answers.

## OOP concepts 

### What is encapsulation in Python?
- Encapsulation is the bundling of data (attributes) and methods that operate on that data into a single unit (a class).
- It also refers to **restricting direct access** to some of an object’s components, usually by naming conventions.

#### In Python:
- Prefixing with a single underscore `_var` is a **convention** indicating internal use.
- Prefixing with double underscores `__var` triggers **name mangling**, making access more difficult from outside the class.

#### Example:
```python
class MyClass:
    def __init__(self):
        self._internal = "protected"
        self.__private = "private"

obj = MyClass()
print(obj._internal)       # accessible, but intended as protected
print(obj._MyClass__private)  # name-mangled, not easily accessed
```

### What is inheritance in Python?
- Inheritance allows a class (child) to **inherit attributes and methods** from another class (parent).
- It promotes code reuse and supports hierarchical relationships.

#### Types:
- **Single Inheritance**: One child, one parent
- **Multiple Inheritance**: One child, multiple parents
- **Multilevel Inheritance**: Inheritance chain across multiple classes

#### Example:
```python
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):
        return "Bark"

d = Dog()
print(d.speak())  # Output: Bark
```


## Python basics

### What is the difference between a function and a method in Python?
- A **function** is an independent block of code defined using `def` and not tied to any object.  
- A **method** is a function that is associated with an object (usually defined within a class) and takes `self` or `cls` as the first parameter.

### What are the `self` and `cls` parameters in Python?
- `self` refers to the instance of the class and is used in **instance methods** to access or modify object attributes.
- `cls` refers to the class itself and is used in **class methods** to access or modify class-level data.

### What’s the difference between `@classmethod` and `@staticmethod`?
`@classmethod` receives the class (`cls`) as the first argument and can modify class state.  
`@staticmethod` receives no implicit first argument and behaves like a regular function inside the class.

#### Example:
```python
class Counter:
    count = 0  # class-level attribute

    def __init__(self):
        # instance method using `self`
        self.id = Counter.count
        Counter.increment()

    @classmethod
    def increment(cls):
        # class method using `cls`
        cls.count += 1
```

### What are decorators in Python?
- Decorators are functions that modify the behavior of other functions or methods without changing their code.
- They are often used for **logging**, **access control**, **timing**, and **caching**.

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```
```text
Before function call  
Hello!  
After function call
```

### What is a thread in Python?
- A thread is the smallest unit of execution within a process.
- Multiple threads in a process share the same memory and resources.
- Threads are useful for performing tasks concurrently, especially **I/O-bound operations**.
- In Python (CPython), due to the Global Interpreter Lock (GIL), threads cannot achieve true parallelism for CPU-bound tasks.

#### Common use cases:
- Handling multiple client connections in a server
- Performing background tasks (e.g., logging, downloading)
- Running I/O operations without blocking the main program

### What is the difference between multithreading and multiprocessing in Python?

- **Multithreading** uses multiple threads within a single process.
  - Threads share the same memory space.
  - Limited by the Global Interpreter Lock (GIL) in CPython, so it's not ideal for CPU-bound tasks.
  - Best suited for I/O-bound tasks (e.g., file or network operations).

- **Multiprocessing** uses separate processes.
  - Each process has its own Python interpreter and memory space.
  - Avoids the GIL, so it's better for CPU-bound tasks (e.g., data processing, model training).
  - More memory-intensive than threads.


#### Summary:
| Feature    | Multithreading            | Multiprocessing       |
|------------|---------------------------|-----------------------|
| Memory     | Shared                    | Separate              |
| GIL Impact | Yes (limited concurrency) | No (true parallelism) |
| Use Case   | I/O-bound tasks           | CPU-bound tasks       |
| Overhead   | Lower                     | Higher                |

```python
import threading

def task():
    print("Thread running")

t = threading.Thread(target=task)
t.start()
t.join()
```

### What are `async` and `await` in Python?
- `async` and `await` are used to write asynchronous, non-blocking code using coroutines.
- Useful for **I/O-bound** tasks like network calls, file operations, or APIs where waiting would otherwise block the program.

#### Key Concepts:
- `async def` defines a **coroutine function**.
- `await` pauses the coroutine until the awaited task completes.

#### Example:
```python
import asyncio

async def say_hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(say_hello())
```

### What is a simple example of a generator function in Python?
- A generator function uses the `yield` keyword to return values one at a time.
- It produces values lazily and maintains state between calls.

#### Example:
```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

gen = count_up_to(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
```

### What is the difference between list comprehensions and generator expressions in Python?
- **List comprehensions** return a full list in memory.
- **Generator expressions** return an iterator that yields items lazily (one at a time).

#### Key Differences:
| Feature      | List Comprehension         | Generator Expression              |
|--------------|----------------------------|-----------------------------------|
| Syntax       | `[x for x in iterable]`    | `(x for x in iterable)`           |
| Memory usage | Loads all items at once    | Generates items on demand         |
| Performance  | Faster for small data sets | Better for large or infinite data |
| Return type  | `list`                     | `generator`                       |

#### Example:
```python
# List comprehension
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# Generator expression
squares_gen = (x**2 for x in range(5))  # use next() or a loop to consume
```

### What is Pydantic and why is it used?
- Pydantic is a Python library for **data validation and settings management** using Python type hints.
- It ensures that input data matches specified types and structures, raising clear validation errors when it doesn't.
- Commonly used with FastAPI and other data pipelines to validate and parse structured data.

#### Example:
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=30)  # valid
user = User(name="Bob", age="30")  # also valid, auto-converts string to int

```

### How does Pydantic handle nested models?
- Pydantic supports nested models by allowing one `BaseModel` to be used as a field in another.
- It automatically validates and parses the nested structure.

#### Example:
```python
from pydantic import BaseModel

class Address(BaseModel):
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    address: Address

user = User(name="Alice", address={"city": "New York", "zip_code": "10001"})
```

### What is the difference between a NumPy array and a pandas DataFrame?
- A **NumPy array** is a multidimensional, fixed-type array for numerical computation.
- A **1D NumPy array** is conceptually similar to a **mathematical vector**
- A NumPy array can also be **2D (matrices)** or **n-dimensional (tensors)**.
- A **pandas DataFrame** is a 2D labeled data structure with columns that can have different data types.

#### Summary:
| Feature      | NumPy array               | pandas DataFrame          |
|--------------|---------------------------|---------------------------|
| Data type    | Homogeneous               | Heterogeneous             |
| Labels       | None (index-based)        | Row and column labels     |
| Use case     | Numerical computation     | Tabular data manipulation |
| Dependencies | Core scientific computing | Built on top of NumPy     |

### What are vectorized operations in NumPy?
- Vectorized operations apply functions to entire arrays without using explicit loops.
- They are faster and more efficient because they use optimized C code under the hood.
- This is a key advantage of NumPy over native Python lists.

#### Example:
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b  # vectorized addition: [5, 7, 9]
```