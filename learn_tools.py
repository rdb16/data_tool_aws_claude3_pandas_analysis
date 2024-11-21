from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import List, Tuple
import random


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: firs int
        b: second int
    """
    return a * b


@tool(response_format="content_and_artifact")
def generate_random_ints(min: int, max: int, size: int) -> Tuple[str, List[int]]:
    """Generate size random ints in the range [min, max]."""
    array = [random.randint(min, max) for _ in range(size)]
    content = f"successfully generated {size} random integers[{min}, {max}]."
    return content, array


if __name__ == '__main__':
    res = multiply.invoke({"a": 5, "b": 2})
    print(res)
    print(multiply.name)
    print(multiply.description)
    print(multiply.args)
    res2 = generate_random_ints.invoke({"min": 5, "max": 10, "size": 10})
    print("$$$$$$$$$")
    print()
    print(res2)
    print()
    res3 = generate_random_ints.invoke(
        {
            "name": "g",
            "args": {"min": 5, "max": 10, "size": 10},
            "id": "123", #required
            "type": "tool_call", #required
        }
    )
    print(res3)

    dotenv_path = "~/.env/"
    load_dotenv(dotenv_path)
