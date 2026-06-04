from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="需要计算的数学表达式，例如 1 + 2 * 3")


def calculator_tool(args: CalculatorInput) -> str:
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not set(args.expression) <= allowed_chars:
            return "表达式包含非法字符"

        return str(eval(args.expression))
    except Exception as e:
        return f"计算失败：{e}"