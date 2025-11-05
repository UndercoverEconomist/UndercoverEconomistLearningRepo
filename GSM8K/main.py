# pip install langgraph ollama
from typing import TypedDict
from langgraph.graph import StateGraph, END
from ollama import Client
import json, math, argparse

# ---- CLI Argument ----
parser = argparse.ArgumentParser(description="Math Agent with optional reflection.")
parser.add_argument("--no_reflect", action="store_true", help="Disable the reflection & replanning stage")
args = parser.parse_args()

# ---- Connect to local Ollama ----
ollama = Client(host="http://localhost:11434")

# ---- Agent State ----
class State(TypedDict, total=False):
    question: str
    code: str
    result: str
    reflection: str
    is_correct: bool
    output: str


# ===== Node 1: Planner =====
def planner(state: State) -> State:
    print("\n========\nPLANNER\n========")
    question = state.get("question", "")
    prompt = f"""
    You are a math reasoning agent. Convert the following word problem into a valid Python
    expression that can be evaluated using `eval()`. Do NOT define functions or use print.
    The expression should compute the correct numeric answer.

    Example:
    Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
    Expected Expression: 48 + (48 / 2)

    Problem: {question}

    Respond with ONLY the Python expression (no explanations, no markdown).
    """
    resp = ollama.chat(
        model="llama3",  # change if needed
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    expression = resp["message"]["content"].strip()
    print("Generated Expression:\n", expression)
    return {**state, "code": expression}


# ===== Node 2: Executor =====
def executor(state: State) -> State:
    print("\n========\nEXECUTOR\n========")
    code = state.get("code", "")
    result = "Error"
    try:
        # Evaluate safely within math context only
        safe_globals = {"__builtins__": {}, "math": math}
        result = eval(code, safe_globals)
    except Exception as e:
        result = f"Evaluation error: {e}"
    print(f"Evaluated Result: {result}")
    return {**state, "result": str(result)}


# ===== Node 3: Reflector =====
def reflector(state: State) -> State:
    print("\n========\nREFLECTOR\n========")
    question = state.get("question", "")
    code = state.get("code", "")
    result = state.get("result", "")

    prompt = f"""
    You are a strict math checker. 
    Verify if the computed result is correct for the problem below.

    Problem: {question}
    Expression: {code}
    Result: {result}

    Respond in JSON:
    {{
      "is_correct": true/false,
      "explanation": "Why it is correct or not."
    }}
    """
    resp = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    reflection_text = resp["message"]["content"]
    print("Raw Reflection Output:\n", reflection_text)

    try:
        reflection = json.loads(reflection_text)
        is_correct = reflection.get("is_correct", False)
        explanation = reflection.get("explanation", "")
    except json.JSONDecodeError:
        reflection = {"is_correct": False, "explanation": "Failed to parse reflection JSON."}
        is_correct = False
        explanation = reflection["explanation"]

    print("Reflection:", explanation)
    print("Correct:", is_correct)
    return {**state, "reflection": explanation, "is_correct": is_correct}


# ===== Node 4: Replanner =====
def replanner(state: State) -> State:
    print("\n========\nREPLANNER\n========")
    question = state.get("question", "")
    reflection = state.get("reflection", "")
    prev_code = state.get("code", "")

    prompt = f"""
    The previous expression was incorrect.
    Problem: {question}

    Previous expression:
    {prev_code}

    Reflection feedback:
    {reflection}

    Please produce a corrected Python expression that computes the right answer.
    Respond with ONLY the corrected Python expression (no explanation).
    """
    resp = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    new_expr = resp["message"]["content"].strip()
    print("Revised Expression:\n", new_expr)
    return {**state, "code": new_expr}


# ===== Node 5: Explainer =====
def explainer(state: State) -> State:
    print("\n========\nEXPLAINER\n========")
    question = state.get("question", "")
    code = state.get("code", "")
    result = state.get("result", "")
    reflection = state.get("reflection", "")

    prompt = f"""
    You are a math tutor. Explain how to solve this problem in simple steps,
    showing how the expression {code} leads to the final result {result}.

    Problem: {question}
    Reflection summary: {reflection}
    """
    resp = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.4},
    )
    output = resp["message"]["content"]
    print("Explanation:\n", output)
    return {**state, "output": output}


# ===== Build Graph =====
graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("reflector", reflector)
graph.add_node("replanner", replanner)
graph.add_node("explainer", explainer)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")

if args.no_reflect:
    graph.add_edge("executor", "explainer")
else:
    graph.add_edge("executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        lambda s: "replanner" if not s.get("is_correct") else "explainer",
    )
    graph.add_edge("replanner", "executor")

graph.add_edge("explainer", END)
app = graph.compile()


# ===== Run Example =====
if __name__ == "__main__":
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether?"
    )

    print("🧮 QUESTION:\n", question)
    result = app.invoke({"question": question})

    print("\n🪞 FINAL REFLECTION:", result.get("reflection", "None"))
    print("\n✅ FINAL ANSWER:\n", result["output"])
