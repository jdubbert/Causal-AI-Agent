import json
from typing import TypedDict, Annotated, Optional
import anthropic

from .models import AgentState, CausalMethod, DataProfile, MethodResult
from .tools.profiler import profile_dataset
from .tools.ab_test import run_ab_test
from .tools.did import run_did
from .tools.synthetic_control import run_synthetic_control
from .tools.reporter import generate_report


# ── Anthropic tool definitions ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "profile_dataset",
        "description": (
            "Load and profile the dataset. Returns column names, data shape, detected "
            "treatment/outcome/time columns, and suggested causal methods. "
            "Always call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "Path to the CSV file"}
            },
            "required": ["data_path"]
        }
    },
    {
        "name": "run_ab_test",
        "description": (
            "Run a two-sample A/B test with Welch's t-test. Includes SRM check, "
            "Cohen's d effect size, and normality check for small samples. "
            "Use when data has a treatment/control assignment and no time dimension needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "profile_json": {"type": "string", "description": "JSON string of the DataProfile"}
            },
            "required": ["data_path", "profile_json"]
        }
    },
    {
        "name": "run_did",
        "description": (
            "Run a Difference-in-Differences analysis using two-way fixed effects OLS. "
            "Includes parallel trends test, placebo test, and pre-period balance check. "
            "Use when data has treatment/control groups AND time periods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "profile_json": {"type": "string", "description": "JSON string of the DataProfile"}
            },
            "required": ["data_path", "profile_json"]
        }
    },
    {
        "name": "run_synthetic_control",
        "description": (
            "Run the Abadie-Diamond-Hainmueller Synthetic Control method. "
            "Finds optimal donor unit weights to construct a counterfactual. "
            "Diagnostics include RMSPE pre-period fit, post/pre RMSPE ratio, "
            "and leave-one-out donor sensitivity. "
            "Use when DiD parallel trends fails, or when there are geographic units "
            "with rich pre-period data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "profile_json": {"type": "string", "description": "JSON string of the DataProfile"}
            },
            "required": ["data_path", "profile_json"]
        }
    },
    {
        "name": "generate_report",
        "description": (
            "Generate the final markdown report summarizing the analysis, "
            "findings, diagnostics, and recommendations. Call this as the last step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "final": {"type": "boolean", "description": "Set to true to finalize"}
            },
            "required": ["final"]
        }
    }
]


SYSTEM_PROMPT = """You are an expert causal inference analyst. Your job is to:
1. Profile a dataset to understand its structure
2. Select the most appropriate causal inference method
3. Run the analysis and inspect the diagnostics
4. If diagnostics fail, adapt — try a fallback method or flag the issue
5. Generate a structured report

## Method Selection Logic
- **A/B Test**: Cross-sectional data with treatment/control assignment. No time dimension needed.
- **DiD**: Panel data with treatment/control groups and time periods. Requires parallel trends.
- **Synthetic Control**: Geographic or unit-level panel data. Fallback when DiD parallel trends fail. Best with many pre-periods and donor units.

## Diagnostic Response Logic
- If A/B test SRM fails → flag it prominently but still report the estimate
- If DiD parallel trends fails → try Synthetic Control as a fallback
- If DiD placebo fails → flag credibility concern and report both estimates  
- If Synthetic Control pre-period fit is poor (>10% RMSPE) → flag it and note unreliability

## Your Response Style
Be concise in your reasoning. When you select a method, briefly explain why.
When diagnostics fail, explain what it means for result credibility.
Always call generate_report as your final step.
"""


def _dispatch_tool(tool_name: str, tool_input: dict, state: AgentState) -> tuple[str, AgentState]:
    """Execute a tool call and update state accordingly."""
    try:
        if tool_name == "profile_dataset":
            profile = profile_dataset(tool_input["data_path"])
            state.data_profile = profile
            return profile.model_dump_json(indent=2), state

        elif tool_name == "run_ab_test":
            profile = DataProfile.model_validate_json(tool_input["profile_json"])
            result = run_ab_test(tool_input["data_path"], profile)
            if state.method_result is None:
                state.method_result = result
                state.selected_method = result.method
            else:
                state.fallback_result = result
                state.fallback_method = result.method
            return result.model_dump_json(indent=2), state

        elif tool_name == "run_did":
            profile = DataProfile.model_validate_json(tool_input["profile_json"])
            result = run_did(tool_input["data_path"], profile)
            if state.method_result is None:
                state.method_result = result
                state.selected_method = result.method
            else:
                state.fallback_result = result
                state.fallback_method = result.method
            return result.model_dump_json(indent=2), state

        elif tool_name == "run_synthetic_control":
            profile = DataProfile.model_validate_json(tool_input["profile_json"])
            result = run_synthetic_control(tool_input["data_path"], profile)
            if state.method_result is None:
                state.method_result = result
                state.selected_method = result.method
            else:
                state.fallback_result = result
                state.fallback_method = result.method
            return result.model_dump_json(indent=2), state

        elif tool_name == "generate_report":
            report = generate_report(state)
            state.report = report
            return "Report generated successfully.", state

        else:
            return f"Unknown tool: {tool_name}", state

    except Exception as e:
        error_msg = f"Tool '{tool_name}' failed: {str(e)}"
        state.errors.append(error_msg)
        return f"ERROR: {error_msg}", state


def run_agent(question: str, data_path: str, verbose: bool = True) -> AgentState:
    """
    Main agent loop. Runs until the LLM calls generate_report or hits max iterations.
    """
    client = anthropic.Anthropic()
    state = AgentState(question=question, data_path=data_path)

    messages = [
        {
            "role": "user",
            "content": (
                f"Business question: {question}\n"
                f"Data file: {data_path}\n\n"
                "Please analyze this data and answer the business question using the most "
                "appropriate causal inference method. Profile the data first, then select "
                "and run the right method, check diagnostics, and generate a report."
            )
        }
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤖 Causal Inference Agent Starting")
        print(f"📋 Question: {question}")
        print(f"📁 Data: {data_path}")
        print(f"{'='*60}\n")

    for iteration in range(state.max_iterations * 3):  # generous bound
        state.iteration = iteration

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text reasoning and tool calls
        tool_calls = []
        reasoning_text = []

        for block in response.content:
            if block.type == "text" and block.text.strip():
                reasoning_text.append(block.text.strip())
                if verbose:
                    print(f"💭 {block.text.strip()}\n")
            elif block.type == "tool_use":
                tool_calls.append(block)

        # Log reasoning
        for text in reasoning_text:
            state.reasoning_log.append(text)

        # If no tool calls and stop_reason is end_turn, we're done
        if not tool_calls:
            if verbose:
                print("✅ Agent finished (no more tool calls)")
            break

        # Add assistant message to history
        messages.append({"role": "assistant", "content": response.content})

        # Execute tool calls and build tool result message
        tool_results = []
        for tool_call in tool_calls:
            if verbose:
                print(f"🔧 Calling tool: {tool_call.name}")
                if tool_call.name != "generate_report":
                    print(f"   Input: {json.dumps(tool_call.input, indent=2)[:200]}...")

            result_text, state = _dispatch_tool(tool_call.name, tool_call.input, state)

            if verbose and tool_call.name != "generate_report":
                preview = result_text[:300] + "..." if len(result_text) > 300 else result_text
                print(f"   Result preview: {preview}\n")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result_text,
            })

            # Stop after report generation
            if tool_call.name == "generate_report":
                if verbose:
                    print("📊 Report generated. Agent complete.\n")
                return state

        messages.append({"role": "user", "content": tool_results})

    if verbose and state.report is None:
        print("⚠️ Agent reached iteration limit without generating report")
        # Generate report anyway
        state.report = generate_report(state)

    return state


# ── Interactive Chat Mode ─────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are an expert causal inference analyst having a conversation about a dataset.
You have access to tools that can profile data, run causal analyses, and generate reports.

You have already analyzed this dataset. The analysis results are provided below.
Use them to answer the user's questions directly and conversationally.

When the user asks a question:
- If you can answer from the existing analysis results, just answer in plain text. Be concise and direct.
- If the user asks to run a different method, re-analyze, or look at something the existing results don't cover, use the appropriate tool.
- If the user asks for a report, use generate_report.
- Keep answers concise — a few sentences is usually enough. Don't generate full reports unless asked.

## Analysis Context
{context}
"""


def _build_chat_context(state: AgentState) -> str:
    """Build a context string from the current agent state for the chat system prompt."""
    parts = []

    parts.append(f"Data file: {state.data_path}")
    parts.append(f"Original question: {state.question}")

    if state.data_profile:
        p = state.data_profile
        parts.append(f"\nData Profile: {p.n_rows} rows × {p.n_cols} columns")
        parts.append(f"Columns: {', '.join(p.columns)}")
        parts.append(f"Treatment col: {p.treatment_col}, Outcome col: {p.outcome_col}")
        if p.time_col:
            parts.append(f"Time col: {p.time_col}")
        if p.notes:
            parts.append(f"Notes: {'; '.join(p.notes)}")

    if state.method_result:
        r = state.method_result
        parts.append(f"\nPrimary Analysis: {r.method}")
        parts.append(f"Estimate: {r.estimate:+.4f}")
        if r.relative_effect is not None:
            parts.append(f"Relative lift: {r.relative_effect*100:+.2f}%")
        if r.std_error is not None:
            parts.append(f"Std error: {r.std_error:.4f}")
        if r.ci_lower is not None and r.ci_upper is not None:
            parts.append(f"95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
        if r.p_value is not None:
            parts.append(f"P-value: {r.p_value:.6f}")
        parts.append(f"Diagnostics passed: {r.diagnostics_passed}")
        for d in r.diagnostics:
            status = "✅" if d.passed else "⚠️"
            parts.append(f"  {status} {d.name}: {d.interpretation}")
        if r.raw_output:
            parts.append(f"Raw details: {json.dumps(r.raw_output, indent=2)}")

    if state.fallback_result:
        fr = state.fallback_result
        parts.append(f"\nFallback Analysis: {fr.method}")
        parts.append(f"Estimate: {fr.estimate:+.4f}, P-value: {fr.p_value}")

    return "\n".join(parts)


def run_chat(data_path: str, initial_question: str = None, verbose: bool = True):
    """
    Interactive chat mode. Runs the initial analysis, then enters a loop
    where the user can ask follow-up questions.
    """
    client = anthropic.Anthropic()

    # ── Step 1: Run the initial analysis ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🤖 Causal Inference Agent — Interactive Mode")
    print(f"📁 Data: {data_path}")
    print(f"{'='*60}")

    question = initial_question or "Analyze this dataset and summarize what you find."

    print(f"\n⏳ Running initial analysis...")
    state = run_agent(question=question, data_path=data_path, verbose=verbose)

    if state.report:
        print(f"\n{'─'*60}")
        print(state.report)
        print(f"{'─'*60}")
    else:
        print("⚠️ Initial analysis did not produce a report.")

    # ── Step 2: Enter chat loop ───────────────────────────────────────────────
    context = _build_chat_context(state)
    system = CHAT_SYSTEM_PROMPT.format(context=context)

    messages = []

    print(f"\n💬 Chat mode active. Ask follow-up questions about your data.")
    print(f"   Type 'quit' or 'exit' to end. Type 'report' to save the report.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if user_input.lower() == "report":
            if state.report:
                output_path = data_path.replace(".csv", "_report.md")
                with open(output_path, "w") as f:
                    f.write(state.report)
                print(f"📄 Report saved to: {output_path}")
            else:
                print("⚠️ No report to save yet.")
            continue

        messages.append({"role": "user", "content": user_input})

        # Allow multiple tool-use turns for a single question
        for _ in range(6):
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2048,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            tool_calls = []
            text_parts = []

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    text_parts.append(block.text.strip())
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # If there are tool calls, execute them and continue the loop
            if tool_calls:
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tool_call in tool_calls:
                    if verbose:
                        print(f"  🔧 {tool_call.name}...")
                    result_text, state = _dispatch_tool(tool_call.name, tool_call.input, state)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result_text,
                    })
                    # Refresh context after tool use
                    context = _build_chat_context(state)
                    system = CHAT_SYSTEM_PROMPT.format(context=context)
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls — print the text response and break
                if text_parts:
                    answer = "\n".join(text_parts)
                    print(f"\n🤖 {answer}\n")
                    messages.append({"role": "assistant", "content": response.content})
                break
