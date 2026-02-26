"""
Causal Inference Agent — Main Entrypoint

Usage:
    python run_agent.py --data data/ab_test.csv --question "Did our email campaign drive incremental activations?"
    python run_agent.py --data data/ab_test.csv --chat  # interactive Q&A mode
    python run_agent.py --demo  # runs all three demos
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_agent.runner import run_agent, run_chat


def run_demo():
    """Run all three methods on their respective test datasets."""
    demos = [
        {
            "data": "causal_agent/data/ab_test.csv",
            "question": "Did our email campaign drive a statistically significant increase in checking account feature activation rates?",
            "output": "causal_agent/data/report_ab_test.md",
        },
        {
            "data": "causal_agent/data/did.csv",
            "question": "Did the marketing campaign cause an incremental lift in revenue across treatment markets compared to control markets?",
            "output": "causal_agent/data/report_did.md",
        },
        {
            "data": "causal_agent/data/synthetic_control.csv",
            "question": "What was the causal impact of the geo-targeted campaign on weekly conversions in the treated DMA?",
            "output": "causal_agent/data/report_synthetic_control.md",
        },
    ]

    for demo in demos:
        print(f"\n{'#'*70}")
        print(f"# DEMO: {demo['data']}")
        print(f"{'#'*70}")

        state = run_agent(
            question=demo["question"],
            data_path=demo["data"],
            verbose=True
        )

        if state.report:
            with open(demo["output"], "w") as f:
                f.write(state.report)
            print(f"\n📄 Report saved to: {demo['output']}")
            print("\n" + "─"*60)
            print(state.report[:1000] + "...\n[truncated — see file for full report]")
        else:
            print("⚠️ No report generated")


def main():
    parser = argparse.ArgumentParser(description="Causal Inference Agent")
    parser.add_argument("--data", type=str, help="Path to CSV data file")
    parser.add_argument("--question", type=str, help="Business question to answer")
    parser.add_argument("--output", type=str, default=None, help="Output path for report (.md)")
    parser.add_argument("--demo", action="store_true", help="Run all demo scenarios")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode — ask follow-up questions")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.chat:
        if not args.data:
            parser.print_help()
            print("\n💡 --chat requires --data. Example:")
            print("   python run_agent.py --data causal_agent/data/ab_test.csv --chat")
            return
        run_chat(
            data_path=args.data,
            initial_question=args.question,
            verbose=not args.quiet,
        )
        return

    if not args.data or not args.question:
        parser.print_help()
        print("\n💡 Run with --demo to test all three methods on synthetic data")
        print("💡 Run with --data <file> --chat for interactive Q&A mode")
        return

    state = run_agent(
        question=args.question,
        data_path=args.data,
        verbose=not args.quiet
    )

    if state.report:
        output_path = args.output or args.data.replace(".csv", "_report.md")
        with open(output_path, "w") as f:
            f.write(state.report)
        print(f"\n📄 Report saved to: {output_path}")
    else:
        print("⚠️ Agent did not produce a report")


if __name__ == "__main__":
    main()
