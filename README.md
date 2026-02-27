# 🧪 Causal Inference Agent

The goal of this project is to help document and guide me down the AI agent rabbit hole, using methods/ideas that I already know. This will continue to be a huge WIP with many iterations as I learn more and more of the cool things you can do.

TLDR: this is pretty basic right now, but we have to start somewhere!

Ultimate Goal: An AI-powered agent that **automatically selects, runs, and interprets causal inference methods** on your data. Give it a CSV and a business question — it profiles your data, picks the right method, validates assumptions with diagnostic checks, and generates a full markdown report.

Built with [Claude](https://docs.anthropic.com/en/docs/intro-to-claude) as the reasoning backbone and pure Python statistical implementations.

---

## How It Works

The agent follows a structured **Goal → Plan → Action → Outcome** workflow:

```
CSV + Business Question
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  🎯 GOAL                                            │
  │  Profile the data — detect columns, structure,      │
  │  treatment/outcome, and restate the causal question  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  📋 PLAN                                            │
  │  Create analysis plan — pick methods, justify the   │
  │  choice, define fallback strategy & concrete steps   │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  ⚡ ACTION                                           │
  │  Execute the plan — run primary method, check       │
  │  diagnostics, fall back if assumptions fail          │
  └──────────────────────┬──────────────────────────────┘
                         │
                    ┌────┴────┐  Diagnostics fail?
                    │         │  → Follows fallback strategy
                    ▼         ▼
  ┌─────────────────────────────────────────────────────┐
  │  📊 OUTCOME                                         │
  │  Synthesize findings, generate structured report    │
  │  with estimates, CIs, diagnostics & recommendations  │
  └─────────────────────────────────────────────────────┘
```

## Supported Methods

| Method | When It's Used | Key Diagnostics |
|--------|---------------|-----------------|
| **A/B Test** (Welch's t-test) | Cross-sectional treatment/control data | Sample Ratio Mismatch, Cohen's d, Normality check |
| **Difference-in-Differences** (TWFE) | Panel data with treatment groups + time periods | Event-study parallel trends (F-test), Placebo test, Pre-period balance, Anticipation effects |
| **Synthetic Control** (ADH) | Geo/unit-level panel data; fallback when DiD fails | Pre-period fit (RMSPE), Fisher permutation p-value, Time placebo, Post/Pre RMSPE ratio, Leave-one-out sensitivity |

The agent automatically falls back — for example, if DiD parallel trends fail, it tries Synthetic Control and reports both.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/jdubbert/Causal-AI-Agent.git
cd Causal-AI-Agent
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Run the demo

```bash
python run_agent.py --demo
```

This runs all three methods on included synthetic datasets and saves reports to `causal_agent/data/`.

---

## Usage

### Analyze your own data

```bash
python run_agent.py \
  --data path/to/your_data.csv \
  --question "Did the new feature cause an increase in conversions?"
```

### All options

```
usage: run_agent.py [-h] [--data DATA] [--question QUESTION]
                    [--output OUTPUT] [--demo] [--chat] [--quiet]

options:
  --data      Path to CSV data file
  --question  Business question to answer
  --output    Output path for report (.md)
  --demo      Run all demo scenarios
  --chat      Interactive chat mode — ask follow-up questions
  --quiet     Suppress verbose output
```

### Interactive chat mode

Instead of a one-shot report, start a conversation with the agent about your data:

```bash
python run_agent.py --data causal_agent/data/ab_test.csv --chat
```

The agent runs the initial analysis, then drops you into a chat loop where you can ask follow-ups:

```
You: Was the sample split evenly?
🤖 Yes — 2,003 control vs 1,997 treatment. The SRM check passed (p=0.91),
   so the randomization looks clean.

You: What if I only look at customers with tenure > 12 months?
🤖 I'd need to re-run the analysis on a filtered subset. The current data
   shows tenure ranges from 1-240 months...

You: Is the effect size large enough to matter?
🤖 Cohen's d = 0.17, which is a small effect. The lift is statistically
   significant but practically modest at ~3.5 percentage points.

You: Can you explain the results to a non-technical stakeholder?
**What we tested:** We split ~4,000 customers into two equal groups — one got the treatment (the new experience/feature/offer) and the other didn't.

**What we found:** Customers who received the treatment were **more likely to activate**. About **15.5 out of 100** treated customers activated, compared to only **11.7 out of 100** in the group that didn't get it. That's roughly **1 in 3 more activations** than we'd normally expect.

**Can we trust this?** Yes. The groups were evenly split, and the result is very unlikely to be due to random chance. We're confident the true improvement is somewhere between **1.7 and 6 extra activations per 100 customers**.

**Bottom line:** The treatment works. It's not a massive effect per individual customer, but at scale across thousands of customers, those extra activations add up. I'd recommend rolling it out more broadly.

You: report
📄 Report saved to: causal_agent/data/ab_test_report.md

You: quit
👋 Goodbye!
```

You can also provide an initial question with chat mode:

```bash
python run_agent.py --data causal_agent/data/did.csv --chat --question "Did the campaign work?"
```

---

## Demo Datasets

Three synthetic datasets are included in `causal_agent/data/` (generated by `generate_demo_data.py`):

### A/B Test — Email Campaign → Feature Activation
- **File**: `causal_agent/data/ab_test.csv`
- **Setup**: 4,000 customers, 50/50 random split
- **True effect**: ~3.5 pp lift in activation rate (12% → 15.5%)
- **Columns**: `customer_id`, `treatment`, `activated`, `age`, `tenure_months`

```bash
python run_agent.py \
  --data causal_agent/data/ab_test.csv \
  --question "Did the email campaign increase checking account feature activations?"
```

### Difference-in-Differences — Marketing Campaign → Revenue
- **File**: `causal_agent/data/did.csv`
- **Setup**: 20 markets (6 treatment, 14 control) × 20 weeks
- **True effect**: ~8% revenue lift post-campaign
- **Columns**: `market`, `week`, `treatment`, `post`, `revenue`

```bash
python run_agent.py \
  --data causal_agent/data/did.csv \
  --question "Did the marketing campaign cause an incremental lift in revenue?"
```

### Synthetic Control — Geo Campaign → DMA Conversions
- **File**: `causal_agent/data/synthetic_control.csv`
- **Setup**: 15 DMAs × 52 weeks, 1 treated DMA
- **True effect**: ~12% lift starting week 27
- **Columns**: `dma`, `week`, `conversions`, `treated`, `post`

```bash
python run_agent.py \
  --data causal_agent/data/synthetic_control.csv \
  --question "What was the causal impact of the campaign on the treated DMA?"
```

---

## Sample Output

Here's a snippet from the A/B test report:

```
### Primary Analysis: A/B Test (Two-Sample T-Test)

| Metric       | Value                            |
|------------- |----------------------------------|
| ATT Estimate | +0.0350                          |
| Relative Lift| +29.17%                          |
| 95% CI       | [0.0210, 0.0490]                 |
| P-value      | 0.0000                           |
| Significance | highly significant (p < 0.001)   |

Diagnostics:
  ✅ Sample Ratio Mismatch — No SRM detected
  ✅ Cohen's d = 0.17 (small effect size)

Finding: Statistically significant causal effect with all diagnostics passed.
```

Full sample reports are in the repo: `sample_report_ab_test.md`, `sample_report_did.md`, `sample_report_sc.md`.

---

## Project Structure

```
├── run_agent.py                  # CLI entrypoint
├── generate_demo_data.py         # Creates the 3 synthetic datasets
├── requirements.txt
├── causal_agent/
│   ├── __init__.py
│   ├── models.py                 # Pydantic models (AgentState, AnalysisPlan, AgentPhase, ...)
│   ├── runner.py                 # Goal→Plan→Action→Outcome agent loop
│   └── tools/
│       ├── __init__.py
│       ├── profiler.py           # Auto-detects columns & suggests methods
│       ├── ab_test.py            # Welch's t-test + SRM + Cohen's d
│       ├── did.py                # TWFE DiD + event-study + placebo + anticipation
│       ├── synthetic_control.py  # ADH SC + Fisher permutation + time placebo + LOO
│       └── reporter.py           # Markdown report generator
└── sample_report_*.md            # Example output reports
```

---

## Data Format Requirements

Your CSV should include some combination of these columns (the profiler auto-detects them):

| Column Type | Example Names | Required For |
|-------------|--------------|-------------|
| **Treatment** | `treatment`, `treated`, `group`, `variant` | All methods |
| **Outcome** | `revenue`, `conversions`, `activated`, `sales` | All methods |
| **Time** | `week`, `date`, `month`, `period` | DiD, Synthetic Control |
| **Geo/Unit** | `dma`, `market`, `state`, `region` | Synthetic Control |
| **Post indicator** | `post`, `is_post`, `after` | DiD (optional — auto-splits if missing) |

---

## How the Agent Reasons

The agent uses Claude's tool-use capability to follow a structured **Goal → Plan → Action → Outcome** loop:

1. **🎯 Goal** — Profile the data, detect its structure, and restate the business question as a causal inference problem
2. **📋 Plan** — Create a formal analysis plan: which methods to try, why they fit, what to do if diagnostics fail, and the concrete steps to execute
3. **⚡ Action** — Execute the plan step-by-step: run the primary method, inspect diagnostics, and follow the fallback strategy if assumptions are violated
4. **📊 Outcome** — Synthesize findings across all methods tried and generate a structured markdown report

The plan and all reasoning steps are logged and included in the final report, so you can see exactly *why* the agent made each decision.

```
🎯  Phase: GOAL — Understanding the question & data
🔧 Calling tool: profile_dataset

📋  Phase: PLAN — Creating analysis plan
   🎯 Goal: Estimate the causal effect of the marketing campaign on revenue
   🔬 Methods: DiD → Synthetic Control
   🔄 Fallback: If parallel trends fail, switch to SC
   📝 Steps:
      1. Run DiD with TWFE and cluster-robust SEs
      2. Check parallel trends via event-study F-test
      3. If parallel trends fail, run Synthetic Control
      4. Generate final report with all findings

⚡  Phase: ACTION — Executing analysis
🔧 Calling tool: run_did

📊  Phase: OUTCOME — Synthesizing results
✅ Agent complete — Goal → Plan → Action → Outcome workflow finished.
```

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- Dependencies: see `requirements.txt`

