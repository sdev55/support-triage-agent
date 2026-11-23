# AI Technical Support Triage Agent

A simple LangGraph-powered workflow that analyzes incoming technical support tickets and recommends appropriate next steps.

Built as part of the LangChain Senior Technical Support Engineer take-home exercise.

## 1. Project Overview

The goal of this project is to explore LangGraph's workflow orchestration and LangSmith's tracing and evaluation capabilities through a realistic technical support scenario.

The application:

- Analyzes incoming support tickets using an LLM.
- Identifies the issue category and severity.
- Determines whether sufficient information is available.
- Routes the ticket to troubleshooting, clarification, or escalation.
- Generates an appropriate support response.

The workflow recommends actions but does not execute infrastructure commands or create real support escalations.

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Application development |
| LangGraph | Workflow orchestration and state management |
| OpenAI GPT-4.1-mini | Ticket analysis and response generation |
| LangChain OpenAI integration | Communication with the OpenAI API |
| LangSmith | Tracing, debugging, datasets, and evaluations |

## 3. LangGraph Architecture

The workflow follows this structure:

```text
START
  |
  v
analyze_ticket
  |
  v
route_ticket (conditional routing)
  |
  |---- Critical severity ---> escalate
  |
  |---- Insufficient info ---> clarify
  |
  |---- Sufficient info ----> troubleshoot
                                  |
                                  v
                                 END
```

### State

The shared state contains:

- `ticket`: Original customer issue.
- `category`: Authentication, Kubernetes, Terraform, Performance, LangSmith, or Other.
- `severity`: Low, medium, high, or critical.
- `sufficient_info`: Whether enough information exists to suggest initial troubleshooting.
- `action`: Selected routing action.
- `response`: Generated support response.

### Nodes and Routing

**analyze_ticket:** Uses an LLM with structured output to classify the incoming ticket.

**route_ticket:** Applies deterministic Python logic to choose the next node.

**clarify:** Generates clarifying questions when information is insufficient.

**troubleshoot:** Recommends initial diagnostic steps.

**escalate:** Recommends escalation for critical incidents.

The workflow combines LLM-based classification with deterministic conditional routing. It is an LLM-powered workflow rather than a fully autonomous tool-executing agent.

## 4. Setup

### Prerequisites

- Python 3.12
- OpenAI API key with available credits
- LangSmith account and API key

### Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure credentials through environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=support-triage-agent
```

Use valid credentials locally. Never commit API keys to GitHub.

### Run the Agent

```bash
python main.py
```

Example input:

```text
Our Kubernetes pods are repeatedly restarting
with CrashLoopBackOff after yesterday's deployment.
```

The agent analyzes the issue and generates an appropriate triage recommendation.

## 5. LangSmith Tracing

LangSmith tracing was enabled to inspect workflow execution.

The traces provide visibility into:

- LangGraph node execution.
- LLM calls and their inputs and outputs.
- Routing decisions.
- Execution latency.
- Final responses.

For example, a widespread production outage was correctly classified as critical and routed to escalation.

This helped validate that critical severity takes priority over insufficient information.

## 6. Evaluation Approach

A synthetic dataset named `support-triage-v1` was created with six support scenarios:

1. API authentication failure.
2. Kubernetes CrashLoopBackOff.
3. Terraform AccessDenied error.
4. Widespread production outage.
5. Vague application issue.
6. Unrelated general-knowledge question.

### SDK Evaluation

The LangSmith Python SDK was used to run the complete LangGraph workflow against the dataset.

Three evaluation metrics were implemented:

- **Category correctness:** Whether the predicted category matches the reference.
- **Action correctness:** Whether the selected routing action matches the reference.
- **Response helpfulness:** LLM-as-a-judge scoring of the generated response.

### Initial Results

The first SDK experiment reported:

| Metric | Displayed average |
|---|---:|
| Category correctness | 100% |
| Action correctness | 80% |
| Response helpfulness | 80% |

These were preliminary results: LangSmith displayed feedback for five of six runs at the time of inspection. They should not be interpreted as finalized six-example accuracy.

### UI Evaluation

A separate experiment was run through the LangSmith Playground using the same dataset and GPT-4.1-mini.

This tested a standalone technical support prompt rather than the complete LangGraph workflow.

A UI-configured LLM-as-a-judge evaluator was also created and successfully tested against a production-outage trace.

The SDK and Playground experiments therefore exercise different application configurations and are not directly comparable as a controlled before-and-after test.

## 7. Findings and Limitations

### Out-of-Scope Questions

The initial workflow answered an unrelated geography question instead of redirecting the user to technical support.

**Potential improvement:** Add explicit scope detection and route unrelated questions appropriately.

### Severity Classification

The initial prompt defined widespread production outages as critical but did not provide detailed definitions for every severity level.

**Potential improvement:** Introduce a standardized severity rubric based on customer impact.

### Response Length

The Kubernetes example generated ten troubleshooting recommendations.

**Potential improvement:** Prioritize three to five initial non-destructive diagnostic steps.

### Production-Safety Considerations

The agent suggested potentially disruptive actions such as deployment rollback.

**Potential improvement:** Require impact assessment and appropriate authorization before recommending production changes.

### Evaluation Limitations

- The dataset contains only six synthetic examples.
- Reference labels were manually selected.
- LLM-as-a-judge feedback is subjective.
- The judge and agent use the same model family.
- No real customer data or production infrastructure was used.

## 8. Developer Experience and Friction Log

### Python Dependency Import Failures

**Observation:** During local setup, imports failed for `LanguageModelInput` and `Event`.

**Investigation:** Checked package metadata, inspected import paths, and identified missing package initialization files.

**Resolution:** Targeted reinstallation restored the affected imports.

**Learning:** Successful dependency metadata checks do not guarantee successful runtime imports.

The underlying cause of the missing files was not established, so this is documented as a local setup issue rather than a confirmed LangChain defect.

### API Authentication and Billing

**Observation:** Initial API calls failed with authentication and insufficient-credit errors.

**Resolution:** Corrected credential configuration and enabled API billing.

**Learning:** Authentication, billing, and application errors require different diagnostic approaches.

### LangSmith Evaluator Configuration

**Observation:** Configuring the UI-based evaluator required mapping template variables to trace input and output fields.

**Resolution:** Inspected the sample data, mapped the required variables, and tested the evaluator.

**Learning:** Example-based guidance showing variable mapping for structured LangGraph outputs could simplify onboarding for new users.

### Invalid HTTP Authorization Header

**Observation:** The evaluator failed with a connection error, but the nested exception identified an invalid authorization header containing trailing whitespace.

**Resolution:** Replaced the affected API key and verified that the evaluator executed successfully.

**Learning:** The outer exception can be misleading; examining the underlying cause is essential.

## 9. Future Improvements

Potential improvements include:

- Explicit out-of-scope detection.
- A documented severity-classification rubric.
- Shorter, prioritized troubleshooting responses.
- Expanded evaluation coverage.
- Clearer safeguards around potentially disruptive operations.
- Retrieval of verified support documentation for more grounded recommendations.

## 10. Key Takeaways

This project provided hands-on experience with:

- LangGraph's state, nodes, edges, and conditional routing.
- LLM integration and structured outputs.
- LangSmith execution tracing.
- Creating evaluation datasets.
- SDK-based and UI-based experiments.
- LLM-as-a-judge feedback.
- Identifying application limitations and developer experience friction.

The main takeaway is that building an LLM workflow is only the first step. Tracing and evaluation are essential for understanding whether the application behaves as expected and where it needs improvement.