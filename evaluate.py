
from langsmith import Client
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from main import graph

client = Client()
DATASET = "support-triage-v1"

# Six synthetic support tickets with expected outcomes
examples = [
    {
        "inputs": {
            "ticket": "Our API returns 401 after rotating the API key."
        },
        "outputs": {
            "category": "Authentication",
            "action": "troubleshoot"
        }
    },
    {
        "inputs": {
            "ticket": "Kubernetes pods keep restarting with CrashLoopBackOff after deployment."
        },
        "outputs": {
            "category": "Kubernetes",
            "action": "troubleshoot"
        }
    },
    {
        "inputs": {
            "ticket": "Terraform apply fails with AccessDenied when creating an S3 bucket."
        },
        "outputs": {
            "category": "Terraform",
            "action": "troubleshoot"
        }
    },
    {
        "inputs": {
            "ticket": "All production customers are unable to access our application."
        },
        "outputs": {
            "category": "Other",
            "action": "escalate"
        }
    },
    {
        "inputs": {
            "ticket": "My application isn't working."
        },
        "outputs": {
            "category": "Other",
            "action": "clarify"
        }
    },
    {
        "inputs": {
            "ticket": "What is the capital of France?"
        },
        "outputs": {
            "category": "Other",
            "action": "clarify"
        }
    }
]


# Create dataset once, or reuse it
if not client.has_dataset(dataset_name=DATASET):
    dataset = client.create_dataset(dataset_name=DATASET)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
    print("Created dataset:", DATASET)
else:
    print("Using existing dataset:", DATASET)


# Execute our existing LangGraph agent
def run_agent(inputs: dict) -> dict:
    return graph.invoke({"ticket": inputs["ticket"]})


# Evaluator 1: Did the agent classify correctly?
def category_correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["category"] == reference_outputs["category"]


# Evaluator 2: Did the agent choose the expected action?
def action_correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["action"] == reference_outputs["action"]


# Evaluator 3: LLM-as-a-judge
class JudgeResult(BaseModel):
    helpful: bool = Field(
        description="Whether the response is useful and appropriate"
    )


judge = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).with_structured_output(JudgeResult)


def response_helpful(inputs: dict, outputs: dict) -> dict:
    judgment = judge.invoke(
        f"""
        You are evaluating a technical support triage agent.

        Customer input:
        {inputs['ticket']}

        Agent action:
        {outputs['action']}

        Agent response:
        {outputs['response']}

        Is the response helpful and appropriate for a
        technical support assistant?

        Consider:
        - Does it stay within technical support scope?
        - Does it provide appropriate next steps?
        - Does it avoid asserting unverified root causes?
        - Does it avoid unsafe or unnecessary actions?

        For unrelated questions, a helpful response should
        politely explain the agent's technical support scope.

        Return your judgment.
        """
    )

    return {
        "key": "response_helpful",
        "score": judgment.helpful
    }


# Run the SDK experiment
results = client.evaluate(
    run_agent,
    data=DATASET,
    evaluators=[
        category_correct,
        action_correct,
        response_helpful
    ],
    experiment_prefix="support-triage-final",
    max_concurrency=1
)

print("Evaluation finished. Check LangSmith experiments.")
