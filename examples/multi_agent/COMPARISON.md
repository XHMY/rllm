# Single-Agent vs Multi-Agent Comparison

This document provides a detailed comparison between single-agent and multi-agent implementations.

## Architecture Comparison

### Single-Agent Architecture

```
Task → Agent → Multiple Steps → Final Answer
```

**Example (DeepCoder single-agent)**:
```python
class CompetitionCodingAgent(BaseAgent):
    def __init__(self):
        self._trajectory = Trajectory()  # Single trajectory, default name="agent"

# In environment:
agent.update_from_env(observation, reward, done, info)
agent.update_from_model(model_response)
# Returns: Episode with single trajectory
```

### Multi-Agent Architecture

```
Task → [Agent1, Agent2, Agent3] → Multiple Trajectories → Final Answer
```

**Example (DeepCoder multi-agent)**:
```python
class CodeGenerator:
    async def generate_code(self, problem):
        output = await self.rollout_engine.get_model_response(
            messages,
            agent_name="generator"  # ← Explicit agent name
        )
        return Trajectory(name="generator", steps=[...])

class TestRunner:
    async def analyze_tests(self, code, tests):
        output = await self.rollout_engine.get_model_response(
            messages,
            agent_name="test_runner"  # ← Different agent name
        )
        return Trajectory(name="test_runner", steps=[...])

class MultiAgentWorkflow(Workflow):
    async def run(self, task, uid):
        gen_traj = await self.generator.generate_code(...)
        test_traj = await self.test_runner.analyze_tests(...)
        refine_traj = await self.refiner.refine_code(...)

        return Episode(
            trajectories=[gen_traj, test_traj, refine_traj]  # ← Multiple trajectories
        )
```

---

## Code Structure Comparison

### 1. DeepCoder

#### Single-Agent (`examples/deepcoder/run_deepcoder.py`)
```python
# Single agent class
agent_class = CompetitionCodingAgent

# Single environment
env_class = SingleTurnEnvironment

# Returns single trajectory per episode
engine = AgentExecutionEngine(
    agent_class=CompetitionCodingAgent,
    env_class=SingleTurnEnvironment,
    ...
)
```

**Trajectory structure**:
```python
Episode(
    trajectories=[
        Trajectory(name="agent", steps=[step1, step2, ...])
    ]
)
```

#### Multi-Agent (`examples/multi_agent/deepcoder/`)
```python
# Multiple specialized agents
class CodeGenerator: ...
class TestRunner: ...
class CodeRefiner: ...

# Workflow orchestrates agents
class MultiAgentDeepCoderWorkflow(Workflow):
    def __init__(self):
        self.generator = CodeGenerator(...)
        self.test_runner = TestRunner(...)
        self.refiner = CodeRefiner(...)

engine = WorkflowExecutionEngine(
    workflow_class=MultiAgentDeepCoderWorkflow,
    ...
)
```

**Trajectory structure**:
```python
Episode(
    trajectories=[
        Trajectory(name="generator", steps=[...]),
        Trajectory(name="test_runner", steps=[...]),
        Trajectory(name="refiner", steps=[...]),
    ]
)
```

---

### 2. SWE

#### Single-Agent
```python
# One agent handles everything
class SWEAgent(BaseAgent):
    def __init__(self, use_fn_calling=False):
        self._trajectory = Trajectory()  # Default name="agent"

    def update_from_env(self, observation, reward, done, info):
        # Handles all: analysis, coding, testing
        ...
```

#### Multi-Agent
```python
# Specialized agents
class IssueAnalyzer:
    # Only analyzes issues and creates plans
    async def analyze_issue(self, issue_description):
        return Trajectory(name="analyzer", ...)

class CodeWriter:
    # Only writes code
    async def write_code(self, issue, plan):
        return Trajectory(name="writer", ...)

class TestValidator:
    # Only validates with tests
    async def validate_changes(self, issue, code, tests):
        return Trajectory(name="validator", ...)
```

---

### 3. Math Tool

#### Single-Agent
```python
class ToolAgent(BaseAgent):
    def __init__(self, tools):
        self.tools = MultiTool(tools=tools)
        self._trajectory = Trajectory()

    # Handles: problem analysis, code writing, execution, verification
    def update_from_model(self, response):
        tool_calls = self.tool_parser.parse(response)
        # All in one agent
```

#### Multi-Agent
```python
class ProblemAnalyzer:
    # Analyzes and plans
    async def analyze_problem(self, problem):
        return Trajectory(name="analyzer", ...)

class CodeExecutor:
    # Executes Python code
    async def execute_solution(self, problem, plan):
        code = self._extract_code(output)
        result = await self._execute_code(code)
        return Trajectory(name="executor", ...)

class AnswerVerifier:
    # Verifies correctness
    async def verify_solution(self, problem, code, result):
        return Trajectory(name="verifier", ...)
```

---

### 4. Search

#### Single-Agent
```python
class ToolAgent(BaseAgent):
    def __init__(self, tools):
        self.tools = MultiTool(tools=tools)
        self._trajectory = Trajectory()

    # Handles: query generation, search, answer extraction
    def update_from_model(self, response):
        tool_calls = self.tool_parser.parse(response)
        # All in one agent
```

#### Multi-Agent
```python
class QueryOptimizer:
    # Optimizes search queries
    async def optimize_queries(self, question):
        return Trajectory(name="query_optimizer", ...)

class DocumentRetriever:
    # Retrieves documents
    async def retrieve_documents(self, question, query):
        search_results = await self._execute_search(query)
        return Trajectory(name="document_retriever", ...)

class AnswerExtractor:
    # Extracts final answer
    async def extract_answer(self, question, all_results):
        return Trajectory(name="answer_extractor", ...)
```

---

## Execution Flow Comparison

### Single-Agent Flow

```
┌─────────────────────────────────────────┐
│ AgentExecutionEngine                    │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Environment.reset()                     │
│ Returns: observation                    │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Loop (while not done):                  │
│   1. agent.update_from_env(obs)         │
│   2. agent.chat_completions → Model     │
│   3. agent.update_from_model(response)  │
│   4. env.step(action)                   │
│   5. Get new observation                │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Returns: Episode                        │
│   - Single trajectory (name="agent")    │
└─────────────────────────────────────────┘
```

### Multi-Agent Flow

```
┌─────────────────────────────────────────┐
│ WorkflowExecutionEngine                 │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Workflow.run(task, uid)                 │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Agent1.do_task()                        │
│   - get_model_response(agent_name="a1") │
│   - Returns: Trajectory(name="a1")      │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Agent2.do_task()                        │
│   - get_model_response(agent_name="a2") │
│   - Returns: Trajectory(name="a2")      │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Agent3.do_task()                        │
│   - get_model_response(agent_name="a3") │
│   - Returns: Trajectory(name="a3")      │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Returns: Episode                        │
│   - trajectories=[traj1, traj2, traj3]  │
│   - Each with different name            │
└─────────────────────────────────────────┘
```

---

## Reward Assignment Comparison

### Single-Agent
```python
# Reward assigned to steps in single trajectory
trajectory.steps[i].reward = compute_reward(...)

Episode(
    trajectories=[trajectory],
    reward=final_reward
)
```

### Multi-Agent
```python
# Different rewards for different agents
generator_trajectory.steps[0].reward = gen_reward
test_runner_trajectory.steps[0].reward = test_reward
refiner_trajectory.steps[0].reward = refine_reward

Episode(
    trajectories=[
        generator_trajectory,
        test_runner_trajectory,
        refiner_trajectory
    ],
    metrics={
        "generator_success": ...,
        "test_runner_accuracy": ...,
        "refiner_success": ...
    }
)
```

---

## Metrics Comparison

### Single-Agent Metrics
```python
{
    "reward": 1.0,
    "is_correct": True,
    "num_steps": 5,
}
```

### Multi-Agent Metrics
```python
{
    "generator_success": 0.0,      # Initial attempt failed
    "test_runner_analyses": 2,      # Ran 2 test analyses
    "refiner_attempts": 2,          # Made 2 refinement attempts
    "refiner_success_rate": 0.5,    # 1 out of 2 succeeded
    "total_iterations": 2,
    "final_success": 1              # Final solution correct
}
```

**Benefit**: Can track which agent contributed to success/failure

---

## Training Implications

### Single-Agent Training
```python
# All steps trained together
# Reward applied to entire trajectory
optimizer.update(trajectory, reward)
```

### Multi-Agent Training
```python
# Can train agents separately or jointly

# Option 1: Joint training
optimizer.update([gen_traj, test_traj, refine_traj], rewards)

# Option 2: Separate training
gen_optimizer.update(gen_traj, gen_reward)
test_optimizer.update(test_traj, test_reward)
refine_optimizer.update(refine_traj, refine_reward)

# Option 3: Different reward functions per agent
gen_traj.reward = accuracy_reward(...)
test_traj.reward = diagnostic_quality_reward(...)
refine_traj.reward = improvement_reward(...)
```

---

## Performance Characteristics

| Aspect | Single-Agent | Multi-Agent |
|--------|-------------|-------------|
| **Complexity** | Lower | Higher |
| **Modularity** | Lower | Higher |
| **Debugging** | Harder to identify failure points | Easier to track which agent failed |
| **Latency** | Lower (fewer LLM calls) | Higher (more LLM calls) |
| **Parallelization** | Limited | Can parallelize independent agents |
| **Specialization** | One model does everything | Each agent can specialize |
| **Training** | Simpler | More complex but more targeted |
| **Interpretability** | Lower | Higher (clear agent roles) |

---

## When to Use Each

### Use Single-Agent When:
- Task is straightforward and doesn't decompose naturally
- Low latency is critical
- Simplicity is preferred
- Limited computational resources

**Examples**:
- Simple math problems
- Single-turn Q&A
- Straightforward classification

### Use Multi-Agent When:
- Task naturally decomposes into sub-tasks
- Different sub-tasks benefit from specialization
- Want to track which component succeeded/failed
- Can afford additional LLM calls
- Need parallel execution of independent sub-tasks

**Examples**:
- Code generation with debugging (generate → test → refine)
- Research questions (plan → gather → synthesize)
- Complex software engineering (analyze → implement → validate)
- Information retrieval (optimize queries → retrieve documents → extract answer)

---

## Migration Path

To migrate from single-agent to multi-agent:

1. **Identify natural decomposition**
   - What distinct roles exist?
   - Which steps could benefit from specialization?

2. **Create agent classes**
   ```python
   class Agent1:
       async def task1(self, input):
           output = await self.rollout_engine.get_model_response(
               messages,
               agent_name="agent1"  # ← Add this
           )
           return Trajectory(name="agent1", ...)  # ← And this
   ```

3. **Create workflow**
   ```python
   class MultiAgentWorkflow(Workflow):
       async def run(self, task, uid):
           traj1 = await self.agent1.task1(...)
           traj2 = await self.agent2.task2(...)
           return Episode(trajectories=[traj1, traj2])
   ```

4. **Update execution**
   ```python
   # Change from AgentExecutionEngine to WorkflowExecutionEngine
   engine = WorkflowExecutionEngine(
       workflow_class=MultiAgentWorkflow,
       ...
   )
   ```

---

## Summary

**Single-Agent**: One agent, one trajectory, simpler but less modular

**Multi-Agent**: Multiple agents, multiple trajectories, more complex but more flexible and interpretable

The choice depends on task complexity, need for specialization, and computational constraints.
