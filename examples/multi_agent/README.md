# Multi-Agent Examples

This directory contains multi-agent implementations of high-potential examples identified for transformation from single-agent to multi-agent systems.

## Overview

Multi-agent systems in this codebase are characterized by:
1. **Multiple Trajectory objects** with different `name` values
2. **Different `agent_name`** passed to `get_model_response()`
3. **All trajectories combined** in a single `Episode`

Each agent has a specific role and collaborates with other agents to solve complex tasks.

## Examples

### 1. Multi-Agent DeepCoder (`deepcoder/`)

**Problem**: Competitive programming tasks

**Agents**:
- **CodeGenerator**: Creates initial code solution
- **TestRunner**: Executes tests and identifies failures
- **CodeRefiner**: Fixes bugs based on test feedback

**Pattern**: Generator → TestRunner → Refiner (iterative refinement loop)

**Key Features**:
- Iterative debugging similar to `math_reasoning` pattern
- Test-driven refinement
- Tracks which agent contributed to final solution

**Usage**:
```bash
python examples/multi_agent/deepcoder/run_multi_agent_deepcoder.py
```

**Metrics**:
- `generator_success`: Success rate of initial generation
- `refiner_success_rate`: Success rate of refinement attempts
- `total_iterations`: Number of refinement cycles needed

---

### 2. Multi-Agent SWE (`swe/`)

**Problem**: Software engineering tasks (bug fixes, feature implementation)

**Agents**:
- **IssueAnalyzer**: Analyzes bug/feature request and creates implementation plan
- **CodeWriter**: Implements code changes based on the plan
- **TestValidator**: Validates changes by running tests

**Pattern**: Analyzer → Writer → Validator (iterative refinement loop)

**Key Features**:
- Separation of planning, implementation, and validation
- Iterative refinement based on test feedback
- Structured approach to software engineering tasks

**Usage**:
```bash
python examples/multi_agent/swe/run_multi_agent_swe.py
```

**Metrics**:
- `analyzer_runs`: Number of analysis phases
- `writer_attempts`: Number of code writing attempts
- `validator_checks`: Number of validation checks performed
- `tests_passed`: Whether final solution passed tests

---

### 3. Multi-Agent Math Tool (`math_tool/`)

**Problem**: Mathematical problem solving with Python tools

**Agents**:
- **ProblemAnalyzer**: Understands problem and plans solution approach
- **CodeExecutor**: Writes and executes Python code
- **AnswerVerifier**: Validates solution correctness

**Pattern**: Analyzer → Executor → Verifier (iterative refinement loop)

**Key Features**:
- Combines mathematical reasoning with code execution
- Verification step catches calculation errors
- Tool-augmented problem solving

**Usage**:
```bash
python examples/multi_agent/math_tool/run_multi_agent_math_tool.py
```

**Metrics**:
- `analyzer_runs`: Number of problem analysis phases
- `executor_attempts`: Number of code execution attempts
- `verifier_checks`: Number of verification checks
- `solution_correct`: Whether final answer is correct

---

### 4. Multi-Agent DeepResearch (`deepresearch/`)

**Problem**: Comprehensive research questions requiring multi-source information

**Agents**:
- **QueryPlanner**: Breaks complex questions into sub-queries
- **InformationGatherer**: Retrieves information using tools (search, web, etc.)
- **AnswerSynthesizer**: Combines findings into coherent answer

**Pattern**: Planner → Gatherer(s) [parallel] → Synthesizer

**Key Features**:
- Parallel information gathering for multiple sub-queries
- Tool-based research (search, scholar, web visit, etc.)
- Synthesis of multi-source information

**Usage**:
```bash
python examples/multi_agent/deepresearch/run_multi_agent_deepresearch.py
```

**Metrics**:
- `num_subqueries`: Number of sub-questions generated
- `gatherer_runs`: Number of information gathering operations
- `total_tool_calls`: Total tool invocations across all gatherers
- `answer_length`: Length of synthesized answer

---

### 5. Multi-Agent Search (`search/`)

**Problem**: Information retrieval and question answering tasks

**Agents**:
- **QueryOptimizer**: Analyzes question and generates optimized search queries
- **DocumentRetriever**: Uses search tools to retrieve relevant documents
- **AnswerExtractor**: Extracts and synthesizes final answer from retrieved documents

**Pattern**: QueryOptimizer → DocumentRetriever(s) [parallel] → AnswerExtractor

**Key Features**:
- Query optimization for better retrieval
- Parallel document retrieval for multiple queries
- Answer extraction and synthesis from multiple sources
- Optional refinement based on answer quality

**Usage**:
```bash
python examples/multi_agent/search/run_multi_agent_search.py
```

**Metrics**:
- `num_queries_generated`: Number of optimized queries generated
- `total_searches`: Total number of search operations performed
- `extractor_attempts`: Number of answer extraction attempts
- `answer_correct`: Whether final answer is correct

---

## Comparison with Single-Agent Examples

### DeepCoder
- **Single-agent** (`examples/deepcoder/`): Single `CompetitionCodingAgent` generates code
- **Multi-agent** (`examples/multi_agent/deepcoder/`): Separate generator, test runner, and refiner agents collaborate

### SWE
- **Single-agent** (`examples/swe/`): Single `SWEAgent` handles entire workflow
- **Multi-agent** (`examples/multi_agent/swe/`): Separate analyzer, writer, and validator agents

### Math Tool
- **Single-agent** (`examples/math_tool/`): Single `ToolAgent` handles problem solving
- **Multi-agent** (`examples/multi_agent/math_tool/`): Separate analyzer, executor, and verifier agents

### DeepResearch
- **Single-agent** (`examples/deepresearch/`): Single `MultiTurnReactAgent` handles research
- **Multi-agent** (`examples/multi_agent/deepresearch/`): Separate planner, gatherer(s), and synthesizer agents

### Search
- **Single-agent** (`examples/search/`): Single `ToolAgent` handles search and answer extraction
- **Multi-agent** (`examples/multi_agent/search/`): Separate query optimizer, document retriever(s), and answer extractor agents

---

## Multi-Agent Patterns

### Pattern 1: Sequential Refinement
**Examples**: DeepCoder, SWE, Math Tool

**Flow**: Generator/Writer → Evaluator/Validator → Refiner → (loop)

**Characteristics**:
- Initial generation
- Validation/testing
- Iterative refinement based on feedback
- Continues until success or max iterations

### Pattern 2: Parallel Gathering + Synthesis
**Examples**: DeepResearch, Search

**Flow**: Planner/Optimizer → [Gatherer/Retriever, Gatherer/Retriever, ...] → Synthesizer/Extractor

**Characteristics**:
- Planning/optimization phase creates sub-tasks or queries
- Multiple agents work in parallel
- Synthesis combines all results
- More efficient for decomposable tasks

### Pattern 3: Generate-Evaluate-Select
**Examples**: solver_judge (from existing examples)

**Flow**: [Generator, Generator, ...] → Judge

**Characteristics**:
- Multiple parallel generations
- Single evaluator selects best
- No iterative refinement
- Good for exploration/diversity

---

## Key Differences from Single-Agent

### Trajectory Structure

**Single-Agent**:
```python
Episode(
    trajectories=[single_trajectory]  # One trajectory, name="agent"
)
```

**Multi-Agent**:
```python
Episode(
    trajectories=[
        generator_trajectory,   # name="generator"
        evaluator_trajectory,   # name="evaluator"
        refiner_trajectory      # name="refiner"
    ]
)
```

### Agent Name in Model Calls

**Single-Agent**:
```python
await rollout_engine.get_model_response(messages)  # No agent_name
```

**Multi-Agent**:
```python
await rollout_engine.get_model_response(messages, agent_name="generator")
await rollout_engine.get_model_response(messages, agent_name="evaluator")
```

---

## Benefits of Multi-Agent Approach

1. **Specialization**: Each agent focuses on a specific task
2. **Modularity**: Easier to modify/improve individual agents
3. **Transparency**: Clear attribution of which agent contributed what
4. **Debugging**: Can track performance of each agent separately
5. **Parallelization**: Some agents can work concurrently
6. **Training**: Can apply different rewards/training to different agents

---

## Training Multi-Agent Systems

Multi-agent workflows can be trained using the standard training pipeline:

```python
from rllm.trainer.workflow_trainer import WorkflowTrainer

trainer = WorkflowTrainer(
    workflow_class=MultiAgentDeepCoderWorkflow,
    workflow_args={
        "reward_function": code_reward_fn,
        "max_refinement_iterations": 3,
    },
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)
trainer.train()
```

**Key considerations**:
- Each agent can have its own reward signal
- Can train agents jointly or separately
- Metrics track individual agent performance

---

## Extending to Other Examples

To convert a single-agent example to multi-agent:

1. **Identify roles**: What distinct roles exist in the task?
2. **Create agent classes**: One class per role (e.g., Generator, Evaluator)
3. **Define trajectories**: Each agent creates a `Trajectory` with unique `name`
4. **Set agent_name**: Pass `agent_name` in `get_model_response()` calls
5. **Orchestrate workflow**: Combine trajectories in a `Workflow.run()` method
6. **Return Episode**: Include all trajectories in final `Episode`

Example structure:
```python
class AgentA:
    async def do_something(self, input: str) -> Trajectory:
        output = await self.rollout_engine.get_model_response(
            messages,
            agent_name="agent_a"  # <-- Important!
        )
        return Trajectory(name="agent_a", steps=[...])

class MultiAgentWorkflow(Workflow):
    async def run(self, task: dict, uid: str) -> Episode:
        traj_a = await self.agent_a.do_something(...)
        traj_b = await self.agent_b.do_something(...)

        return Episode(
            id=uid,
            task=task,
            trajectories=[traj_a, traj_b],  # <-- Multiple trajectories
            ...
        )
```

---

## Future Work

Potential additional multi-agent examples:
- **Countdown**: Solver-Judge pattern (like solver_judge)
- **AppWorld**: Planner-Executor-Validator pattern
- **Terminal**: Command-Planner-Executor-Checker pattern

---

## References

- Existing multi-agent examples:
  - `examples/solver_judge/` - Generate-and-Judge pattern
  - `examples/math_reasoning/` - Sequential refinement pattern

- Related documentation:
  - rLLM workflows: `rllm/workflows/workflow.py`
  - Agent trajectories: `rllm/agents/agent.py`
  - Training: `rllm/trainer/workflow_trainer.py`
