"""
prompt_builder.py — Semantic Prompt Construction for LLM Interventions
=======================================================================

Builds structured, context-rich prompts from training state for the LLM
to analyze and generate reward shaping adjustments.

Supports multiple prompt templates for different intervention types:
    - reward_reshaping: Adjust subtask weights
    - sub_objective: Assign a new focus area
    - diagnosis: Analyze what's going wrong
"""

from typing import Optional


class PromptBuilder:
    """
    Constructs LLM prompts from training metrics and state context.
    """

    # DAG description shared across all templates
    DAG_DESCRIPTION = (
        "Strict Dependency DAG: (Wood+Stone) -> Pickaxe -> Iron -> "
        "(Sword+Armor) -> Bridge -> Defeat Enemy -> Gold.\n"
        "Agent 0 (Lumberjack) collects Wood and builds Bridge.\n"
        "Agent 1 (Miner) collects Stone, mines Iron, crafts Sword/Armor, defeats Enemy, mines Gold.\n"
        "Both agents must be at Workbench to craft, and both at Bridge location to build."
    )

    def build_intervention_prompt(
        self,
        metrics: dict,
        td_stats: dict,
        trigger_reason: str,
        trigger_count: int,
        template: str = "reward_reshaping",
    ) -> str:
        """
        Build a complete intervention prompt.

        Args:
            metrics: Training metrics dict with success_rate, subtask_pcts, etc.
            td_stats: TD error statistics.
            trigger_reason: Human-readable reason the trigger fired.
            trigger_count: How many times the trigger has fired so far.
            template: Prompt template to use.

        Returns:
            Complete prompt string.
        """
        if template == "reward_reshaping":
            return self._build_reshaping_prompt(metrics, td_stats, trigger_reason, trigger_count)
        elif template == "sub_objective":
            return self._build_sub_objective_prompt(metrics, td_stats, trigger_reason)
        elif template == "diagnosis":
            return self._build_diagnosis_prompt(metrics, td_stats, trigger_reason)
        else:
            raise ValueError(f"Unknown template: {template}")

    def build_hrl_prompt(self, inventory, a0_status, a1_status) -> str:
        """
        Builds a semantic Chain-of-Thought HRL prompt to assign discrete options.
        """
        return f"""
    You are the high-level Cognitive Orchestrator for two MARL agents.
    Strict Dependency DAG: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Enemy -> Gold.

    ### CURRENT STATE:
    Inventory: {inventory}
    Agent 0 Status: {a0_status}
    Agent 1 Status: {a1_status}

    ### YOUR TASK:
    Assign exactly ONE discrete Option to each agent to progress the DAG.
    Available Options: ["COLLECT_WOOD", "COLLECT_STONE", "CRAFT_PICKAXE", "MINE_IRON", "CRAFT_SWORD", "CRAFT_ARMOR", "BUILD_BRIDGE", "FIGHT_ENEMY", "COLLECT_GOLD", "IDLE"]

    Respond ONLY with valid JSON exactly matching this schema:
    {{
      "dag_check": "<1 sentence reasoning verifying you have the required inventory for the assigned options>",
      "agent_0_option": "<Option>",
      "agent_1_option": "<Option>"
    }}
    """

    def _build_reshaping_prompt(
        self,
        metrics: dict,
        td_stats: dict,
        trigger_reason: str,
        trigger_count: int,
    ) -> str:
        """Build a reward reshaping prompt."""
        subtask_pcts = metrics.get("subtask_pcts", {})
        current_weights = metrics.get("adaptive_weights", {})
        success_rate = metrics.get("success_rate", 0.0)
        avg_reward = metrics.get("avg_env_reward", 0.0)
        update = metrics.get("update", 0)

        # Format subtask completion rates
        subtask_lines = []
        for name in ["wood", "stone", "pickaxe", "iron", "sword", "armor", "bridge", "enemy", "gold"]:
            pct = subtask_pcts.get(name, 0.0) * 100
            subtask_lines.append(f"  * {name.capitalize()}: {pct:.1f}%")
        subtask_block = "\n".join(subtask_lines)

        # Format current weights
        weight_lines = []
        for k in ['w_wood', 'w_stone', 'w_workbench', 'w_iron', 'w_bridge', 'w_enemy', 'w_gold']:
            v = current_weights.get(k, 1.0)
            weight_lines.append(f"  * {k}: {v:.3f}")
        weight_block = "\n".join(weight_lines)

        return f"""You are an expert RL reward designer for a cooperative multi-agent POMDP environment.

{self.DAG_DESCRIPTION}

### TRIGGER EVENT:
The system's automatic critic-monitoring has detected a learning problem.
Trigger #{trigger_count} at training update {update}.
Reason: {trigger_reason}

### CURRENT TRAINING STATE:
* Overall Success Rate (Gold Mined): {success_rate:.1%}
* Average Environment Reward: {avg_reward:.4f}

### SUBTASK COMPLETION RATES:
{subtask_block}

### CRITIC HEALTH (TD Error Statistics):
* Mean TD Error: {td_stats.get('mean_td_error', 0):.4f}
* TD Error Std Dev: {td_stats.get('std_td_error', 0):.4f}
* TD Error Variance: {td_stats.get('variance_td_error', 0):.6f}
* Agent 0 (Lumberjack) Mean TD: {td_stats.get('td_error_agent0_mean', 0):.4f}
* Agent 1 (Miner) Mean TD: {td_stats.get('td_error_agent1_mean', 0):.4f}

### CURRENT REWARD SHAPING WEIGHTS:
{weight_block}

### YOUR TASK:
1. Diagnose which part of the dependency chain is the bottleneck.
2. Explain your reasoning in 1-2 sentences.
3. Output updated reward shaping weights (0.0 to 3.0) to fix the bottleneck.
   - INCREASE weights for bottleneck subtasks to make them more attractive.
   - DECREASE weights for mastered subtasks to prevent reward hacking.
   - Consider which AGENT is struggling based on the per-agent TD errors.

### CRITICAL INSTRUCTION ON TRAPS:
The environment has "trap" goals (like Enemy and Gold) that have huge base rewards. Even if you give them a tiny weight like 0.1, the agents will be greedily distracted by them and get stuck!
Therefore, you MUST set the weight to EXACTLY 0.0 for any task that is currently impossible because its prerequisites haven't been met yet (e.g. w_gold and w_enemy MUST be 0.0 if the agents haven't built tools yet).

Respond ONLY with valid JSON matching this schema:
{{
  "reasoning": "<1-2 sentences explaining the bottleneck and your fix>",
  "w_wood": <float 0.0-3.0>,
  "w_stone": <float 0.0-3.0>,
  "w_workbench": <float 0.0-3.0>,
  "w_iron": <float 0.0-3.0>,
  "w_bridge": <float 0.0-3.0>,
  "w_enemy": <float 0.0-3.0>,
  "w_gold": <float 0.0-3.0>
}}"""

    def _build_sub_objective_prompt(
        self,
        metrics: dict,
        td_stats: dict,
        trigger_reason: str,
    ) -> str:
        """Build a sub-objective assignment prompt (for HRL Phase 4)."""
        subtask_pcts = metrics.get("subtask_pcts", {})
        subtask_block = "\n".join(
            f"  * {name.capitalize()}: {pct * 100:.1f}%"
            for name, pct in subtask_pcts.items()
        )

        return f"""You are an RL hierarchical controller for a cooperative MARL environment.

{self.DAG_DESCRIPTION}

### TRIGGER EVENT:
{trigger_reason}

### SUBTASK COMPLETION RATES:
{subtask_block}

### YOUR TASK:
Assign the next high-level Option for each agent. Choose from:
COLLECT_WOOD, COLLECT_STONE, CRAFT_PICKAXE, MINE_IRON, CRAFT_SWORD,
CRAFT_ARMOR, BUILD_BRIDGE, DEFEAT_ENEMY, MINE_GOLD, IDLE

Respond ONLY with valid JSON:
{{
  "reasoning": "<1 sentence>",
  "agent_0_option": "<OPTION_NAME>",
  "agent_1_option": "<OPTION_NAME>"
}}"""

    def _build_diagnosis_prompt(
        self,
        metrics: dict,
        td_stats: dict,
        trigger_reason: str,
    ) -> str:
        """Build a diagnostic prompt (for analysis/logging only)."""
        return f"""Analyze the following RL training state and diagnose the problem:

{self.DAG_DESCRIPTION}

Trigger Reason: {trigger_reason}
Success Rate: {metrics.get('success_rate', 0):.1%}
TD Error Variance: {td_stats.get('variance_td_error', 0):.6f}
Subtask Progress: {metrics.get('subtask_pcts', {})}

Respond with JSON: {{"diagnosis": "<detailed analysis>", "severity": "low|medium|high|critical"}}"""
