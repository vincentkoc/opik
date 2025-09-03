from typing import Any, Dict, List, Optional, Set

import json
import logging
import random

from deap import tools
from deap import creator as _creator

from . import prompts as evo_prompts
from . import reporting
from ..optimization_config import chat_prompt
from .. import utils


logger = logging.getLogger(__name__)
creator = _creator


class PopulationMixin:
    def _initialize_population(
        self, prompt: chat_prompt.ChatPrompt
    ) -> List[chat_prompt.ChatPrompt]:
        """Initialize the population with diverse variations of the initial prompt,
        including some 'fresh start' prompts based purely on task description.
        All generated prompts should aim to elicit answers matching self.output_style_guidance.
        """
        with reporting.initializing_population(verbose=self.verbose) as init_pop_report:
            init_pop_report.start(self.population_size)

            population = [prompt]
            if self.population_size <= 1:
                return population

            num_to_generate_total = self.population_size - 1
            num_fresh_starts = max(1, int(num_to_generate_total * 0.2))
            num_variations_on_initial = num_to_generate_total - num_fresh_starts

            task_desc_for_llm = self._get_task_description_for_llm(prompt)
            current_output_style_guidance = self.output_style_guidance

            # Fresh starts
            if num_fresh_starts > 0:
                init_pop_report.start_fresh_prompts(num_fresh_starts)
                fresh_start_user_prompt = f"""Here is a description of a task:
    {task_desc_for_llm}

    The goal is to generate prompts that will make a target LLM produce responses in the following style: '{current_output_style_guidance}'.

    Please generate {num_fresh_starts} diverse and effective prompt(s) for a language model to accomplish this task, ensuring they guide towards this specific output style.
    Focus on clarity, completeness, and guiding the model effectively towards the desired style. Explore different structural approaches.

    Example of valid response: [
        ["role": "<role>", "content": "<Prompt targeting specified style.>"],
        ["role": "<role>", "content": "<Another prompt designed for the output style.>"]
    ]

    Your response MUST be a valid JSON list of AI messages. Do NOT include any other text, explanations, or Markdown formatting like ```json ... ``` around the list.

    """
                try:
                    response_content = self._call_model(
                        messages=[
                            {
                                "role": "system",
                                "content": evo_prompts.fresh_start_system_prompt(
                                    current_output_style_guidance
                                ),
                            },
                            {"role": "user", "content": fresh_start_user_prompt},
                        ],
                        is_reasoning=True,
                    )

                    logger.debug(
                        f"Raw LLM response for fresh start prompts: {response_content}"
                    )

                    fresh_prompts = utils.json_to_dict(response_content)
                    if isinstance(fresh_prompts, list):
                        if all(isinstance(p, dict) for p in fresh_prompts) and all(
                            p.get("role") is not None for p in fresh_prompts
                        ):
                            population.append(
                                chat_prompt.ChatPrompt(messages=fresh_prompts)
                            )
                            init_pop_report.success_fresh_prompts(1)
                        elif all(isinstance(p, list) for p in fresh_prompts):
                            population.extend(
                                [
                                    chat_prompt.ChatPrompt(messages=p)
                                    for p in fresh_prompts[:num_fresh_starts]
                                ]
                            )
                            init_pop_report.success_fresh_prompts(
                                len(fresh_prompts[:num_fresh_starts])
                            )
                        else:
                            init_pop_report.failed_fresh_prompts(
                                num_fresh_starts,
                                f"LLM response for fresh starts was not a valid list of strings or was empty: {response_content}. Skipping fresh start prompts.",
                            )
                except json.JSONDecodeError as e_json:
                    init_pop_report.failed_fresh_prompts(
                        num_fresh_starts,
                        f"JSONDecodeError generating fresh start prompts: {e_json}. LLM response: '{response_content}'. Skipping fresh start prompts.",
                    )
                except Exception as e:
                    init_pop_report.failed_fresh_prompts(
                        num_fresh_starts,
                        f"Error generating fresh start prompts: {e}. Skipping fresh start prompts.",
                    )

            # Variations on the initial prompt
            if num_variations_on_initial > 0:
                init_pop_report.start_variations(num_variations_on_initial)
                user_prompt_for_variation = f"""Initial prompt:
    '''{prompt.get_messages()}'''

    Task context:
    {task_desc_for_llm}
    Desired output style from target LLM: '{current_output_style_guidance}'

    Generate {num_variations_on_initial} diverse alternative prompts based on the initial prompt above, keeping the task context and desired output style in mind.
    All generated prompt variations should strongly aim to elicit answers from the target LLM matching the style: '{current_output_style_guidance}'.
    For each variation, consider how to best achieve this style, e.g., by adjusting specificity, structure, phrasing, constraints, or by explicitly requesting it.

    Return a JSON array of prompts with the following structure:
    {{
        "prompts": [
            {{
                "prompt": [{{"role": "<role>", "content": "<content>"}}],
                "strategy": "brief description of the variation strategy used, e.g., 'direct instruction for target style'"
            }}
            // ... more prompts if num_variations_on_initial > 1
        ]
    }}
    Ensure a good mix of variations, all targeting the specified output style from the end LLM.

    Return a valid JSON object that is correctly escaped. Return nothing else, d`o not include any additional text or Markdown formatting.
    """
                try:
                    response_content_variations = self._call_model(
                        messages=[
                            {
                                "role": "system",
                                "content": evo_prompts.variation_system_prompt(
                                    current_output_style_guidance
                                ),
                            },
                            {"role": "user", "content": user_prompt_for_variation},
                        ],
                        is_reasoning=True,
                    )
                    logger.debug(
                        f"Raw response for population variations: {response_content_variations}"
                    )
                    json_response_variations = json.loads(response_content_variations)
                    generated_prompts_variations = [
                        p["prompt"]
                        for p in json_response_variations.get("prompts", [])
                        if isinstance(p, dict) and "prompt" in p
                    ]

                    if generated_prompts_variations:
                        init_pop_report.success_variations(
                            len(
                                generated_prompts_variations[:num_variations_on_initial]
                            )
                        )
                        population.extend(
                            [
                                chat_prompt.ChatPrompt(messages=p)
                                for p in generated_prompts_variations[
                                    :num_variations_on_initial
                                ]
                            ]
                        )
                    else:
                        init_pop_report.failed_variations(
                            num_variations_on_initial,
                            "Could not parse 'prompts' list for variations. Skipping variations.",
                        )
                except Exception as e:
                    init_pop_report.failed_variations(
                        num_variations_on_initial,
                        f"Error calling LLM for initial population variations: {e}",
                    )

            # Ensure population is of the required size using unique prompts
            final_population_set: Set[str] = set()
            final_population_list: List[chat_prompt.ChatPrompt] = []
            for p in population:
                if json.dumps(p.get_messages()) not in final_population_set:
                    final_population_set.add(json.dumps(p.get_messages()))
                    final_population_list.append(p)

            init_pop_report.end(final_population_list)
            return final_population_list[: self.population_size]

    def _should_restart_population(self, curr_best: float) -> bool:
        """Update internal counters and decide if we should trigger a population restart."""
        if self._best_primary_score_history:
            threshold = self._best_primary_score_history[-1] * (
                1 + self.DEFAULT_RESTART_THRESHOLD
            )
            if curr_best < threshold:
                self._gens_since_pop_improvement += 1
            else:
                self._gens_since_pop_improvement = 0
        self._best_primary_score_history.append(curr_best)
        return self._gens_since_pop_improvement >= self.DEFAULT_RESTART_GENERATIONS

    def _restart_population(
        self,
        hof: tools.HallOfFame,
        population: List[Any],
        best_prompt_so_far: chat_prompt.ChatPrompt,
    ) -> List[Any]:
        """Return a fresh, evaluated population seeded by elites."""
        if self.enable_moo:
            elites = list(hof)
        else:
            elites = tools.selBest(population, self.elitism_size)

        seed_prompt = (
            chat_prompt.ChatPrompt(
                messages=max(elites, key=lambda x: x.fitness.values[0])
            )
            if elites
            else best_prompt_so_far
        )

        prompt_variants = self._initialize_population(seed_prompt)
        new_pop = [creator.Individual(p.get_messages()) for p in prompt_variants]

        for ind, fit in zip(new_pop, map(self.toolbox.evaluate, new_pop)):
            ind.fitness.values = fit

        self._gens_since_pop_improvement = 0
        return new_pop

