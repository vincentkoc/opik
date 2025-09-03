from typing import Optional

# Centralized prompt templates used by EvolutionaryOptimizer. This file contains
# only string builders and constants; it has no side effects.


INFER_STYLE_SYSTEM_PROMPT = """You are an expert in linguistic analysis and prompt engineering. Your task is to analyze a few input-output examples from a dataset and provide a concise, actionable description of the desired output style. This description will be used to guide other LLMs in generating and refining prompts.

Focus on characteristics like:
- **Length**: (e.g., single word, short phrase, one sentence, multiple sentences, a paragraph)
- **Tone**: (e.g., factual, formal, informal, conversational, academic)
- **Structure**: (e.g., direct answer first, explanation then answer, list, yes/no then explanation)
- **Content Details**: (e.g., includes only the answer, includes reasoning, provides examples, avoids pleasantries)
- **Keywords/Phrasing**: Any recurring keywords or phrasing patterns in the outputs.

Provide a single string that summarizes this style. This summary should be directly usable as an instruction for another LLM.
For example: 'Outputs should be a single, concise proper noun.' OR 'Outputs should be a short paragraph explaining the reasoning, followed by a direct answer, avoiding conversational pleasantries.' OR 'Outputs are typically 1-2 sentences, providing a direct factual answer.'
Return ONLY this descriptive string, with no preamble or extra formatting.
"""


def semantic_mutation_system_prompt(output_style_guidance: Optional[str]) -> str:
    style = (
        output_style_guidance
        or "Produce clear, effective, and high-quality responses suitable for the task."
    )
    return (
        "You are a prompt engineering expert. Your goal is to modify prompts to improve their "
        f"effectiveness in eliciting specific types of answers, particularly matching the style: '{style}'. "
        "Follow the specific modification instruction provided."
    )


def synonyms_system_prompt() -> str:
    return (
        "You are a helpful assistant that provides synonyms. Return only the synonym word, "
        "no explanation or additional text."
    )


def rephrase_system_prompt() -> str:
    return (
        "You are a helpful assistant that rephrases text. Return only the modified phrase, "
        "no explanation or additional text."
    )


def fresh_start_system_prompt(output_style_guidance: Optional[str]) -> str:
    style = (
        output_style_guidance
        or "Produce clear, effective, and high-quality responses suitable for the task."
    )
    return (
        "You are an expert prompt engineer. Your task is to generate novel, effective prompts from scratch "
        "based on a task description, specifically aiming for prompts that elicit answers in the style: "
        f"'{style}'. Output ONLY a raw JSON list of strings."
    )


def variation_system_prompt(output_style_guidance: Optional[str]) -> str:
    style = (
        output_style_guidance
        or "Produce clear, effective, and high-quality responses suitable for the task."
    )
    return f"""You are an expert prompt engineer specializing in creating diverse and effective prompts. Given an initial prompt, your task is to generate a diverse set of alternative prompts.

For each prompt variation, consider:
1. Different levels of specificity and detail, including significantly more detailed and longer versions.
2. Various ways to structure the instruction, exploring more complex sentence structures and phrasings.
3. Alternative phrasings that maintain the core intent but vary in style and complexity.
4. Different emphasis on key components, potentially elaborating on them.
5. Various ways to express constraints or requirements.
6. Different approaches to clarity and conciseness, but also explore more verbose and explanatory styles.
7. Alternative ways to guide the model's response format.
8. Consider variations that are substantially longer and more descriptive than the original.

The generated prompts should guide a target LLM to produce outputs in the following style: '{style}'

Return a JSON array of prompts with the following structure:
{{
    "prompts": [
        {{
            "prompt": "alternative prompt 1",
            "strategy": "brief description of the variation strategy used, e.g., 'focused on eliciting specific output style'"
        }},
        {{
            "prompt": "alternative prompt 2",
            "strategy": "brief description of the variation strategy used"
        }}
    ]
}}
Each prompt variation should aim to get the target LLM to produce answers matching the desired style: '{style}'.
"""


def llm_crossover_system_prompt(output_style_guidance: Optional[str]) -> str:
    style = (
        output_style_guidance
        or "Produce clear, effective, and high-quality responses suitable for the task."
    )
    return f"""You are an expert prompt engineer specializing in creating novel prompts by intelligently blending existing ones.
Given two parent prompts, your task is to generate one or two new child prompts that effectively combine the strengths, styles, or core ideas of both parents.
The children should be coherent and aim to explore a potentially more effective region of the prompt design space, with a key goal of eliciting responses from the target language model in the following style: '{style}'.

Consider the following when generating children:
- Identify the key instructions, constraints, and desired output formats in each parent, paying attention to any hints about desired output style.
- Explore ways to merge these elements such that the resulting prompt strongly guides the target LLM towards the desired output style.
- You can create a child that is a direct blend, or one that takes a primary structure from one parent and incorporates specific elements from the other, always optimizing for clear instruction towards the desired output style.
- If generating two children, try to make them distinct from each other and from the parents, perhaps by emphasizing different aspects of the parental combination that could lead to the desired output style.

All generated prompts must aim for eliciting answers in the style: '{style}'.

Return a JSON object that is a list of both child prompts. Each child prompt is a list of LLM messages. Example:
[
    {{"role": "<role>", "content": "<content>"}},
    {{"role": "<role>", "content": "<content>"}}
]


"""


def radical_innovation_system_prompt(output_style_guidance: Optional[str]) -> str:
    style = (
        output_style_guidance
        or "Produce clear, effective, and high-quality responses suitable for the task."
    )
    return f"""You are an expert prompt engineer and a creative problem solver.
Given a task description and an existing prompt for that task (which might be underperforming), your goal is to generate a new, significantly improved, and potentially very different prompt.
Do not just make minor edits. Think about alternative approaches, structures, and phrasings that could lead to better performance.
Consider clarity, specificity, constraints, and how to best guide the language model for the described task TO PRODUCE OUTPUTS IN THE FOLLOWING STYLE: '{style}'.
Return only the new prompt string, with no preamble or explanation.
"""
