import string
import json
import re
import os
import logging
import pickle
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from factscore.atomic_facts_ext import AtomicFactGenerator
from factscore.openai_lm import OpenAIModel

# Consolidated exclusion patterns using regex alternation and common keywords
exclusion_patterns = re.compile(
    r'\b(?:I\s+(?:cannot|was unable to|do not|have no)\s+(?:find|provide|locate|access)|'
    r'(?:No|Insufficient|Cannot|Unable)\s+(?:information|data|details|records|bio|profile|content)|'
    r'(?:I\s+am\s+not\s+sure|Note\s+(?:that|,))|'
    r'(?:No\s+(?:additional|further|relevant)\s+information))\b',
    flags=re.IGNORECASE
)

# Constants for relevance classification
SYMBOL = 'Foo'
NOT_SYMBOL = 'Not Foo'

_PROMPT_PLACEHOLDER = '[PROMPT]'
_RESPONSE_PLACEHOLDER = '[RESPONSE]'
_STATEMENT_PLACEHOLDER = '[ATOMIC FACT]'

_RELEVANCE_FORMAT = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Quoc is well-respected by many \
of his peers, such as Geoffrey Hinton, who is an adjunct professor at the \
University of Montreal and teaches courses on deep learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. The phrase "Quoc is well-respected by many of his peers, such \
as Geoffrey Hinton" from the RESPONSE shows that the relationship between Quoc \
Le and Geoffrey Hinton is that they are peers. For this reason, the subjects \
Quoc Le and Geoffrey Hinton are [{SYMBOL}].


Example 2:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Geoffrey Hinton is an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and Geoffrey Hinton is. Thus, the subjects Quoc Le and \
Geoffrey Hinton are [{NOT_SYMBOL}].


Your Task:
QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_REVISE_FORMAT = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
Acorns is a company.

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED STATEMENT:
The subject in the statement "Acorns is a company" is "Acorns". "Acorns" is \
not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "Acorns" is not a vague reference. Thus, the revised statement \
is:
```
Acorns is a company.
```


Example 2:
STATEMENT:
He teaches courses on deep learning.

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Le is also an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

REVISED STATEMENT:
The subject in the statement "He teaches course on deep learning" is "he". \
From the RESPONSE, we can see that this statement comes from the sentence "Le \
is also an adjunct professor at the University of Montreal, where he teaches \
courses on deep learning.", meaning that "he" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
Quoc Le teaches courses on deep learning.
```


Example 3:
STATEMENT:
The television series is called "You're the Worst."

RESPONSE:
Xochitl Gomez began her acting career in theater productions, and she made her \
television debut in 2016 with a guest appearance on the Disney Channel series \
"Raven's Home." She has also appeared in the television series "You're the \
Worst" and "Gentefied."

REVISED STATEMENT:
The subject of the statement "The television series is called "You're the \
Worst."" is "the television series". This is a reference to an unknown entity, \
since it is unclear what television series is "the television series". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "the television series" is a \
vague reference that should be replaced by "the television series that Xochitl \
Gomez appeared in". Thus, the revised response is:
```
The television series that Xochitl Gomez appeared in is called "You're the \
Worst."
```


Example 4:
STATEMENT:
Dean joined Google.

RESPONSE:
Jeff Dean is a Google Senior Fellow and the head of Google AI, leading \
research and development in artificial intelligence. Dean joined Google in \
1999 and has been essential to its continued development in the field.

REVISED STATEMENT:
The subject of the statement "Dean joined Google" is "Dean". From the \
response, we can see that "Dean" is the last name of "Jeff Dean". Therefore \
"Dean" is a non-full name, making it a vague reference. It should be replaced \
by "Jeff Dean", which is the full name. Thus, the revised response is:
```
Jeff Dean joined Google.
```


Your Task:
STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""


# Utility functions for extracting information from model responses
def extract_first_square_brackets(text):
    import re
    match = re.search(r'\[([^\[\]]+)\]', text)
    result = match.group(1) if match else None
    logging.debug(f"Extracted from square brackets: {result}")
    return result

def extract_first_code_block(text, ignore_language=False):
    if not isinstance(text, str):
        logging.error(f"Expected string in extract_first_code_block, got {type(text)}")
        return None
    import re
    pattern = r'```(?:[\w\+\-\.]+\n)?([\s\S]+?)```' if ignore_language else r'```[\w\+\-\.]+\n([\s\S]+?)```'
    match = re.search(pattern, text)
    result = match.group(1).strip() if match else None
    logging.debug(f"Extracted code block: {result}")
    return result

def strip_string(s):
    result = s.strip()
    logging.debug(f"Stripped string: {result}")
    return result

def post_process_generation(generated_text: str) -> str:
    """
    Post-process the generated text to ensure completeness and remove extraneous content.
    
    Args:
        generated_text (str): The raw text generated by the model.
    
    Returns:
        str: The cleaned and validated bio.
    """
    # Ensure the bio ends with a period. If not, truncate to the last complete sentence.
    if not generated_text.endswith('.'):
        last_period = generated_text.rfind('.')
        if last_period != -1:
            generated_text = generated_text[:last_period + 1]
    
    # Remove explanations about inability to find more information using the consolidated pattern
    gen = exclusion_patterns.split(generated_text)[0].strip()
    
    return gen

class FactProbeScorer(object):

    def __init__(self,
                 model_name="llama3.1+probe",
                 data_dir="/scratch/ms23jh/.cache/factscore",
                 cache_dir="/scratch/ms23jh/.cache/factscore",
                 openai_key="api.key",
                 abstain_detection_type=None,
                 batch_size=256,
                 probe_path=None,
                 hf_model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                 device='cuda',
                 max_memory='80GIB'):
        self.model_name = model_name

        self.batch_size = batch_size
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None

        self.device = device

        # Initialize the LLaMA 3.1 model
        logging.debug("Initializing LLaMA 3.1 model...")
        self.tokenizer, self.model = self.initialize_model(hf_model_name, device=device, max_memory=max_memory)
        self.model.eval()

        # Load the trained probe and other information from a pickle file
        self.probe = None
        self.probe_path = probe_path
        if probe_path:
            logging.debug(f"Loading trained probe and configurations from {probe_path}...")
            with open(probe_path, 'rb') as f:
                probe_data = pickle.load(f)
                self.probe = probe_data['probe']
                self.layer_group = probe_data['layer_group']  # (layer_start, layer_end)
                self.token_pos = probe_data['token_pos']  # e.g., 'lt' or 'slt'
                # Load any other necessary configurations
        else:
            logging.error("Probe path not provided or file not found. Cannot proceed without a trained probe.")
            raise ValueError("Probe path must be provided.")

        # Initialize the model for relevance classification and fact refinement
        logging.debug("Initializing relevance and refinement model...")
        self.relevance_lm = OpenAIModel("ChatGPT",
                                        cache_file=os.path.join(cache_dir, "RelevanceChatGPT.pkl"),
                                        key_path=openai_key)

    def initialize_model(self, hf_model_name, device='cuda', max_memory='80GIB'):
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            device_map='auto',
            max_memory={0: max_memory}
        )
        logging.info(f'Model "{hf_model_name}" loaded on device "{device}".')
        return tokenizer, model
    
    def process_generation_yield(self, topic, generation):
        """Process a single generation and return structured results."""
        # Extract atomic claims and token indices
        atomic_facts_with_spans = self.extract_atomic_facts(generation)
        prior_sent = None
        sent_facts = []

        for atom_data in atomic_facts_with_spans:
            atom_text = atom_data['fact'].strip()
            token_indices = atom_data['token_indices']
            sentence = atom_data['sent_text']

            # Step 1: Revise the atomic fact
            revised_atom = self.revise_fact(generation, atom_text)
            logging.info(f"Revised atomic fact: {revised_atom}")

            # Step 2: Check relevance (commented out as per original code)
            is_relevant = self.check_relevance(topic, sentence, revised_atom)
            logging.info(f"Is relevant: {is_relevant}")

            # if not is_relevant:
            #     # Skip irrelevant facts
            #     logging.info("Atomic fact is irrelevant, skipping...")
            #     continue

            # Get predicted score from the linear probe
            is_supported, support_prob = self.predict_support(revised_atom)

            # Collect the result in a structured format
            result = {
                'sentence': sentence,
                'original_atomic_claim': atom_text,
                'revised_atomic_claim': revised_atom,
                'token_indices': token_indices,
                'is_relevant': is_relevant,  
                'support_probability': support_prob,
                'is_supported': is_supported
            }

            # Check if the current sentence is different from the prior sentence
            if prior_sent and prior_sent != sentence:
                # Yield the facts of the prior sentence
                yield sent_facts, prior_sent
                # Reset the sent_facts for the new sentence
                sent_facts = [result]
                prior_sent = sentence
            else:
                # Append the result to sent_facts
                sent_facts.append(result)
                # Set the prior_sent if it's not already set
                if not prior_sent:
                    prior_sent = sentence

        # After the loop ends, yield any remaining facts
        if sent_facts and prior_sent:
            yield sent_facts, prior_sent


    def process_generation(self, topic, generation):
        """Process a single generation and return structured results."""
        all_results = []
        atomic_facts_with_spans = self.extract_atomic_facts(generation)
        for atom_data in atomic_facts_with_spans:
            atom_text = atom_data['fact'].strip()
            token_indices = atom_data['token_indices']
            sentence = atom_data['sent_text']

            # Step 1: Revise the atomic fact
            revised_atom = self.revise_fact(sentence, atom_text)
            logging.info(f"Revised atomic fact: {revised_atom}")

            # Step 2: Check relevance
            is_relevant = self.check_relevance(topic, sentence, revised_atom)
            logging.info(f"Is relevant: {is_relevant}")

            if not is_relevant:
                # Skip irrelevant facts
                logging.info("Atomic fact is irrelevant, skipping...")
                continue

            # Get predicted score from the linear probe
            is_supported, support_prob = self.predict_support(revised_atom)
            
            print(f"Sentence: {sentence}")
            print(f"  Original Atomic Claim: {atom_text}")
            print(f"  Revised Atomic Claim: {revised_atom}")
            print(f"  Token Indices: {token_indices}")
            print(f"  Is Relevant: {is_relevant}")
            print(f"  Support Probability: {support_prob:.4f}")
            print(f"  Is Supported: {is_supported}")
            print("-" * 50)

            # Collect the result in a structured format
            result = {
                'sentence': sentence,
                'original_atomic_claim': atom_text,
                'revised_atomic_claim': revised_atom,
                'token_indices': token_indices,
                'is_relevant': is_relevant,
                'support_probability': support_prob,
                'is_supported': is_supported
            }
            all_results.append(result)
        return all_results


    def extract_atomic_facts(self, generation):
        """Extract atomic facts from a sentence along with token indices."""
        if self.af_generator is None:
            logging.debug("Initializing AtomicFactGenerator...")
            self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                    demon_dir=os.path.join(self.data_dir, "demos"),
                                                    gpt3_cache_file=os.path.join(self.cache_dir, "GPT4o.pkl"))
        # Use the atomic fact generator to get facts with token spans
        atomic_facts_pairs, _ = self.af_generator.run(generation)
        atomic_facts_with_spans = []
        for sent_text, facts in atomic_facts_pairs:
            for fact_text, token_indices in facts:
                atomic_facts_with_spans.append({
                    'fact': fact_text,
                    'token_indices': token_indices,
                    'sent_text': sent_text
                })
        return atomic_facts_with_spans

    def revise_fact(self, response, atomic_fact):
        # Use the model to revise the atomic fact
        full_prompt = _REVISE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
        full_prompt = strip_string(full_prompt)
        logging.debug(f"Prompt for revising fact: {full_prompt}")

        try:
            model_response = self.relevance_lm.generate(full_prompt)
            logging.debug(f"Model response for revising fact: {model_response}")
            if isinstance(model_response, tuple):
                model_response = model_response[0]
            if not isinstance(model_response, str):
                logging.error(f"Model response is not a string: {type(model_response)}")
                model_response = None
        except Exception as e:
            logging.error(f"Error generating revised fact: {e}")
            model_response = None

        if model_response is None:
            logging.warning("Failed to obtain model response, using original atomic fact.")
            return atomic_fact

        revised_fact = extract_first_code_block(model_response, ignore_language=True)
        if revised_fact:
            return revised_fact.strip()
        else:
            # If the revision fails, use the original atomic fact
            logging.warning("Failed to revise atomic fact, using original.")
            return atomic_fact

    def check_relevance(self, prompt, response, atomic_fact):
        # Use the model to check if the atomic fact is relevant
        full_prompt = _RELEVANCE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
        full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
        full_prompt = strip_string(full_prompt)
        logging.debug(f"Prompt for checking relevance: {full_prompt}")

        model_response = self.relevance_lm.generate(full_prompt)[0]
        logging.debug(f"Model response for checking relevance: {model_response}")
        answer = extract_first_square_brackets(model_response)
        if answer:
            is_relevant = answer.strip().lower() == SYMBOL.lower()
            logging.debug(f"Extracted answer: {answer}, Is relevant: {is_relevant}")
            return is_relevant
        else:
            # If we cannot determine relevance, assume it's relevant
            logging.warning("Failed to determine relevance, assuming relevant.")
            return True

    def predict_support(self, atomic_fact):
        """Predict if the atomic fact is supported using the linear probe."""
        # Tokenize the atomic fact
        inputs = self.tokenizer(atomic_fact, return_tensors='pt').to(self.device)

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of hidden states from all layers

        # Extract the specified layers and concatenate them
        layer_start, layer_end = self.layer_group  # e.g., (16, 20)
        selected_hidden_states = hidden_states[layer_start:layer_end+1]  # +1 because the end index is inclusive

        # Handle 'slt' (second last token) and 'lt' (last token)
        if self.token_pos == 'lt':
            token_index = inputs['input_ids'].shape[1] - 1
        elif self.token_pos == 'slt':
            token_index = inputs['input_ids'].shape[1] - 2
        else:
            raise ValueError(f"Unsupported token position: {self.token_pos}")

        concatenated_hidden_state = torch.cat([h[:, token_index, :] for h in selected_hidden_states], dim=-1)
        # Shape: [batch_size, concatenated_hidden_size]

        # Convert to numpy
        concatenated_hidden_state_np = concatenated_hidden_state.cpu().numpy()

        # Use the probe to predict
        support_prob = self.probe.predict_proba(concatenated_hidden_state_np)[0, 1]  # Assuming binary classification
        is_supported = support_prob > 0.5  
        return is_supported, support_prob
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.5):
        """Generates a response to the prompt using the model, limited to 512 tokens."""
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate response
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,   # Limit to 512 tokens
            do_sample=True,                  # Enable sampling for more diverse responses
            temperature=temperature,         # Control randomness in sampling
            top_p=0.95,                      # Nucleus sampling to consider top 95% probability mass
            eos_token_id=self.tokenizer.eos_token_id  # Ensure proper end of sequence
        )
        
        # Decode the output tokens to get the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the generated response (excluding the prompt)
        response = post_process_generation(generated_text[len(prompt):].strip())
        return response


def main():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Initialize the FactProbeScorer with sample configurations
    fs = FactProbeScorer(
        model_name="llama3.1+probe",
        data_dir="/scratch/ms23jh/.cache/factscore",
        cache_dir="/scratch/ms23jh/.cache/factscore",
        openai_key="api.key",
        abstain_detection_type=None,
        probe_path="../../metrics/Llama3.1-8B_best_probe_layers_13-17_C_0.5.pkl",  # Replace with your probe file path
        hf_model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
        device='cuda',
        max_memory='80GIB'
    )

    # Sample topic and generation
    topic = "Who is Michael Bronstein?"
    generation = fs.generate_response(topic)

    # Process the generation
    fs.process_generation(topic, generation)

if __name__ == '__main__':
    main()
