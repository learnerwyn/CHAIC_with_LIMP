from openai import OpenAI
import json
import math
import re

client = OpenAI(
    api_key=''
)

def parse_hypothesis(question):
    """This functions takes in a question (string) about the scenario with three hypotheses as choices, 
    and then returns a dictionary containing three hypotheses.
    """
    # Regex to capture letter and hypothesis
    pattern = r"([A-Z])\)\s*(.*?)(?=\s*[A-Z]\)|$)"
    matches = re.findall(pattern, question, re.DOTALL)

    # Convert to dictionary
    choices = {label: text.strip() for label, text in matches}
    
    return choices

def compute_prob(init_state, info, hypothesis, agent_type):
    """This function takes in all the information regarding the scenario, and then return the probabilty of the hypothesis."""
    
    # check if the person has utterances, then calculate a probability
    if info["utterance"] is not None:
        probability = compute_prob_utterance(init_state, agent_type, hypothesis, info["utterance"])
    else:
        probability = 1.0
        
    # check if the person has actions, if yes, renew the probability with a multiplication of the compute_prob_action result
    if info["action"] is not None:
        # loop through the previous actions taken by the main person, multiply the conditional probility of the choice given the action
        # note that P(H|a,u) ∝ P(H) * P(a,u|H) ∝ P(a,u|H) = P(u|H) * P(a|H)
        previous_actions = ""
        for index, action in enumerate(info["action"]):
            prob = compute_prob_action(init_state, hypothesis, agent_type, previous_actions, action)
            previous_actions += action + ", "
            print(f"Probability of step {index}: {prob}")
            probability = probability * prob
    return probability
                



def compute_prob_utterance(init_state, agent_type, hypothesis, utterance):
    """This functions takes in the previous utterances information and one choice's descriptions, and then returns a the probability that 
    the case in this choice is likely to happen."""
    
    # combining available information into the prompt, and get the response of A/B from LLM
    evaluation_prompt = f"""
    Hypothesis: {hypothesis}
    Based on the information and hypothesis, decide if it is likely for the person to say this word in these given conditions.  
    Respond with only either A or B:
    Agent type: {agent_type}
    Initial state: {init_state}
    The person's utterance: {utterance}
    A) Likely
    B) Unlikely
    """
    
    response2 = client.chat.completions.create(
        messages=[
            {"role": "system", "content": evaluation_prompt},
        ],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )
    
    # re-structure the reponse data
    response_json_str = response2.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None
    
    # loop through the list of log probabilities and get the log probabilities of outputing token A and B
    # note that response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs'] gives a dictionary of possible tokens and there corresponding log_probs 
    for top_logprob in response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs']:
        if top_logprob['token'] == 'A':
            logprob_a = top_logprob['logprob']
        elif top_logprob['token'] == 'B':
            logprob_b = top_logprob['logprob']
            
    prob_a = math.exp(logprob_a) if logprob_a is not None else 0.0 # convert the log probability into real probability
    
    # return only the probability that the case is likely (probability of choosing A)
    return prob_a


def compute_prob_action(init_state, hypothesis, agent_type, previous_actions, action):
    """This functions takes in one previous action information and one choice's descriptions, and then returns a the probability that 
    the case in this choice is likely to happen."""
    
    # combining available information into the prompt, and get the response of A/B from LLM
    evaluation_prompt = f"""
    Hypothesis: {hypothesis}
    Decide if the person's action is likely with the hypothesis provided, respond with only either A or B:
    Agent type: {agent_type}
    Initial state: {init_state}
    Previous actions: {previous_actions}
    The person's action: {action}
    A) Likely
    B) Unlikely
    """

    response2 = client.chat.completions.create(
        messages=[
            {"role": "system", "content": evaluation_prompt},
        ],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )
    
    # re-structure the reponse data
    response_json_str = response2.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None
    
    # loop through the list of log probabilities and get the log probabilities of outputing token A and B
    # note that response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs'] gives a dictionary of possible tokens and there corresponding log_probs 
    for top_logprob in response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs']:
        if top_logprob['token'] == 'A':
            logprob_a = top_logprob['logprob']
        elif top_logprob['token'] == 'B':
            logprob_b = top_logprob['logprob']
            
    prob_a = math.exp(logprob_a) if logprob_a is not None else None
    # return only the probability that the case is likely (probability of choosing A)
    return prob_a