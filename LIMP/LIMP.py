import os
import json
import text_parsing
import ast
import compute_prob_GPT
import scipy.special
import numpy as np
from openai import OpenAI
import vlm_summary
import ipdb
import random
from tqdm import tqdm
client = OpenAI(
    api_key=''
)


def get_choice(final_prob, question):
    """This functinos takes in a list of probabilities(list) and a text question(str) and then returns the choice made(str) in this question."""
    
    final_answer = f"""You will read a question with choices and likelihood of each statement for choices in a probability formal. Based on these information, answer the question and only include the letter of choice in your answer. 
    Question: {question}
    
    """
    choice_list = ["A", "B", "C", "D", "E"]
    for index, prob in enumerate(final_prob):
        final_answer += f"Probability of statement in choice {choice_list[index]} is True: {prob}\n"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": final_answer},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    model_choice = response.choices[0].message.content.strip()[0]
    return model_choice

def generate_plan(agent_type, text, hypothesis):
    """This functions takes in a scenario description text and a chosen hypothesis, then returns a high-level plan."""
    
    prompt =f"""
    You will read a text describing a scenario including a person acting in an environment and a text about the hypothesis of this person's goal and obstacle faced.
    Based on these information, give a plan about how you, as a normal helper agent, can help with the person in the scenario.
    For example, if there is a child agent in the scenario, and he cannot grab something on the shelf due to his limited height, you may output "I can go and grab that for him."
    Keep concise, avoiding output irrelevant information.
    Agent type: {agent_type}
    Text describing a scenario: {text}
    Hypothesis: {hypothesis}
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    
    plan = response.choices[0].message.content.strip()
    
    return plan 

if __name__ == "__main__":
    
    try:
        path = "/path/to/your/video.mp4" # put the path for the dataset video here
        text, agent_type = vlm_summary.get_video_description(path)
        print("Agent: ", agent_type)
        print("Scenario: ", text)
        
        # get a strcutured data including the actions and utterances of the constrained agent
        info = text_parsing.parse_text_info(text)
        print(info) # print out the information as a dictionary (eg. {"action": action_list, "utterance": utterance_list})
        
        # get a question including the LLM generated hypotheses
        question = text_parsing.hypotheses_generation(text)
        print("Question and choices: ", question)
        
        # deduce the initial states and analysing each options in the question
        init_state = text_parsing.latent_variable_extraction(info, question)
        print("Initial state: ", init_state)
        
        # calculating the probabilities
        prob_list = []
        choices = compute_prob_GPT.parse_hypothesis(question) # get a dictionary including three hypotheses eg.{'A': 'hypothesis one', 'B': 'hypothesis two', 'C': 'hypothesis three'}
        for choice, hypothesis in choices.items(): # iterate through each choice and their corresponding hypothesis
            probability = compute_prob_GPT.compute_prob(init_state, info, hypothesis, agent_type)
            prob_list.append(probability)
        final_prob = scipy.special.softmax(prob_list)
        print(final_prob)
        
        # let the LLM get a final choice based on the probabilities and what the question is asking
        model_choice = get_choice(final_prob, question)
        print("Model choose ", model_choice)
        
        # generate a high-level plan for the hypothesis chosen
        plan = generate_plan(agent_type, text, hypothesis)
        print(plan)
    
    except Exception as e:
        raise e
        print(str(e))
        print("Episode {} have error when processing".format(episode))
    
        