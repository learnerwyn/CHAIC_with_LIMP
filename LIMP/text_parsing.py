from openai import OpenAI
import os
import json
import re
import ipdb
import ast
client = OpenAI(
    api_key=''
)

latent_variable_prompt = """
    You will read a question about agents' mind and ideas, and the initial state of the environment from which agents' are interacting in. Agents' knowledge & belief are about this initial state, but not necessarily changed state after some actions. For each choice, extract one set of second person's belief (make sure to turn it into some statement about the environment state), second person's social goal toward first peron's actions (help, hinder or some similar words of indepedent), and second person's believed first person's physical goal (some arrangement of objects). Organize the answer in this way: A: Belief: contents; Social goal: contents; Believed Goal: contents. B: Belief: contents; Social goal: contents; Believed Goal: contents. C: Belief: contents; Social goal: contents; Believed Goal: contents. Do not include any other information or extra contents. Make sure your answer follow the format requirement, use ";" to separate variables within each choice and end response with ".". Separate contents of "A", "B" and "C" with "."

    Question: {}
"""



def parse_text_info(text): #parse any kind of action and utterance text
    """This functions takes in a question description, 
    return a dictionary including the actions and utterances of the person
    as described in the question.
    """
    
    info_extraction_prompt = """
        You will read a piece of text describing actions of a person. Summarize the person's actions and utterance separately in a chronological order. If you cannot find either utterance or actions of the person in the text, leave the corresponding section blank. When reading words like "it", replace it with inferred object or location to make actions clearer. 
        Possible actions include: walk towards somewhere, grab something from somewhere successfully, grab something from somewhere unsuccessfully, open some container, close some container, put something somewhere successfully. Only summarize these actions and their synonyms and their status(successfully/unsuccessfully) if have in this form and abandon mismatch actions. Omit peron's name. 
        Organize your answer in this form:
        Actions:
        ["action one", "action two", "action three", ...]
        ...
        Utterance:
        ["utterance one", "utterance two", "utterance three", ...]
        ... 
        
        Text: {}
        
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": info_extraction_prompt.format(text)},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    
    # get the LLM output. note that in GPT, response.choices[0].message.content gives the content of the chat completions output.
    info = response.choices[0].message.content.strip() # get the LLM output with removing any leading, and trailing whitespaces
    
    # find and reconstruct the information regarding actions and utterances
    actions_match = re.search(r'Actions:\s*(\[[^\]]*\])', info) # capturing a str format as [action1, actions2, ..., actionN]
    utterance_match = re.search(r'Utterance:\s*(\[[^\]]*\])', info) # capturing a str format as [utterance1, utterance2, ..., utteranceN]
    actions = actions_match.group(1) if actions_match else None # group(1) gives the first set of the parentheses in the matched string
    utterance = utterance_match.group(1) if utterance_match else None
    
    # convert those strings into lists
    action_list = ast.literal_eval(actions)
    utterance_list = ast.literal_eval(utterance)
    if len(action_list) == 0:
        action_list = None
    if len(utterance_list) == 0:
        utterance_list = None
        
    info = {"action": action_list, "utterance": utterance_list}
    
    return info

def hypotheses_generation(text):
    """This functions takes in a text descirbing a scenario, and then generate a string
    containing a qeustion and three hypotheses as choices."""
    
    prompt = """
        You will read a piece of text describing actions of a person. Give three hypotheses about the goal of the person and the obstacle facing by the person.
        For example,a hypothesis may looks like this: "the women wants to carry the sofa to the truck, but the sofa is too heavy"
        Organize your output in this form:
        "Given the above scenario description, which of the following statements is MOST likely to be the goal of the person?\nA) hypothesis one \nB) hypothesis two \nC) hypothesis three "
        
        Text: {}
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt.format(text)},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    question = response.choices[0].message.content.strip()
    return question
    


def init_state_extraction(info, question):
    """This functions takes in a "info" dictionary and the question itself(str), 
    and then returns the initial states(str) and a dictionary containing the belif, social goal, believed goal 
    for each option in the question."""
    
    # combining the action information in a dictionary into a string
    action_str = ""
    print(info)
    for name in info.keys():
        if info[name]["action"] is not None:
            action_str += f"{name}'s actions:\n"
            for index, action in enumerate(info[name]["action"]):
                action_str += f"{index+1}: {action}\n"
                
    init_state_prompt = """
    You will read one or two person's actions in a list like form. From the actions taken, extract the initial state of the environment before any people act. 
    Check each grab action or synonyms. Describe it in the form "There is a [object grabbed] [on/inside location of grabbing].
    Only include environment states statements. Do not include any other information or extra contents.

    Actions: {action_str}
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": init_state_prompt},
        ],
        model="gpt-4o",
        temperature=0.0
    )
    init_state = response.choices[0].message.content.strip()
    
    return init_state

if __name__ == "__main__":
    pass