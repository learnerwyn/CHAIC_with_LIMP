from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

def get_video_description(path):
    """This functions takes in a path(str) storing the video as mp4 file, and then return a structured description for the video."""
    
    myfile = client.files.upload(file=path) # sample path may looks like /path/to/your/video.mp4
    
    prompt1 = """
    Task: You will watch a video depicting an agent performing some actions. Your goal is to infer and describe these actions in chronological order. 
    
    For the agent, provide details about his/her actions, including what objects he/she handles, where he/she obtains them from, and where he/she places them. More importantly, pay attention to whether they have done their intended actions successfully. Formulate all actions and whether the actions are succeed into a single line without including any newline characters. 
    
    Note that when the agent moves his/her arm, it likely indicates opening a container or picking up an item or placing an item. If you cannot decipher where the agent grabs an object from, make your best guess based on the context in the video. If the object cannot be effectively identified, refer to it as \"grab some object\" without attempting to guess the exact one.
    """
    
    response1 = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=[myfile, prompt1],
        config=types.GenerateContentConfig(temperature=0.0, 
                                           responseLogprobs=True, 
                                           logprobs=5)
        )
    text = response1.text
    
    prompt2 = """
    You will watch a video depicting an agent performing some actions. The agent usually has some kind of constraints.
    
    Your goal is to identify what constraints it has and only output the constrained agent type only without adding any other words.
    
    For example, if you see a child, output "Child Agent"; if you see a person on a wheelchair, output "Wheelchair Agent".
    
    There might be chances where a normal person is demonstrated, then you just output "Normal Agent".
    """
    
    response2 = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=[myfile, prompt2],
        config=types.GenerateContentConfig(temperature=0.0, 
                                           responseLogprobs=True, 
                                           logprobs=5)
        )
    agent_type = response2.text
    
    return text, agent_type

if __name__ == "__main__":
    pass