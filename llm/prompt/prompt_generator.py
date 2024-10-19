from config import prompt_template_loader

def generate_prompt(prompt_template:str, input:dict):
    prompt = prompt_template
    for k in input.keys():
        prompt = prompt.replace(''.join(['{',k,'}']), input[k])
    return prompt
