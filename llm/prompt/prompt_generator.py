def generate_prompt(prompt_template:str, input:dict = None):
    prompt = prompt_template
    if not input is None and len(input) > 0:
        for k in input.keys():
            prompt = prompt.replace(''.join(['{',k,'}']), input[k])
    return prompt
