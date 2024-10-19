from os.path import dirname, abspath
import os

current_script_path = abspath(__file__)
config_dir = dirname(current_script_path)
prompt_dir = os.path.join(config_dir, 'prompt_template')


def load_prompt_template(name:str):
    template_path = os.path.join(prompt_dir, name + '.txt')
    template = None
    with open(template_path, 'r') as f:
        template = f.read()
    return template
