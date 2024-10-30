import json
import re
from typing import List
import datetime

from config import prompt_template_loader
from llm.prompt import prompt_generator
from utils.general_utils import async_run

TEMPLATE_NAME_QUERY_DECO = 'query_decompose'
def load_query_decompose_prompt_template():
    return prompt_template_loader.load_prompt_template(TEMPLATE_NAME_QUERY_DECO)
QUERY_DECOMPOSE_PROMPT = load_query_decompose_prompt_template()

TEMPLATE_NAME_REQ_EXTR = 'requirement_extraction'
def load_requirement_extraction_prompt_template():
    return prompt_template_loader.load_prompt_template(TEMPLATE_NAME_REQ_EXTR)
REQUIREMENT_EXTRACTION_PROMPT = load_requirement_extraction_prompt_template()

    
async def run_llm(llm, prompt:str, history:List[List[str]] = None):
    results = []
    async for answer_result in llm.generatorAnswer(prompt=prompt,
                                            history=history,
                                            streaming=False):
        results.append((answer_result))
    return results

def generate_requirement_extraction_prompt(slot, requirement, query):
    requirement_type,requirement_body = slot
    param = {
        'query':query,
        'today_date': str(datetime.date.today()),
        'requirement': requirement,
        'requirement_type': requirement_type,
        'requirement_desc': requirement_body['desc'],
        'requirement_output_format': requirement_body['format'] if 'format' in requirement_body else 'null'
    }
    return prompt_generator.generate_prompt(REQUIREMENT_EXTRACTION_PROMPT, param)

def llm_extraction(llm, slot, requirement, history):
    prompt = generate_requirement_extraction_prompt(slot, requirement, history[-1][0])
    answer = async_run(run_llm(llm= llm, prompt = prompt, history=history))
    parsed = json.loads(answer[0].history[-1][1])
    if 'explicitly metioned in the query or chat history' in parsed and parsed['explicitly metioned in the query or chat history']:
        if 'extracted value' in parsed:
            return parsed['extracted value']
    return None
    

travel_type = {
    "desc":"The type of the travel. Examples: \"Solo\", \"Family\", \"Couple tour\", \"Parent with kids\", \"Group tour\",\"Tour with best friends\" and etc.", 
    'format': "a short phase in Chinese describing the type of the travel. ",
    "required":True,
    'extraction':llm_extraction
}
traveller_type = {
    "desc":"Indicates the traveller. Could be one of \"Solo\", \"Family\", \"Couple tour\", \"Parent with kids\", \"Group tour\",\"Tour with best friends\" and etc.", 
    "required":True,
}
days = {
    "desc":"The number of days that the whole travel will last. ",
    'format': "a number representing the number of days. For example: \"2\",\"3\",\"4\",\"5\",\"6\", \"7\" and etc.",
    'validation':str.isnumeric,
    'extraction':llm_extraction
}
travel_date = {
    "desc":"The date that the travel take place. Note that the provided **Today date** can not be used as the **travel date** directly unless there is explicitly metioned that the travel starts today.",
    'format':"a date sting formatted as YYYYMMDD. For example: \"20231203\" representing Dec 03, 2023.",
    "required":True,
    'extraction':llm_extraction,
}
event = {
    "desc":"Is there a holiday or event during the travel? Like \"National day\", \"Laybor day\",\"Childrens's day\" and ect.",
}
itinerary = {
    "desc":"The general requirements regarding the travel.",
}

region = {
    "desc":"A specific administrative region (like a city or a district) where the travel will take place. Note that the region requirement should not be a specific place like attraction or restaurant.",
    'format': "the name of an administrative region. Should only populate the name of the region, instead of a josn object.",
    'required':True,
    'extraction':llm_extraction,
}
attraction = {
    "desc": "A specific place of attraction that the travel should include.",
    'format': "the name of the attraction in Chinese. ",
    "required":True,
    'extraction':llm_extraction
}
eat_or_drink = {
    "desc": "A specific place of food or drink that the travel should include.",
}
accommodation = {
    "desc": "A specific place of accommodation that the travel should include.",
}
shopping = {
    "desc": "A specific place for shopping that the travel should include.",
}
entertainment = {
    "desc": "A specific place of entertaining that the travel should include.",
}

requirement_types = {
    "travel type": travel_type,
    # "traveller": traveller_type,
    "days": days,
    "travel date": travel_date,
    "event": event, 
    "itinerary": itinerary, 
    "region": region, 
    "attraction": attraction, 
    "eat or drink": eat_or_drink, 
    "accommodation": accommodation, 
    "shopping": shopping, 
    "entertainment": entertainment}

check_slot = {
    k:requirement_types[k] for k in requirement_types.keys() if 'required' in requirement_types[k]
}

def generate_query_decompose_prompt(requirement_types = requirement_types, query:str = 'Nothing'):
    tags = {}
    for k,v in requirement_types.items():
        tags[k] = v['desc']  +  f" If the **{k} requirement** is not metioned, do not populate this requirement in your return."
            # f" \ Do not infer any **{k} requirement**\
            #  that is not explicitly mentioned in the chat history.\
               
    param = {
        'query' : query,
        '_requirement_types' : ','.join(tags.keys()),
        '_requirement_desc'  : '\n'.join( [' **' + k +'** : ' + v for k,v in tags.items()])
    }
    return prompt_generator.generate_prompt(QUERY_DECOMPOSE_PROMPT, param)

        

class QueryDecomposer:
    def __init__(self, llm:object):
        self._llm:object = llm
        

    def check_slot(self, slots) -> set:
        ret_slot = {}
        for k,v in check_slot.items():
            checked = False
            for slot in slots:
                if slot['pos_or_neg'] == 'neg':
                    continue
                if slot['type'] == k:
                    checked = True
                    break
            if not checked:
                ret_slot[k] = v
        return ret_slot
    
    def decompose_query(self, query:str, history:List[List[str]] = []) -> dict:

        history.append([query])
        slots = dict(requirement_types)
        slots_len = 0
        result = []

        prompt = generate_query_decompose_prompt(slots, query)
        my_history = list(history)
        answer = async_run(run_llm(llm = self._llm, prompt = prompt, history=my_history))
        if answer and len(answer) > 0:
            raw_answer = answer[0].history[-1][1]
            try:
                llm_ret = json.loads(raw_answer)
                for requirement in llm_ret:
                    my_history = list(history)
                    requirement_processed = self.process_requirement(requirement = requirement, history = my_history)
                    if requirement_processed:
                        result.append(requirement_processed )
            except json.JSONDecodeError as e:
                pass
            
            slots = self.check_slot(result)
            for k, _ in slots.items():
                my_history = list(history)
                requirement_processed = self.process_requirement({'requirement': '', 'pos_or_neg': 'pos', 'type':k}, my_history)
                if requirement_processed:
                        result.append(requirement_processed )
            slots = self.check_slot(result)
        return result, slots

    def process_requirement(self, requirement, history):
        ret_requirement = dict(requirement)
        key = ret_requirement['type']
        if key in requirement_types:
            slot = requirement_types[key]
            extracted = ret_requirement['requirement']
            if 'extraction' in slot:
                extracted = slot['extraction'](requirement = extracted, slot=(key, slot), history = history, llm = self._llm)
                if extracted is None or extracted == '':
                    return None
            if 'validation' in slot and (not slot['validation'](extracted)):
                return None
            ret_requirement['requirement'] = extracted
        return ret_requirement

                
            
