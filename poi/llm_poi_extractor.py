from typing import Callable
from llm.prompt import prompt_generator
from config import prompt_template_loader
from poi.base_poi_extractor import BasePoiExtractor
from utils.general_utils import num_tokens, get_time, async_run
import json


TEMPLATE_NAME_POI_EXTRACTION = 'poi_extraction'
def load_poi_extraction_prompt_template():
    return prompt_template_loader.load_prompt_template(TEMPLATE_NAME_POI_EXTRACTION)
POI_EXTRACTION_PROMPT = load_poi_extraction_prompt_template()

def generate_poi_extraction_prompt(input:dict):
    return prompt_generator.generate_prompt(POI_EXTRACTION_PROMPT, input)

class LlmPoiExtractor(BasePoiExtractor):
    
    def __init__(self, poi_retriever:Callable, llm:object, save_poi_extraction = False, poi_extraction_save_dir:str = None):
        super().__init__(poi_retriever)
        self.llm: object = llm
        self.save_poi_extraction = save_poi_extraction
        self.poi_extract_dir = poi_extraction_save_dir
    
    def extract_poi(self, id:str, text:str, title:str = None, from_image = None, from_video = None):
        text_in_image = None
        if from_image:
            text_in_image = '\n'.join(['\n'.join([t for t in segment]) for segment in from_image])
        full_text = ['Title:', str(title), 'Text:', text, 'Text in the images:', str(text_in_image), 'Transcribed text:', str(None)]
        
        prompt = generate_poi_extraction_prompt({'post_info':'\n\n'.join(full_text)})
        async def async_iter():
            results = []
            async for answer_result in self.llm.generatorAnswer(prompt=prompt,
                                                      history=[],
                                                      streaming=False):
                results.append((answer_result))
            return results
        
        answer = async_run(async_iter())
        poi_set = []
        if answer and len(answer) > 0:
            formated = '[' + answer[0].llm_output['answer'][62:-16].replace('POI Name','\"name\"').replace('POI Address Information', '\"address\"').replace('\\n\\n',',').replace('{{\\n','{').replace('\\n}}','}').replace('\\n',',').replace('\\"','"') + ']'
            print(answer[0].llm_output['answer'])
            print('========================================')
            print(formated)
            parsed = json.loads(formated)
            poi_set = set([p['name'] for p in parsed])
        poi_retrieved = self.retrieve_poi(poi_set)
                        
        return poi_set, poi_retrieved
    