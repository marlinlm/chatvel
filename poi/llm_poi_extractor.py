from typing import Callable
from llm.prompt import prompt_generator
from config import prompt_template_loader
from poi.base_poi_extractor import BasePoiExtractor
from service.service_context import ServiceContext
from utils.general_utils import num_tokens, get_time, async_run
import json


TEMPLATE_NAME_POI_EXTRACTION = 'poi_extraction_with_desc'
def load_poi_extraction_prompt_template():
    return prompt_template_loader.load_prompt_template(TEMPLATE_NAME_POI_EXTRACTION)
POI_EXTRACTION_PROMPT = load_poi_extraction_prompt_template()

def generate_poi_extraction_prompt(input:dict):
    return prompt_generator.generate_prompt(POI_EXTRACTION_PROMPT, input)

class LlmPoiExtractor(BasePoiExtractor):
    
    def __init__(self, context:ServiceContext, poi_retriever:Callable, save_poi_extraction = False, poi_extraction_save_dir:str = None):
        super().__init__(context, poi_retriever)
        self._context = context
        self.llm: object = context.llm
        self.save_poi_extraction = save_poi_extraction
        self.poi_extract_dir = poi_extraction_save_dir
    
    def extract_poi(self, text:str, title:str = None, from_image = None, from_video = None):
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
        parsed = {}
        if answer and len(answer) > 0:
            formated = answer[0].history[-1][1].replace('```','').replace('json','')
            print(answer[0].llm_output['answer'])
            print('========================================')
            print(formated)
            parsed = json.loads(formated)
        poi_retrieved = self.retrieve_poi(parsed)
                        
        return parsed, poi_retrieved
    