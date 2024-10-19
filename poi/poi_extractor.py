from os.path import dirname
import sys
sys.path.append(dirname(dirname(__file__)))

from typing import Iterator, List, Optional, Set, Callable

from utils.logger import debug_logger
from config.model_config import POS_PATH, POS_MODEL_REPO, POS_MODEL_PATH
from splitter import ChineseTextSplitter
from transformers import AutoTokenizer, BertTokenizer
from modelscope.pipelines import pipeline                  
from modelscope.utils.constant import Tasks                
from modelscope.models import Model
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor
from modelscope import snapshot_download
import subprocess
import os
from poi.base_poi_extractor import BasePoiExtractor

if not os.path.exists(POS_MODEL_PATH):
    # snapshot_download(repo_id=LOCAL_EMBED_REPO, local_dir=LOCAL_EMBED_PATH, local_dir_use_symlinks="auto")
    debug_logger.info(f"开始下载part-of-speech模型：{POS_MODEL_REPO}")
    cache_dir = snapshot_download(model_id=POS_MODEL_REPO)
    # 如果存在的话，删除LOCAL_EMBED_PATH
    os.system(f"rm -rf {POS_PATH}")
    output = subprocess.check_output(['ln', '-s', cache_dir, POS_PATH], text=True)
    debug_logger.info(f"模型下载完毕！cache地址：{cache_dir}, 软链接地址：{POS_PATH}")

pass_list = ['好汉碑','好汉坡','毛主席纪念堂','天安门广场','天坛公园','十七孔桥','新建宫门',
             '中间小岛','故宫博物院','八达岭长城','北京天文馆','国家博物馆','美术馆','景山公园',
             '王府井小吃街','环球影城','清华大学','北京大学','前门大街','人民大会堂','中国美术馆',
             '和珅府邸','前门大街','天桥站','奥体中心','天安门印象','天安门城楼','大兴国际机场',
             '首都国际机场','自然博物馆','祈年殿','北京站','北京东站','北京西站','北京南站',
             '北京北站','人民大会堂','中央电视台','北京电视台','万达广场','央视总部大楼','中国尊',
             '国贸大厦','日坛公园','小江胡同','烟袋斜街','大柳树夜市','关鑫市场','崇文门','四季民福',
             '便宜坊','全聚德','大鸭梨烤鸭','南门涮肉','东来顺','方砖厂69号炸酱面','大海碗京菜炸酱面',
             '圆明园遗址公园','香山公园','奥林匹克公园','朝阳公园','北水古镇','中山公园','中国国家博物馆',
             '陶然亭公园','北海公园','北京植物园','海淀公园','太和门','慈宁宫花园','御花园','坤宁宫',
             '西洋楼遗址','西洋楼','国家博物院','五道营胡同','箭厂胡同','官鑫市场','清真满恒记'
             ]
block_list = ['京酱肉丝','宫保鸡丁','北京','X','北京市','内','中国','京','红','祈年']
# fallback_bolck_list = []
max_phase = 4
SOS = 'SOS'
EOS = 'EOS'
TRUNC = 'TRUNC'

def success_fast_fail_fast(phases, context):
    my_phase_str = "".join([n['span'] for n in phases])
    if my_phase_str in pass_list:
        return my_phase_str, len(phases), context
    if my_phase_str in block_list:
        context['match_block_list'] = True
        return my_phase_str, len(phases), context
    return None, 0, context

def success_slow_fail_slow(phases, context):
    my_phase_str = "".join([n['span'] for n in phases])
    if my_phase_str in pass_list:
        context['match_pass_list'] = True
        context['match_pass_length'] = len(phases)
        context['match_pass_key'] = my_phase_str
    if my_phase_str in block_list:
        context['match_block_list'] = True
        context['match_block_length'] = len(phases)
        context['match_block_key'] = my_phase_str
    return None, 0, context

def fall_back_to_first(phases, context):
    if('match_pass_list' in context and context['match_pass_list']):
        return context['match_pass_key'], context['match_pass_length'], context
    if('match_block_list' in context and context['match_block_list']):
        return None, context['match_block_length'], context

    if len(phases) > 0 and len(phases[0]['span']) > 1:
        return phases[0]['span'], 1, context
    return None, 0, context

def fall_back_to_none(phases, context):
    if('match_pass_list' in context and context['match_pass_list']):
        return context['match_pass_key'], context['match_pass_length'], context
    if('match_block_list' in context and context['match_block_list']):
        return None, context['match_block_length'], context
    return None, 0, context

rule_def_1 = {
    SOS:    {'next':['NR'], 'post_proc':fall_back_to_first},
    'NR':   {'next':[EOS,'NR','NN',['CD','M','NN']]},
    'NN':   {'next':[EOS,'NN']},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }

rule_def_2 = {
    SOS:    {'next':['CD'], 'post_proc':fall_back_to_none},
    'CD':   {'next':['M']},
    'M':    {'next':['NN']},
    'NN':   {'next':[EOS]},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }

rule_def_3 = {
    SOS:    {'next':[['NN','JJ','NN']], 'post_proc':fall_back_to_none},
    'NN':   {'next':[EOS]},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }

rule_def_4 = {
    SOS:    {'next':[['JJ','NN']], 'post_proc':fall_back_to_none},
    'NN':   {'next':[EOS,'NN']},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }

rule_def_5 = {
    SOS:    {'next':[['NN']], 'post_proc':fall_back_to_none},
    'NN':   {'next':[EOS]},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }

rule_def_6 = {
    SOS:    {'next':[['NN','NN']], 'post_proc':fall_back_to_none},
    'NN':   {'next':[EOS]},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }
rule_def_7 = {
    SOS:    {'next':[['JJ','NR']], 'post_proc':fall_back_to_none},
    'NR':   {'next':[EOS]},
    EOS:    {'next':[], 'pre_proc': success_slow_fail_slow},
    }

rules = [rule_def_1,
         rule_def_2,
         rule_def_3,
         rule_def_4,
         rule_def_5,
         rule_def_6,
         rule_def_7]

def process(rule_def, poss, idx, phases = [], pos = SOS, context = {}):
    # hit condition
    rule = None
    next_rules = None
    
    if not pos in rule_def:
        raise Exception('Rule has problems.')
    
    rule = rule_def[pos]
    
    if 'pre_proc' in rule and rule['pre_proc']:
        pre_result, pre_offset, context = rule['pre_proc'](phases, context)
        if pre_result:
            return pre_result, pre_offset, context


    next_rules = rule['next']
    if EOS in next_rules:
        eos_result, eos_offset, context = process(rule_def, poss, idx, phases, EOS, context)
        if eos_result:
            return eos_result, eos_offset, context

    offset = len(phases)
    if idx + offset < len(poss) and offset <= max_phase:

        matches_rule = False
        me = poss[idx + offset]
        my_phases = list(phases)
        for next_rule in next_rules:
            if isinstance(next_rule, list):
                temp_phase = poss[idx + offset: idx + offset + len(next_rule)]
                
                if len(next_rule) > len(temp_phase):
                    continue
                
                matched_bundle_rule = True
                for i in range(len(next_rule)):
                    if temp_phase[i]['type'] != next_rule[i]:
                        matched_bundle_rule = False
                        break
                if matched_bundle_rule:
                    matches_rule = True
                    my_phases.extend(temp_phase)
                    me = temp_phase[-1]
                    break
            else:
                if me['type'] == next_rule:
                    my_phases.append(me)
                    matches_rule = True
                    break
                
        if matches_rule:
            iter_result, iter_offset, context = process(rule_def, poss, idx, my_phases, me['type'], context)
            if iter_result:
                return iter_result, iter_offset, context
    
        if 'post_proc' in rule and rule['post_proc']:
            post_result, post_offset, context = rule['post_proc'](my_phases, context)
            if post_result:
                return post_result, post_offset, context
        
    return None, 0, context

def get_poi(poss:list):
    idx = 0
    pois = []
    while idx < len(poss):
        step = 1
        for r in rules:
            context = {}
            poi, length, _ = process(r, poss, idx, context = context)
            
            if length > 0:
                step = length
                
            if poi:
                pois.append((poi,idx))

        idx = idx + step
    return pois
    

class PosPoiExtractor(BasePoiExtractor):
    
    def __init__(self, poi_retriever: Callable, save_poi_extraction = False, poi_extraction_save_dir:str = None):
        super().__init__(poi_retriever)
        model = Model.from_pretrained(POS_PATH)
        tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
        self.pos = pipeline(task=Tasks.part_of_speech, model=model, preprocessor=tokenizer)
        self.spliter = ChineseTextSplitter(pdf=False, sentence_size=128)
        self.save_poi_extraction = save_poi_extraction
        self.poi_extract_dir = poi_extraction_save_dir
    
    def extract_poi(self, id:str, text:str, title:str = None, from_image = None, from_video = None):
        full_text = []
        if title:
            full_text.append(title)
        if text:
            full_text.append(text)
        if from_image:
            full_text.append('\n'.join([''.join([t for t in segment]) for segment in from_image]))
        full_text = '\n'.join(full_text)
        
        scents = self.spliter.split_text(full_text)
        scents_trun = []
        max_sentence_len=120
        for s in scents:
            if len(s) > max_sentence_len:
                scents_trun.extend([s[i:i+max_sentence_len] for i in range(len(s))[::max_sentence_len]])
            else:
                scents_trun.append(s)
                
        pois = []
        poss = []
        pois_flat = []
        results = self.pos(input=scents_trun)
        for result in results:
            poss.append(result['output'])
            extracted_pois = get_poi(result['output'])
            pois.append(extracted_pois)
            pois_flat.extend([p[0] for p in extracted_pois])
        
        poi_set = set(pois_flat)
        poi_retrieved = self.retrieve_poi(poi_set)
            
        if self.save_poi_extraction:
            part_of_speeches = [r['output'] for r in results]
            with open(os.path.join(self.poi_extract_dir, id + ".txt"),'w') as f:
                for idx_sent, pos_sent in enumerate(part_of_speeches):
                    idx_poi = {k:v for v,k in pois[idx_sent]}
                    for idx, pos in enumerate(pos_sent):
                        line = "|".join([str(idx_sent) + '-' + str(idx), 
                                         pos['type'], pos['span'], 
                                         (idx_poi[idx] + ':' if idx in idx_poi else '') 
                                            + (str(poi_retrieved[idx_poi[idx]]) if idx in idx_poi and idx_poi[idx] in poi_retrieved else ''), 
                                        ])
                        f.write(line)
                        f.write('\n')
            
        return poi_set, poi_retrieved