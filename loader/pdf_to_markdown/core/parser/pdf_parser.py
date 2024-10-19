# -*- coding: utf-8 -*-
import os
import random

import fitz
import xgboost as xgb
from io import BytesIO
import torch
import re
import pdfplumber
import logging
from PIL import Image
import numpy as np

from PyPDF2 import PdfReader as pdf2_read

from loader.pdf_to_markdown.core.vision import Recognizer, LayoutRecognizer, TableStructureRecognizer_LORE
from loader.pdf_to_markdown.core.nlp import huqie
# from qanything_kernel.dependent_server.ocr_server.ocr import OCRQAnything
from config.model_config import OCR_MODEL_PATH, PDF_MODEL_PATH
from utils.logger import debug_logger
from tqdm import tqdm
from copy import deepcopy

logging.getLogger("pdfminer").setLevel(logging.WARNING)


class HuParser:
    def __init__(self, device='cpu'):
        # self.ocr = OCRQAnything(model_dir=OCR_MODEL_PATH, device=device)  # 省显存
        if hasattr(self, "model_speciess"):
            self.layouter = LayoutRecognizer("layout." + self.model_speciess)
        else:
            self.layouter = LayoutRecognizer("layout")
        self.tbl_det = TableStructureRecognizer_LORE()

        self.updown_cnt_mdl = xgb.Booster()
        if torch.cuda.is_available():
            self.updown_cnt_mdl.set_param({"device": "cuda"})

        model_dir = os.path.join(
                PDF_MODEL_PATH,
                "checkpoints/updown")
        self.updown_cnt_mdl.load_model(os.path.join(
            model_dir, "updown_concat_xgb.model"))
        self.page_from = 0


    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // len(c["text"])

    def __height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]),
                   abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(
            self, a, b):
        return (
            b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = huqie.qie(down["text"][:LEN]).split(" ")
        tks_up = huqie.qie(up["text"][-LEN:]).split(" ")
        tks_all = up["text"][-LEN:].strip() \
            + (" " if re.match(r"[a-zA-Z0-9]+",
                               up["text"][-1] + down["text"][0]) else "") \
            + down["text"][:LEN].strip()
        tks_all = huqie.qie(tks_all).split(" ")
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(
                r"([。？！；!?;+)）]|[a-z]\.)$",
                up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            True if re.search(
                r"(^.?[/,?;:\]，。；：’”？！》】）-])",
                down["text"]) else False,
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[\(（][^\)）]+$", up["text"])
            and re.search(r"[\)）]", down["text"]) else False,
            self._match_proj(down),
            True if re.match(r"[A-Z]", down["text"]) else False,
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()
                                                                        ) > 1 and len(
                down["text"].strip()) > 1 else False,
            up["x0"] > down["x1"],
            abs(self.__height(up) - self.__height(down)) / min(self.__height(up),
                                                               self.__height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"])) /
            max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1],
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and huqie.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and huqie.tag(tks_up[0]).find("n") >= 0
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threashold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threashold \
                        and arr[j + 1]["top"] < arr[j]["top"] \
                        and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and \
                    o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    def _table_transformer_job(self, ZM):
        logging.info("Table processing...")
        imgs, pos = [], []
        tbcnt = [0]
        MARGIN = 10
        self.tb_cpns = []
        assert len(self.page_layout) == len(self.page_images)
        for p, tbls in enumerate(self.page_layout):  # for page
            tbls = [f for f in tbls if f["type"] == "table"]
            tbcnt.append(len(tbls))
            if not tbls:
                continue
            for tb in tbls:  # for table
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, \
                    tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                pos.append((left, top))
                imgs.append(self.page_images[p].crop((left, top, right, bott)))

        assert len(self.page_images) == len(tbcnt) - 1
        if not imgs:
            return
        recos = self.tbl_det(imgs)
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # for page
            pg = []
            for j, tb_items in enumerate(
                    recos[tbcnt[i]: tbcnt[i + 1]]):  # for table
                poss = pos[tbcnt[i]: tbcnt[i + 1]]
                for it in tb_items:  # for table components
                    it["x0"] = (it["x0"] + poss[j][0])
                    it["x1"] = (it["x1"] + poss[j][0])
                    it["top"] = (it["top"] + poss[j][1])
                    it["bottom"] = (it["bottom"] + poss[j][1])
                    for n in ["x0", "x1", "top", "bottom"]:
                        it[n] /= ZM
                    it["top"] += self.page_cum_height[i]
                    it["bottom"] += self.page_cum_height[i]
                    it["pn"] = i
                    it["layoutno"] = j
                    pg.append(it)
            self.tb_cpns.extend(pg)

        def gather(kwd, fzy=10, ption=0.6):
            eles = Recognizer.sort_Y_firstly(
                [r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)

        # add R,H,C,SP tag to boxes within table layout
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in self.tb_cpns if re.match(
            r"table column$", r["label"])], key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
        for b in self.boxes:
            if b.get("layout_type", "") != "table":
                continue
            ii = Recognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            ii = Recognizer.find_overlapped_with_threashold(
                b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            ii = Recognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

    def __ocr_pdf(self,pagenum,bxs_pymupdf,ZM=3):
        """
        use pymupdf parse pdf to save time
        """
        bxs = bxs_pymupdf
        if not bxs:
            self.boxes.append([])
            return 
        bxs = [(np.array(item[0]),'',item[1]) for item in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [{"x0": b[0][0] / ZM, "x1": b[1][0] / ZM,
            "top": b[0][1] / ZM, "text": rec_text, "txt": t,
            "bottom": b[-1][1] / ZM,
            "page_number": pagenum} for b, t, rec_text in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
            self.mean_height[-1] / 3
        )        
        for b in bxs:
            del b["txt"]
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[-1] == 0:
            self.mean_height[-1] = np.median([b["bottom"] - b["top"]
                                            for b in bxs])
        self.boxes.append(bxs)  


                
    def _layouts_rec(self, ZM, drop=True):
        assert len(self.page_images) == len(self.boxes)
        self.boxes, self.page_layout = self.layouter(
            self.page_images, self.boxes, ZM, drop=drop)
        # cumlative Y
        for i in range(len(self.boxes)):
            self.boxes[i]["top"] += \
                self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += \
                self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _text_merge(self):
        # merge adjusted boxes
        bxs = self.boxes

        def end_with(b, txt):
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        # horizontally merge adjacent box with the same layout
        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]
            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure",
                                                                                                 "equation"]:
                i += 1
                continue
            if abs(self._y_dis(b, b_)
                   ) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
                # merge
                if b_["x0"] > b["x0"]:
                    bxs[i]["x1"] = b_["x1"]
                    bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                    bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                    bxs[i]["text"] += b_["text"]
                    bxs.pop(i + 1)
                else:
                    bxs[i + 1] = b
                    bxs[i] = b_
                    b = bxs[i]
                    b_ = bxs[i + 1]
                    bxs[i]["x1"] = b_["x1"]
                    bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                    bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                    bxs[i]["text"] += b_["text"]
                    bxs.pop(i + 1)
                continue
            i += 1
            continue

            dis_thr = 1
            dis = b["x1"] - b_["x0"]
            if b.get("layout_type", "") != "text" or b_.get(
                    "layout_type", "") != "text":
                if end_with(b, "，") or start_with(b_, "（，"):
                    dis_thr = -8
                else:
                    i += 1
                    continue

            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 5 \
                    and dis >= dis_thr and b["x1"] < b_["x1"]:
                # merge
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
        self.boxes = bxs

    def _naive_vertical_merge(self):
            bxs = Recognizer.sort_Y_firstly(
                self.boxes, np.median(
                    self.mean_height) / 3)
            i = 0
            while i + 1 < len(bxs):
                b = bxs[i]
                b_ = bxs[i + 1]
                # if b["page_number"] < b_["page_number"] and re.match(
                #         r"[0-9  •一—-]+$", b["text"]):
                #     bxs.pop(i)
                #     continue
                if b["page_number"] < b_["page_number"]:
                    # bxs.pop(i)
                    i += 1
                    continue
                if not b["text"].strip():
                    bxs.pop(i)
                    continue
                concatting_feats = [
                    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                    len(b["text"].strip()) > 1 and b["text"].strip(
                    )[-2] in ",;:'\"，‘“、；：",
                    b["text"].strip()[0] in "。；？！?”）),，、：",
                ]
                # features for not concating
                feats = [
                    b.get("layoutno", 0) != b_.get("layoutno", 0),
                    b["text"].strip()[-1] in "。？！?",
                    self.is_english and b["text"].strip()[-1] in ".!?",
                    b["page_number"] == b_["page_number"] and b_["top"] -
                    b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
                    b["page_number"] < b_["page_number"] and abs(
                        b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
                ]
                # split features
                detach_feats = [b["x1"] < b_["x0"],
                                b["x0"] > b_["x1"]]
                if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                    # print(
                    #     b["text"],
                    #     b_["text"],
                    #     any(feats),
                    #     any(concatting_feats),
                    #     any(detach_feats))
                    i += 1
                    continue
                # merge up and down
                b["bottom"] = b_["bottom"]
                b["text"] += b_["text"]
                b["x0"] = min(b["x0"], b_["x0"])
                b["x1"] = max(b["x1"], b_["x1"])
                bxs.pop(i + 1)
            self.boxes = bxs

    def _concat_downward(self, concat_between_pages=True):
        blocks = {}
        for sec in self.boxes:
            pn = sec['page_number']
            if not sec.get('layoutno'): continue
            sec_no = str(pn) + '-' + sec['layoutno']
            if sec_no not in blocks.keys():
                blocks[sec_no] = [sec]
            else:
                blocks[sec_no].append(sec)
        boxes = []
        # for b in blocks:
        for k, b in blocks.items():
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(
                        r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]):
                    t["text"] += " "
                if t["text"] and t['layout_type'] == 'content':
                    t["text"] += "\n"
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] \
                        and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)
        # print(self.boxes, file=open('boxes3.txt', 'w'))

    def _filter_forpages(self):
        if not self.boxes:
            return
        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(r"(致谢|acknowledge)$",
                            re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            findit = True
            eng = re.match(
                r"[0-9a-zA-Z :'.-]{5,}",
                self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(
                self.boxes[i]["text"].strip().split(" ")[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(
                    self.boxes[i]["text"].strip().split(" ")[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if b["text"].strip()[0] != b_["text"].strip()[0] \
                    or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm") \
                    or huqie.is_chinese(b["text"].strip()[0]) \
                    or b["top"] > b_["bottom"]:
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)


    def get_markdown_header(self,markdown_str):
        """
        extract header from markdown table
        """
        markdown_str = markdown_str.lstrip('\n')
        header = '\n'.join(markdown_str.split('\n')[:2])
        header = '\n\n' + header
        return header

    def merge_header_markdown(self,header,table):
        table = table.lstrip('\n').rstrip('\n')
        table = '\n'.join([table.split('\n')[0]] + table.split('\n')[2:])
        res = header + '\n' + table + '\n\n'
        return res

    def _extract_table_figure(self, need_image, ZM,
                              return_html, need_position,image_dir):
        tables = {}
        figures = {}
        captions = {}
        merged_captions = {}
        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + \
                "-" + str(self.boxes[i]["layoutno"])
            if TableStructureRecognizer_LORE.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] == 'caption':
                # print(self.boxes[i]['text'])
                nomerge_lout_no.append(lst_lout_no)
                if lout_no not in captions:
                    captions[lout_no] = []
                captions[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                continue
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1
        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()],
                      key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))
        i = len(tbls) - 1
        table_merge_header = {}
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            # tables[k0].extend(tables[k])
            table_merge_header[k0] = True
            # del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])
        for k, v in captions.items():
            txt = ''
            x0 = min([b['x0'] for b in captions[k]])
            x1 = max([b['x1'] for b in captions[k]])
            top = min([b['top'] for b in captions[k]])
            bottom = max([b['bottom'] for b in captions[k]])
            v = sorted(v, key=lambda x: x['top'])
            i = 0
            while i < len(v):
                txt += v[i]['text']
                i += 1
            merged_captions[k] = {'x0': x0, 'x1': x1, 'top': top, 'text': txt, 'bottom': bottom, 
                                  'page_number': int(k.split('-')[0]), 'layout_type': 'caption', 
                                  'layoutno': '-'.join(k.split('-')[1:])}
        
        # find captions and pop out
        for ck, cv in captions.items():
            caption = captions[ck]
            # mh = self.mean_height[c["page_number"]-1]
            # find the nearest layouts
            def nearest(tbls):
                minks, minvs = [], []
                nonlocal caption
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    if k.find('equation') >=0:
                        continue 
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        for c in caption:
                            if b['page_number'] != c['page_number']:
                                continue
                            y_dis = self._y_dis(c, b)
                            x_dis = self._x_dis(
                                c, b) if not x_overlapped(
                                c, b) else 0
                            dis = y_dis * y_dis + x_dis * x_dis
                            if dis < minv:
                                minv = dis
                                mink = k
                            minvs.append(minv)
                            minks.append(mink)
                if minvs:
                    mink = max(minks, key=minks.count)
                    minv = np.mean([minvs[i] for i in range(len(minvs)) if minks[i] == mink])
                    mink2 = minks[minvs.index(min(minvs))]
                    minv2 = np.mean([minvs[i] for i in range(len(minvs)) if minks[i] == mink2])
                    if mink != mink2 and minv > minv2:
                        mink = mink2
                        minv = minv2
                return mink, minv
            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, merged_captions[ck])
                logging.debug(
                    "TABLE:" +
                    self.boxes[i]["text"] +
                    "; Cap: " +
                    tk)
            elif fk:
                figures[fk].insert(0, merged_captions[ck])
                logging.debug(
                    "FIGURE:" +
                    self.boxes[i]["text"] +
                    "; Cap: " +
                    tk)
            # self.boxes.pop(i)

        res = []
        positions = []

        def cropout(bxs, ltype, poss):
            nonlocal ZM
            pn = set([b["page_number"] - 1 for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {
                    "x0": np.min([b["x0"] for b in bxs if b['layout_type'] != 'caption']),
                    "top": np.min([b["top"] for b in bxs if b['layout_type'] != 'caption']) - ht,
                    "x1": np.max([b["x1"] for b in bxs if b['layout_type'] != 'caption']),
                    "bottom": np.max([b["bottom"] for b in bxs if b['layout_type'] != 'caption']) - ht
                }
                louts = [l for l in self.page_layout[pn] if l["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    logging.warn(
                        f"Missing layout match: {pn + 1},%s" %
                        (bxs[0].get(
                            "layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                poss.append((pn + self.page_from, left, right, top, bott))
                return self.page_images[pn] \
                    .crop((left * ZM, top * ZM,
                           right * ZM, bott * ZM))
            pn = {}
            for b in bxs:
                p = b["page_number"] - 1
                if p not in pn:
                    pn[p] = []
                pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            pic = Image.new("RGB",
                            (int(np.max([i.size[0] for i in imgs])),
                             int(np.sum([m.size[1] for m in imgs]))),
                            (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        # crop figure out and add caption
        for k, bxs in figures.items():
            caption = ''.join([b["text"] for b in bxs if b['layout_type'] == 'caption'])
            # txt = "\n".join([b["text"] for b in bxs])
            # if not txt:
            #     continue

            poss = []
            img = cropout(
                    bxs,
                    "figure", poss)
            img.save('{}/{}.jpg'.format(image_dir,k))
            res.append(
                (
                #  img,
                 caption,
                 k))
            positions.append(poss)
        merge_header = False
        table_header = ''            
        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean(
                [(b["bottom"] - b["top"]) / 2 for b in bxs]))
            poss = []
            img = cropout(bxs, "table", poss)
            pn = list(set([b["page_number"] - 1 for b in bxs]))[0]
            try:
                if merge_header:
                    res_dict = self.tbl_det.construct_table(bxs, img, poss[0][1:], self.page_cum_height[pn], html=return_html, is_english=self.is_english)
                    res_dict['table_markdown'] = self.merge_header_markdown(table_header,res_dict['table_markdown'])
                    res.append((res_dict,k))
                else:
                    res_dict = self.tbl_det.construct_table(bxs, img, poss[0][1:], self.page_cum_height[pn], html=return_html, is_english=self.is_english)           
                    res.append((res_dict,k))
                if k in table_merge_header.keys():   #下一个表格需要添加当前表头
                    merge_header = True
                    table_header = self.get_markdown_header(res_dict['table_markdown'])
                else:
                    merge_header = False
                    table_header = ''
            except Exception as e:
                print(e.args)
                res.append((
                            # img, self.tbl_det.construct_table(bxs, img, html=return_html, is_english=self.is_english)))
                            self.tbl_det.construct_table(bxs, img, poss[0][1:], self.page_cum_height[pn], html=return_html, is_english=self.is_english), 
                            k))
            # img.save('{}.jpg'.format(k))
            positions.append(poss)

        assert len(positions) == len(res)

        if need_position:
            return list(zip(res, positions))
        return res
    


    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12)
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##" \
            .format("-".join([str(p) for p in pn]),
                    bx["x0"], bx["x1"], top, bott)

    def __filterout_scraps(self, boxes, ZM):

        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(
                    b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = self.proj_match(
                boxes[0]["text"]) or boxes[0].get(
                "layout_type",
                "") == "title"

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                width_mean = np.mean(widths)
                mmj = self.proj_match(
                    line["text"]) or line.get(
                    "layout_type",
                    "") == "title"
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if not mmj and self._y_dis(
                            line, boxes[i]) >= 3 * mh and height(line) < 1.5 * mh:
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or \
                            (self._x_dis(boxes[i], line) < pw / 10): \
                            # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception as e:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append(
                    "\n".join([c["text"] + self._line_tag(c, ZM) for c in lines]))
            else:
                logging.debug("REMOVED: " +
                              "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            pdf = pdfplumber.open(
                fnm) if not binary else pdfplumber.open(BytesIO(binary))
            return len(pdf.pages)
        except Exception as e:
            pdf = fitz.open(fnm) if not binary else fitz.open(
                stream=fnm, filetype="pdf")
            return len(pdf)
    
    def page_ocr(self,page,zoomin):
        blocks = page.get_text(
            "dict", flags=0,
        )["blocks"]
        ocr_res = []
        for b in blocks:
            for line in b["lines"]:
                line_text = ''
                line_bbox_lst = [[],[],[],[]]
                for item in line['spans']:
                    line_text += item['text']
                    line_bbox_lst[0].append(item['bbox'][0]*zoomin)
                    line_bbox_lst[1].append(item['bbox'][1]*zoomin)
                    line_bbox_lst[2].append(item['bbox'][2]*zoomin)
                    line_bbox_lst[3].append(item['bbox'][3]*zoomin)
                line_bbox = [min(line_bbox_lst[0]),min(line_bbox_lst[1]),max(line_bbox_lst[2]),max(line_bbox_lst[3])]
                four_point_bbox = [[line_bbox[0],line_bbox[1]],[line_bbox[2],line_bbox[1]],
                                    [line_bbox[2],line_bbox[3]],[line_bbox[0],line_bbox[3]]] 
                ocr_res.append([four_point_bbox,line_text,1])
        return ocr_res


    def __images__(self, fnm, zoomin=3, page_from=0,
                   page_to=299, callback=None):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        self.pdf = fitz.open(fnm) if isinstance(
            fnm, str) else fitz.open(
            stream=fnm, filetype="pdf")
        self.page_images = []
        self.page_chars = []
        self.ocr_res = []
        mat = fitz.Matrix(zoomin, zoomin)
        self.total_page = len(self.pdf)
        for i, page in enumerate(self.pdf):
            if i < page_from:
                continue
            if i >= page_to:
                break
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height],
                                    pix.samples)
            self.page_images.append(img)
            self.page_chars.append([])
            page_ocr_res = self.page_ocr(page,zoomin)
            self.ocr_res.append(page_ocr_res)


        self.outlines = []
        try:
            self.pdf = pdf2_read(fnm if isinstance(fnm, str) else BytesIO(fnm))
            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)
            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        logging.info("Images converted.")
        self.is_english = [re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(
            random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i]))))) for i in
            range(len(self.page_chars))]
        if sum([1 if e else 0 for e in self.is_english]) > len(
                self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False

        debug_logger.info("Start OCR！")
        for i, img in tqdm(enumerate(self.page_images)):
            chars = self.page_chars[i] if not self.is_english else []
            self.mean_height.append(
                np.median(sorted([c["height"] for c in chars])) if chars else 0
            )
            self.mean_width.append(
                np.median(sorted([c["width"] for c in chars])) if chars else 8
            )
            self.page_cum_height.append(img.size[1] / zoomin)
            j = 0
            while j + 1 < len(chars):
                if chars[j]["text"] and chars[j + 1]["text"] \
                        and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"]) \
                        and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"],
                                                                       chars[j]["width"]) / 2:
                    chars[j]["text"] += " "
                j += 1
            # self.__ocr(i + 1, img, chars, zoomin)
            self.__ocr_pdf(i+1, self.ocr_res[i], zoomin)
            if callback:
                callback(prog=(i + 1) * 0.6 / len(self.page_images), msg="")

        if not self.is_english and not any(
                [c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                                        "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        logging.info("Is it English:", self.is_english)

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(
            need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    def remove_tag(self, txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    def crop(self, text, ZM=3, need_position=False):
        imgs = []
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", text):
            pn, left, right, top, bottom = tag.strip(
                "#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(
                right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")],
                        left, right, top, bottom))
        if not poss:
            if need_position:
                return None, None
            return

        max_width = max(
            np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(
            0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + GAP),
                     min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + 120)))

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                bottom += self.page_images[pn - 1].size[1]
            imgs.append(
                self.page_images[pns[0]].crop((left * ZM, top * ZM,
                                               right *
                                               ZM, min(
                                                   bottom, self.page_images[pns[0]].size[1])
                                               ))
            )
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(
                    bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                imgs.append(
                    self.page_images[pn].crop((left * ZM, 0,
                                               right * ZM,
                                               min(bottom,
                                                   self.page_images[pn].size[1])
                                               ))
                )
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(
                        bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB",
                        (width, height),
                        (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert('RGBA')
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append((pn, bx["x0"], bx["x1"], top, min(
            bott, self.page_images[pn - 1].size[1] / ZM)))
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append((pn, bx["x0"], bx["x1"], top, min(
                bott, self.page_images[pn - 1].size[1] / ZM)))
        return poss
    
            



class PlainParser(object):
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(
                filename if isinstance(
                    filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        return [(l, "") for l in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


if __name__ == "__main__":
    pass