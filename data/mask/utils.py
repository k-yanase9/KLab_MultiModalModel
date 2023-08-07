import re
import random


def _merge_consecutive_elements(lst:list)->list:
    """連続するマスクをマージする

    Parameters
    ----------
    lst : list
        source list

    Returns
    -------
    list
        result
    """
    return [element for i, element in enumerate(lst) if i == 0 or element != lst[i-1]]

def _replace_mask(src_txt:str)->str:
    """<mask> を <extra_id_1>,<extra_id_2>,...に置き換える

    Parameters
    ----------
    src_txt : str
        source text

    Returns
    -------
    str
       result
    """
    pattern = re.compile(r"<mask>")
    counter = 1

    def replace(match):
        nonlocal counter
        replacement = "<extra_id_{}>".format(counter)
        counter += 1
        return replacement

    result = re.sub(pattern, replace, src_txt)
    return result

def make_mask_textpair(src_str:str,mask_rasio:float=0.15)->list[str,str]:
    """マスクされたソースとターゲットのペアを作成する

    Parameters
    ----------
    src_str : str
        src_text
    mask_rasio : float, optional
        mask rasio, by default 0.15

    Returns
    -------
    list[str,str]
        [src_text,tgt_text]
    
    Note
    ---------
    最低でも一つはマスクするため、厳密にはmask_rasioにはならない
    """
    src_str_list = src_str.split(" ")
    index_list = list(range(len(src_str_list)))
    #0.6を加算することで最低でも一つはマスクする
    choice_num = round(len(index_list)*mask_rasio + 0.6)
    choice_num = round(len(index_list)*mask_rasio + 0.6)
    mask_list = random.sample(index_list,choice_num)
    mask_list.sort()
    #ソースのリストとターゲットのリストを作成
    src_list = _merge_consecutive_elements([src_str_list[index] if not index in mask_list else "<mask>" for index in index_list])
    tgt_list = _merge_consecutive_elements(['<mask>' if not index in mask_list else src_str_list[index] for index in index_list])
    src_txt = _replace_mask(" ".join(src_list))
    tgt_txt = _replace_mask(" ".join(tgt_list))
    return src_txt,tgt_txt

if __name__ == "__main__":
    print(make_mask_textpair("Woman is Smile."))