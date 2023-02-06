
# may need python3.6 to install all the package
import os
import re
from harvesttext import HarvestText
import pyhanlp
from loguru import logger
import pandas as pd
import tqdm
from datetime import datetime

# from multiprocesspandas import applyparallel
# 该模块并不是一个很好的多线程的实现
# import vaex


# def clean(text):
#     text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)","", text)  # 去除正文中的@和回复/转发中的用户名
#     text = re.sub(r"#\S+#","", text)      # 保留话题内容
#     URL_REGEX = re.compile(
#         r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
#         re.IGNORECASE)
#     text = re.sub(URL_REGEX, "", text)       # 去除网址
#     text = text.replace('http', '')
#     text=text.replace('分享图片', '')
#     text = text.replace('分享视频', '')
#     text = text.replace("转发微博","")       # 去除无意义的词语
#     text = re.sub(r"\s+", "", text) # 合并正文中过多的空格
#     text = text.replace('\u200b', '')#去除不可见字符\u200d \U0001fac0 \U0001f964 http  \U0001fa74 \U0001f99a
#     return text.strip()#保证字符串尾部没有多余空格
# #去除链接

def clean_s(x):
    x = x.str.replace(r"(回复)?(//)?\s*@\S*?\s*(:| |$)","", regex=True)  # 去除正文中的@和回复/转发中的用户名
    x = x.str.replace(r"#\S+#","",  regex=True)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = x.str.replace(URL_REGEX, "",  regex=True)       # 去除网址
    text = text.str.replace('http', '')
    text=text.str.replace('分享图片', '')
    text = text.str.replace('分享视频', '')
    text = text.str.replace("转发微博","")       # 去除无意义的词语
    text = text.str.replace(r"\s+", "", regex=True) # 合并正文中过多的空格
    text = text.str.replace('\u200b', '')#去除不可见字符\u200d \U0001fac0 \U0001f964 http  \U0001fa74 \U0001f99a
    return text.str.strip()#保证字符串尾部没有多余空格
#

def deletehttp(sentence):
    sentence = sentence.split(r"http://t.cn/")[0]
    return sentence

def deletehttp_s(sentence):
    sentence = sentence.str.split(r'http://t.cn/',expand=True)[0]
    return sentence


#移除所有不可见字符，除\n外
def remove_invisible_chars(s):
    str = ''.join([x for x in s if (x.isprintable()) or (x is '\n')])
    
    # str = ''
    # for x in s:
    #     if x is not '\n' and not x.isprintable():
    #         str += ''
    #     else:
    #         str += x
    return str



def remove_url(src):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', src, flags=re.MULTILINE)
    return vTEXT
    
def remove_url_s(src):
    pat = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', flags=re.MULTILINE)
    vTEXT = src.str.replace(pat,'',regex=True)
    return vTEXT

#去除停用词
# 创建停用词
# import jieba
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     return stopwords
# # 对句子进行分词
# def seg_sentence(sentence):
#     f_sw = 'D:/wdy/stopwords-master/stopwords-master/hit_stopwords.txt'
#     sentence_seged = jieba.cut(sentence.strip())
#     stopwords = stopwordslist(f_sw)
#     outstr = ''
#     for word in sentence_seged:
#         if word not in stopwords:
#             if word != '\t':
#                 outstr += word
#                 outstr += " "
#     return outstr

# # 若为字母，使用空格代替
# def deletewords(info):
#     for i in info:
#         if i.isalpha():
#             info = info.replace(i, " ")
#     return info

# def clean_a_test(row):

#     text = row['text']
#     ht = HarvestText()
#     CharTable = pyhanlp.JClass('com.hankcs.hanlp.dictionary.other.CharTable')
#    # logger.info(f'old:{len(data)}\n')
#     #print("old:",len(data))
#     content = CharTable.convert(text)
#     cleaned_content = remove_url(ht.clean_text(content, emoji=False))  # 过滤@后最多6个字符
#     cleaned_content=clean(cleaned_content)
#     cleaned_content=remove_invisible_chars(cleaned_content)
#     cleaned_content = deletehttp(cleaned_content)
#     # cleaned_content = deletewords(cleaned_content)

#     return cleaned_content

if __name__ == "__main__":
    # print("test")
    log_path = "./log.txt"
    logger.add(log_path, rotation='1 week', retention='30 days', enqueue=True)
    
    input_path = "weibo_output/parquet"
    output_path = "weibo_output/weibo_text_clean_parquet"
    
    need_process = [item for item in os.listdir(input_path) if item not in list(os.listdir(output_path))]
    print(len(os.listdir(output_path)),len(os.listdir(input_path)))
    
    ht = HarvestText()
    CharTable = pyhanlp.JClass('com.hankcs.hanlp.dictionary.other.CharTable')
         
         
    for file_name in tqdm.tqdm(os.listdir(input_path),desc="data need precess"):

        #file_name = "China_2105_01-07.parquet"
        data  = pd.read_parquet('%s/%s'%(input_path,file_name))
        sample = data.sample(100)
        # sample=data
          # text = sample['text'].values[1]
    # text_clean = clean_a_test(text)
    # print(text,"\n",text_clean)
        # tqdm.tqdm.pandas(desc=file_name)
        #sample['text_clean'] = sample.text.apply_parallel(clean_a_test_p, num_processes=6)
# 369 ms ± 70.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        t1 = datetime.now()
        x = sample["text"].map(CharTable.convert).map(lambda x:ht.clean_text(x,emoji=False))
        # TODO: 这里可以实现多线程进一步加快处理速度
        x = x.map(remove_url)
        # x = x.apply_parallel(remove_url,num_processes=6)
        # x = remove_url_s(x) 
        x = clean_s(x).map(
            remove_invisible_chars).map(deletehttp)
        # x = deletehttp_s(x)
        sample["text_clean"] = x
        # sample['text_clean'] = sample.progress_apply(clean_a_test,axis=1)
        print("use ",datetime.now()-t1)
        break
    # sample['text_clean'] = sample.apply(clean_a_test,axis=1)
    # 462 ms ± 36.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # sample.to_parquet("%s/%s"%(output_path,file_name))
    


