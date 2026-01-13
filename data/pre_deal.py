

#1.处理预训练数据
def split_text(text,chunk_size=512):
    """将文本按照指定长度分块"""
    return [text[i:i+chunk_size] for i in range(0,len(text),chunk_size)]