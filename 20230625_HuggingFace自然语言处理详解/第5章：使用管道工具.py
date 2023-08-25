def text_classification_test():
    # 第5章/文本分类
    from transformers import pipeline
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/distilbert-base-uncased-finetuned-sst-2-english"

    classifier = pipeline(task="sentiment-analysis", model=Path(f'{model_name_or_path}'), framework="pt")
    result = classifier("I hate you")[0]
    print(result)
    result = classifier("I love you")[0]
    print(result)


def question_answerer_test():
    # 第5章/阅读理解
    from transformers import pipeline
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/distilbert-base-cased-distilled-squad"
    question_answerer = pipeline(task="question-answering", model=Path(f'{model_name_or_path}'), framework="pt")
    context = r"""
    Extractive Question Answering is the task of extracting an answer from a text given a question. 
    An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. 
    If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/PyTorch/question-answering/run_squad.py script.
    """
    result = question_answerer(
        question="What is extractive question answering?",
        context=context,
    )
    print(result)
    result = question_answerer(
        question="What is a good example of a question answering dataset?",
        context=context,
    )
    print(result)



def fill_mask_test():
    # hf链接：https://huggingface.co/distilroberta-base
    # 第5章/完形填空
    from transformers import pipeline
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/distilroberta-base"
    unmasker = pipeline(task="fill-mask", model=Path(f'{model_name_or_path}'), framework="pt") #加载本地模型
    # unmasker = pipeline("fill-mask") #加载线上模型
    from pprint import pprint
    sentence = 'HuggingFace is creating a <mask> that the community uses to solve NLP tasks.'
    print(unmasker(sentence))



def text_generator_test():
    # hf链接：https://huggingface.co/gpt2
    #第5章/文本生成
    from transformers import pipeline
    # text_generator=pipeline("text-generation")
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/gpt2"
    text_generator = pipeline(task="text-generation", model=Path(f'{model_name_or_path}'), framework="pt")
    result = text_generator("As far as I am concerned, I will", max_length=50, do_sample=False)
    print(result)


def ner_pipe_test():
    # hf链接：https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english
    # 第5章/命名实体识别
    from transformers import pipeline
    # ner_pipe = pipeline("ner")
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/bert-large-cased-finetuned-conll03-english"
    ner_pipeline = pipeline(task="ner", model=Path(f'{model_name_or_path}'), framework="pt")
    sequence = """Hugging Face Inc. is a company based in New York City. Its
    headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."""
    for entity in ner_pipeline(sequence):
        print(entity)



def summarization_test():
    # https://huggingface.co/sshleifer/distilbart-cnn-12-6

    # 第5章/文本摘要
    from transformers import pipeline
    # summarizer = pipeline("summarization")
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/distilbart-cnn-12-6"
    summarizer = pipeline(task="summarization", model=Path(f'{model_name_or_path}'), framework="pt")
    ARTICLE = ARTICLE = """New York (CNN) When Liana Barrientos was 23 years old, she got married in Westchester County, 
     New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
     Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, 
     sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. 
     In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, 
     is facing two criminal counts of "offering a false instrument for filing in the first degree," 
     referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. 
     On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney,Christopher Wright, who declined to comment further. 
     After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit,
     said Detective Annette Markowski,a police spokeswoman.In total,Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002. 
     All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, 
     she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages. 
     Any divorces happened only after such filings were approved.It was unclear whether any of the men will be prosecuted. 
     The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. 
     Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. 
     Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. 
     If convicted, Barrientos faces up to four years in prison. Her next court appearance is scheduled for May 18. """
    result = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)
    print(result)


def translator_test():
    # hf链接：https://huggingface.co/t5-base
    # 第5章/翻译
    from transformers import pipeline
    # translator = pipeline("translation_en_to_de") #英文译德文
    from pathlib import Path
    model_name_or_path = "L:/20230713_HuggingFaceModel/t5-base"
    translator = pipeline(task="summarization", model=Path(f'{model_name_or_path}'), framework="pt")
    sentence = "Hugging Face is a technology company based in New York and Paris" #Hugging Face是一家总部位于纽约和巴黎的科技公司。
    result = translator(sentence, max_length=40)
    print(result)


def translator1_test():
    # 第5章/替换模型执行中译英任务
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    # 要使用该模型，需要安装sentencepiece
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    # translator = pipeline(task="translation_zh_to_en", model=model, tokenizer=tokenizer)

    # from pathlib import Path
    from pathlib import Path
    model = "L:/20230713_HuggingFaceModel/opus-mt-zh-en"
    tokenizer = "L:/20230713_HuggingFaceModel/opus-mt-zh-en"
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=Path(f'{model}'))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=Path(f'{tokenizer}'))
    translator = pipeline(task="translation_zh_to_en", model=model, tokenizer=tokenizer, framework="pt")

    sentence = "我叫萨拉，我住在伦敦。"
    result = translator(sentence, max_length=20)
    print(result)


def translator2_test():
    # 第5章/替换模型执行英译中任务
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    # 要使用该模型，需要安装sentencepiece
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # translator = pipeline(task="translation_zh_to_en", model=model, tokenizer=tokenizer)

    # from pathlib import Path
    from pathlib import Path
    model = "L:/20230713_HuggingFaceModel/opus-mt-en-zh"
    tokenizer = "L:/20230713_HuggingFaceModel/opus-mt-en-zh"
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=Path(f'{model}'))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=Path(f'{tokenizer}'))
    translator = pipeline(task="translation_zh_to_en", model=model, tokenizer=tokenizer, framework="pt")

    sentence = "My name is Sarah and I live in London"
    result = translator(sentence, max_length=20)
    print(result)


if __name__ == '__main__':
    # 文本分类
    # text_classification_test()

    # 阅读理解
    # question_answerer_test()

    # 完型填空
    # fill_mask_test()

    # 文本生成
    # text_generator_test()

    # 命名实体识别
    # ner_pipe_test()

    # 文本摘要
    # summarization_test()

    # 翻译任务
    # translator_test()

    # 中文翻译英文任务
    # translator1_test()

    # 英文翻译中文任务
    translator2_test()