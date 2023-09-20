# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# ENGLISH_TO_TAMIL_MODEL_NAME = "facebook/nllb-200-distilled-600M"
#
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=ENGLISH_TO_TAMIL_MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=ENGLISH_TO_TAMIL_MODEL_NAME)


def english_to_tamil(english_text: str):
    tokenizer.tgt_lang = "ta"

    translated_tokens = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))
    return [tokenizer.decode(token, skip_special_tokens=True) for token in translated_tokens]
