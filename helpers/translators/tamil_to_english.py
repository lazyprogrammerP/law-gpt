# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# TAMIL_TO_ENGLISH_MODEL_NAME = "facebook/nllb-200-distilled-600M"
#
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=TAMIL_TO_ENGLISH_MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=TAMIL_TO_ENGLISH_MODEL_NAME)


def tamil_to_english(tamil_text: str):
    tokenizer.tgt_lang = "en"

    translated_tokens = model.generate(**tokenizer(tamil_text, return_tensors="pt", padding=True))
    return [tokenizer.decode(token, skip_special_tokens=True) for token in translated_tokens]
