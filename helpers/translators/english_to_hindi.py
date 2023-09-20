from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ENGLISH_TO_HINDI_MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=ENGLISH_TO_HINDI_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=ENGLISH_TO_HINDI_MODEL_NAME)


def english_to_hindi(english_text: str):
    translated_tokens = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))
    return [tokenizer.decode(token, skip_special_tokens=True) for token in translated_tokens]
