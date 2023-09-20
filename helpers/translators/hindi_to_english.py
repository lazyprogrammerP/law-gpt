from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HINDI_TO_ENGLISH_MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=HINDI_TO_ENGLISH_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=HINDI_TO_ENGLISH_MODEL_NAME)


def hindi_to_english(hindi_text: str):
    translated_tokens = model.generate(**tokenizer(hindi_text, return_tensors="pt", padding=True))
    return [tokenizer.decode(token, skip_special_tokens=True) for token in translated_tokens]
