# SERVICE IMPORTS

import torch
import gradio as gr

"""
Translation Service
DEVELOPER DOCUMENTATION

Purpose: Translates English marketing text into Yoruba using the N-ATLAS language model.

Note: This module was exported from the working Kaggle prototype.
Its model objects, imports, and configuration may still depend
on the notebook environment. Integration work is required
before using it as a standalone application service.
"""

def translate_to_yoruba(english_text):
    if not english_text.strip():
        raise gr.Error("Please transcribe the video first.")

    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following English marketing text into "
                "natural Yoruba. Preserve the original meaning. "
                "Return only the Yoruba translation:\n\n"
                + english_text
            )
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda:0")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()



model = None
tokenizer = None

def configure_translation(translation_model, translation_tokenizer):
    """Register the loaded N-ATLAS model and tokenizer."""
    global model, tokenizer
    model = translation_model
    tokenizer = translation_tokenizer

