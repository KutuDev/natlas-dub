
"""N-ATLAS Video Localizer — Gradio application."""

import gradio as gr

from services import transcription, translation, speech_generation, video_processing


def configure_models(
    whisper_model,
    translation_model,
    translation_tokenizer,
    speech_model,
    speech_tokenizer,
):
    """Connect previously initialized AI models to the services."""
    transcription.configure_transcription(whisper_model)
    translation.configure_translation(
        translation_model, translation_tokenizer
    )
    speech_generation.configure_speech(
        speech_model, speech_tokenizer
    )


def render_localized_video(video_path, audio_path, yoruba_text):
    """Render the localized video and return preview and download paths."""
    output_path = video_processing.create_localized_video(
        video_path, audio_path, yoruba_text
    )
    return output_path, output_path


def build_app():
    """Create the English-to-Yoruba localization interface."""
    with gr.Blocks(title="N-ATLAS Video Localizer") as demo:
        gr.Markdown("# N-ATLAS Video Localizer")
        gr.Markdown(
            "Convert English marketing videos into Yoruba "
            "with human-reviewed translations."
        )

        video_input = gr.Video(label="Upload English Video")

        transcribe_button = gr.Button("1. Transcribe English Speech")
        english_text = gr.Textbox(
            label="English Transcription", lines=4
        )

        translate_button = gr.Button("2. Translate to Yoruba")
        yoruba_text = gr.Textbox(
            label="Review and Edit Yoruba Translation", lines=4
        )

        voice_button = gr.Button("3. Generate Yoruba Voice")
        voice_output = gr.Audio(
            label="Yoruba Narration", type="filepath"
        )

        render_button = gr.Button("4. Create Localized Video")
        final_video = gr.Video(label="Localized Video Preview")
        download_video = gr.File(label="Download Localized MP4")

        transcribe_button.click(
            transcription.transcribe_video,
            inputs=video_input,
            outputs=english_text,
        )

        translate_button.click(
            translation.translate_to_yoruba,
            inputs=english_text,
            outputs=yoruba_text,
        )

        voice_button.click(
            speech_generation.generate_yoruba_voice,
            inputs=yoruba_text,
            outputs=voice_output,
        )

        render_button.click(
            render_localized_video,
            inputs=[video_input, voice_output, yoruba_text],
            outputs=[final_video, download_video],
        )

    return demo


if __name__ == "__main__":
    raise SystemExit(
        "Initialize the AI models and call configure_models(...) "
        "before launching the application. See README.md."
    )
