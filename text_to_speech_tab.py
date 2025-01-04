import gradio as gr
from utils import read_file_content

def create_text_to_speech_tab(models_list, choices, device_options, generate_audio_enhanced, update_input_visibility):
    with gr.Tab("Text-to-Speech"):
        with gr.Row():
            with gr.Column():
                # Text input options
                input_type = gr.Radio(
                    choices=["Direct Text", "File Upload"],
                    value="Direct Text",
                    label="Input Type"
                )
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text here...",
                    visible=True
                )
                file_input = gr.File(
                    label="Upload Text File",
                    visible=False
                )

                # Model and voice selection
                model_dropdown = gr.Dropdown(
                    list(models_list.items()),
                    label="Model",
                )
                voice_dropdown = gr.Dropdown(
                    list(choices.items()),
                    label="Voice",
                )

                # Generation settings
                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speed"
                )
                device_dropdown = gr.Dropdown(
                    device_options,
                    label="Device",
                    value="auto"
                )
                process_type = gr.Checkbox(
                    label="Process as Long Text",
                    value=True
                )
                generate_button = gr.Button("Generate")

            with gr.Column():
                # Output components
                audio_output = gr.Audio(label="Output Audio")
                text_output = gr.Textbox(label="Output Phonemes")
                status_output = gr.Textbox(label="Status", value="Ready")

        # Event handlers
        input_type.change(
            update_input_visibility,
            inputs=[input_type],
            outputs=[text_input, file_input]
        )
        file_input.change(
            lambda file: text_input.update(value=read_file_content(file) if file else ""),
            inputs=[file_input],
            outputs=[text_input]
        )

        def generate_wrapper(text, file, input_type, model_name, voice_name, speed, selected_device, is_long_text):
            try:
                # Determine input text
                if input_type == "File Upload" and file is not None:
                    text = read_file_content(file)
                elif input_type == "Direct Text" and not text:
                    text = ""
                elif input_type == "File Upload" and file is None:
                    text = ""

                # Update status
                yield None, None, "Processing..."

                # Generate audio
                audio_result, phonemes, wav_path = generate_audio_enhanced(
                    text=text,
                    model_name=model_name,
                    voice_name=voice_name,
                    speed=speed,
                    selected_device=selected_device,
                    is_long_text=is_long_text
                )

                status = "Generation complete!"
                yield audio_result, phonemes, status

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                yield None, error_msg, error_msg

        generate_button.click(
            generate_wrapper,
            inputs=[
                text_input,
                file_input,
                input_type,
                model_dropdown,
                voice_dropdown,
                speed_slider,
                device_dropdown,
                process_type
            ],
            outputs=[
                audio_output,
                text_output,
                status_output
            ]
        )

        return (input_type, text_input, file_input, model_dropdown, voice_dropdown,
                speed_slider, device_dropdown, process_type, generate_button,
                audio_output, text_output, status_output)