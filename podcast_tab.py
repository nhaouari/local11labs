import gradio as gr

import uuid
import json
from podcast_generator import generate_podcast_script
from src.long_speech_generation import generate_long_text_optimized, convert_to_mp3
from src.podcast_generation import merge_audio_files
from src.utils import read_file_content, generate_audio_enhanced
import os
from pprint import pprint

def create_podcast_dialogue_tab():
    with gr.Tab("Podcast Dialogue Generation"):
        gr.Markdown("Generate the dialogue script for your podcast.")
        with gr.Row():
            with gr.Column():
                gemini_api_key_textbox = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Enter your Gemini API key here...",
                    type="password"
                )
                podcast_hosts_textbox = gr.Textbox(
                    label="Podcast Host Names",
                    placeholder="Enter host names separated by commas (e.g., Noureddine, Noura)"
                )
                podcast_content_file_input = gr.File(
                    label="Upload Text File for Podcast Content"
                )
                generate_podcast_script_button = gr.Button("Generate Podcast Dialogue")
                send_to_audio_input_button = gr.Button("Send to Podcast Audio Input")

            with gr.Column():
                podcast_script_json_output = gr.JSON(label="Podcast Script")
                podcast_dialogue_status_output = gr.Textbox(label="Status", value="Ready")

        def generate_podcast_dialogue_wrapper(api_key, host_names_str, content_file):
            try:
                if not api_key:
                    raise ValueError("Please enter your Gemini API key.")
                if not host_names_str:
                    raise ValueError("Please enter the names of the podcast hosts.")
                if not content_file:
                    raise ValueError("Please upload a text file for the podcast content.")

                hosts = [h.strip() for h in host_names_str.split(',')]
                news_content = read_file_content(content_file)
                podcast_script_data = generate_podcast_script(api_key, news_content, hosts[0], hosts[1] if len(hosts) > 1 else None) # Assuming 2 hosts
                return {"script": podcast_script_data}, "Podcast dialogue generation complete!"
            except Exception as e:
                return None, str(e)

        generate_podcast_script_button.click(
            generate_podcast_dialogue_wrapper,
            inputs=[gemini_api_key_textbox, podcast_hosts_textbox, podcast_content_file_input],
            outputs=[podcast_script_json_output, podcast_dialogue_status_output]
        )
    return generate_podcast_script_button, send_to_audio_input_button, podcast_script_json_output, podcast_dialogue_status_output

#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
def generate_podcast_audio_segments(podcast_script, host_voice_map, model_name, speed, selected_device, output_dir,load_model_and_voice,process_type):
    """Generates audio segments for each dialogue entry in the podcast script."""
    audio_files = []
    for i, entry in enumerate(podcast_script):
        host = entry['host']
        dialogue = entry['dialogue']
        voice_config = host_voice_map.get(host)
        if not voice_config:
            raise ValueError(f"No voice configured for host: {host}")

        voice = voice_config['voice']
        lang = voice_config['lang']

        model, voice_data = load_model_and_voice(selected_device, model_name, voice)        
        try:
            # Generate audio and explicitly save WAV file
            # audio, phonemes, wav_path = generate_long_text_optimized(
            #     model=model,
            #     text=dialogue,
            #     voicepack=voice_data,
            #     lang=lang,
            #     output_dir=output_dir,
            #     verbose=False
            # )
            
            audio, phonemes, wav_path = generate_audio_enhanced(
                text=dialogue,
                model_name=model_name,
                voice_name=voice,
                selected_device=selected_device,
                speed=speed,
                is_long_text=process_type,
                load_model_and_voice=load_model_and_voice,
                
                
            )
            
            # Ensure WAV file exists before converting
            #the merge func seems using wav not mp3 , so ill let the convert after the merge
            if os.path.exists(wav_path):
                audio_files.append((i,wav_path))                
                # mp3_path = convert_to_mp3(wav_path)
                # if mp3_path and os.path.exists(mp3_path):
                #     audio_files.append((i,mp3_path)) #this takes tube i,mp3_path cz for loop in the merge function  takes i, (index, file) from enumerate so i add it to fix toomany value too many values to unpack (expected 2) etc..
                # else:
                #     print(f"Failed to create MP3 for segment {i}")
            else:
                print(f"WAV file not found for segment {i}: {wav_path}")
                
        except Exception as e:
            print(f"Error processing segment {i}: {str(e)}")
            continue

    return audio_files

def merge_podcast_audio(audio_files, output_path):
    """Merges the generated audio segments into a single podcast file."""
    try:
        if not audio_files:
            raise ValueError("No audio files to merge")
            
            
        merge_audio_files(audio_files, output_file=output_path)
        return output_path
    except Exception as e:
        print(f"Error MPA: {str(e)}")
        return None

#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------

def create_podcast_audio_tab(models_list, choices, device_options, kokoro_path,load_model_and_voice ):
    
    with gr.Tab("Podcast Audio Generation"):
        gr.Markdown("Generate audio for the podcast dialogue.")
        with gr.Row():
            with gr.Column():
                podcast_script_input_source = gr.Radio(
                    choices=["From Previous Tab", "Upload File", "Edit Here"],
                    value="Upload File",
                    label="Podcast Script Source"
                )
                podcast_script_json_input = gr.JSON(label="Podcast Script" , visible=False)
                podcast_script_textbox = gr.Code(label="Edit Podcast Script (JSON)", language="json", visible=False)
                podcast_script_file_input = gr.File(label="Upload Podcast Script (JSON)", visible=True)


                podcast_model_dropdown = gr.Dropdown(
                    list(models_list.items()),
                    label="Model",
                    value=os.path.join(kokoro_path, "fp16/kokoro-v0_19-half.pth"),
                )
                podcast_speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speed"
                )
                podcast_device_dropdown = gr.Dropdown(
                    device_options,
                    label="Device",
                    value="auto"
                )
                hosts_state =gr.State([])
                voice_components=[]
                @gr.render(inputs=hosts_state)
                def render_host_voice_assignment_inputs(hosts):
                    voice_components.clear()
                    with gr.Column() as column:  # Create a new column container
                        for host in hosts:
                            dropdown = gr.Dropdown(
                                list(choices.items()), 
                                label=f"Voice for {host}", 
                                value="af",
                                elem_id=f"voice_dropdown_{host}"
                            )
                            podcast_host_voice_assignment_inputs[f"dropdown_element_{host}"] = dropdown
                              
                    def update_host_voice_assignment_inputs(x,host):
                        print("-"*50)
                        print("Before")
                        pprint(podcast_host_voice_assignment_inputs)
                        print("-"*50)
                        print("After")
                        podcast_host_voice_assignment_inputs[host] = x
                        pprint(podcast_host_voice_assignment_inputs)
                        print("-"*50)
                    for key in podcast_host_voice_assignment_inputs:
                        if key.startswith('dropdown_element_'):
                            host = key.replace('dropdown_element_', '')
                            podcast_host_voice_assignment_inputs[key].change(
                                lambda x, h=host: update_host_voice_assignment_inputs(x, h),
                                inputs=[podcast_host_voice_assignment_inputs[key]]
                        )
                    
                podcast_host_voice_assignment_inputs = {}
                
                    
                def update_host_voice_assignment_inputs(podcast_script_json):
                    voice_components = []
                    hosts_g = []
                    podcast_host_voice_assignment_inputs.clear()

                    if podcast_script_json and "script" in podcast_script_json:
                        script = podcast_script_json["script"]["script"]
                        hosts = sorted(
                            list(
                                set(
                                    entry['host'] for entry in script if isinstance(entry, dict)  
                                )
                            )
                        )
                        hosts_g = hosts
                        for host in hosts:
                            podcast_host_voice_assignment_inputs[host] = "af" #seting default values
                            podcast_host_voice_assignment_inputs[f"dropdown_element_{host}"] = None #seting default values
                        
                    return hosts_g
                
                def on_script_change(podcast_script_json):
                    return update_host_voice_assignment_inputs(podcast_script_json)
                #! --------------------------------------------
                #! --------------------------------------------
                #! --------------------------------------------
                process_type = gr.Checkbox(
                    label="Process each speech as Long Text",
                    value=True
                )
                load_hosters_button = gr.Button("load hosters from json")
                generate_podcast_audio_button = gr.Button("Generate Podcast Audio")

            with gr.Column():
                podcast_audio_output = gr.Audio(label="Podcast Audio")
                podcast_audio_status_output = gr.Textbox(label="Status", value="Ready")
        def generate_podcast_audio_process(podcast_script_json, model, speed, device,process_type):
            try:
                if not podcast_script_json or "script" not in podcast_script_json:
                    raise ValueError("Please provide a podcast script.")

                podcast_script = podcast_script_json["script"]["script"]
                host_voice_map = {}
                for entry in podcast_script:
                    if isinstance(entry, dict):
                        host = entry['host']
                        host_voice_map[host] = {"voice": podcast_host_voice_assignment_inputs[host], "lang": podcast_host_voice_assignment_inputs[host][0]}
                        
                print("-"*50)
                print("Host Voice Map")
                pprint(host_voice_map)
                print("-"*50)
                
                
                output_dir = os.path.abspath('output/merges')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                      
                audio_files = generate_podcast_audio_segments(
                    podcast_script, host_voice_map, model, speed, device, output_dir, load_model_and_voice,process_type
                )
                output_file = os.path.join(output_dir, f"podcast_{uuid.uuid4()}.mp3")
                podcast_audio_path = merge_podcast_audio(audio_files, output_file)

                return podcast_audio_path, "Podcast audio generation complete!"
            except Exception as e:
                return None, str(e)

        
        
        def load_script_from_file(file):
            if file:
                loaded_json = json.loads(read_file_content(file))
                return loaded_json, gr.update(value=loaded_json)
            return None, gr.update(value=None)

        def update_script_input_display(source):
            if source == "From Previous Tab":
                return {
                    podcast_script_json_input: gr.update(visible=True),
                    podcast_script_file_input: gr.update(visible=False),
                    podcast_script_textbox: gr.update(visible=False)
                }
            elif source == "Upload File":
                return {
                    podcast_script_json_input: gr.update(visible=False),
                    podcast_script_file_input: gr.update(visible=True),
                    podcast_script_textbox: gr.update(visible=False)
                }
            elif source == "Edit Here":
                return {
                    podcast_script_json_input: gr.update(visible=False),
                    podcast_script_file_input: gr.update(visible=False),
                    podcast_script_textbox: gr.update(visible=True)
                }

        podcast_script_input_source.change(
            update_script_input_display,
            inputs=[podcast_script_input_source],
            outputs=[podcast_script_json_input, podcast_script_file_input, podcast_script_textbox]
        )

        podcast_script_file_input.upload(
            load_script_from_file,
            inputs=[podcast_script_file_input],
            outputs=[podcast_script_json_input, podcast_script_json_input]
        )

        podcast_script_textbox.change(
            lambda x: (json.loads(x) if x else None, gr.update(value=json.loads(x) if x else None)),
            inputs=[podcast_script_textbox],
            outputs=[podcast_script_json_input, podcast_script_json_input]
        )

        generate_podcast_audio_button.click(
            generate_podcast_audio_process,
            inputs=[podcast_script_json_input, podcast_model_dropdown, podcast_speed_slider, podcast_device_dropdown,process_type],
            outputs=[podcast_audio_output, podcast_audio_status_output]
        )


        load_hosters_button.click(
            fn=on_script_change,
            inputs=[podcast_script_json_input],
            outputs=[ hosts_state]
        )

        

        for trigger in [podcast_script_json_input,podcast_script_file_input, podcast_script_textbox]:
            trigger.change(
                fn=on_script_change,
                inputs=[podcast_script_json_input],
                outputs=[hosts_state]
            )
    return generate_podcast_audio_button, podcast_script_json_input, podcast_host_voice_assignment_inputs, podcast_model_dropdown, podcast_speed_slider, podcast_device_dropdown, podcast_audio_output, podcast_audio_status_output, load_hosters_button

#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
#! ------------------------------------------------------------------
def create_podcast_tab(models_list, choices, device_options, kokoro_path, load_model_and_voice):
    with gr.Tab("Podcast"):
      
        generate_podcast_script_button, send_to_audio_input_button, podcast_script_json_output, podcast_dialogue_status_output = create_podcast_dialogue_tab()

        generate_podcast_audio_button, podcast_script_json_input, podcast_host_voice_assignment_inputs, podcast_model_dropdown, podcast_speed_slider, podcast_device_dropdown, podcast_audio_output, podcast_audio_status_output, load_hosters_button = create_podcast_audio_tab(models_list, choices, device_options, kokoro_path, load_model_and_voice )

        send_to_audio_input_button.click(
            lambda script_json: (gr.update(value=script_json), script_json),
            inputs=[podcast_script_json_output],
            outputs=[
                podcast_script_json_input, 
                podcast_script_json_input,
            ]
        )
    return (generate_podcast_script_button, send_to_audio_input_button, podcast_script_json_output,
            podcast_dialogue_status_output, podcast_script_json_input, podcast_host_voice_assignment_inputs,
            podcast_model_dropdown, podcast_speed_slider, podcast_device_dropdown, generate_podcast_audio_button,
            podcast_audio_output, podcast_audio_status_output)