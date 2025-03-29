# -*- coding: utf-8 -*-
import json

import gradio as gr


def load_data(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


data_list = load_data("data/dataset/metadata.jsonl")
total_samples = len(data_list)
batch_size = 5


def update_display(start):
    current_data = data_list[start : start + batch_size]

    outputs = []
    for item in current_data:
        audio_path = item["audio"]["path"]
        transcript = item["text"]
        audio = gr.Audio(value=audio_path)
        text = gr.Textbox(value=transcript)
        outputs.extend([audio, text])
    return outputs


def prev_click(start):
    start = int(start)
    new_start = max(0, start - batch_size)
    return new_start


def next_click(start):
    start = int(start)
    max_start = max(0, len(data_list) - batch_size)
    new_start = min(start + batch_size, max_start)
    return new_start


def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            prev_btn = gr.Button("上一组")
            next_btn = gr.Button("下一组")

            start_input = gr.Slider(
                maximum=total_samples - 1,
                step=1,
                label="起始序号",
                interactive=True,
            )

        components = []
        for item in data_list[:batch_size]:
            with gr.Row():
                audio_path = item["audio"]["path"]
                audio_source = item["source"]
                transcript = item["text"]
                audio = gr.Audio(value=audio_path, label=audio_source)
                text = gr.Textbox(value=transcript)
                components.extend([audio, text])

        start_input.change(update_display, start_input, components)
        prev_btn.click(prev_click, inputs=[start_input], outputs=start_input).then(
            update_display,
            inputs=[start_input],
            outputs=components,
        )
        next_btn.click(next_click, inputs=[start_input], outputs=start_input).then(
            update_display,
            inputs=[start_input],
            outputs=components,
        )
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
