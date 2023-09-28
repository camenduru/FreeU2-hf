import gradio as gr
from PIL import Image  
import torch

from diffusers import StableDiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "stable-diffusion-v1-4"
pip_1_4 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_1_4 = pip_1_4.to("cuda")

model_id = "runwayml/stable-diffusion-v1-5"
pip_1_5 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_1_5 = pip_1_5.to("cuda")

model_id = "stabilityai/stable-diffusion-2-1"
pip_2_1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_2_1 = pip_2_1.to("cuda")

prompt_prev = None
sd_options_prev = None
seed_prev = None 
sd_image_prev = None

def infer(prompt, sd_options, seed, b1, b2, s1, s2):
    global prompt_prev
    global sd_options_prev
    global seed_prev
    global sd_image_prev

    if sd_options == 'SD1.5':
        pip = pip_1_5
    elif sd_options == 'SD2.1':
        pip = pip_2_1
    else:
        pip = pip_1_4

    run_baseline = False
    if prompt != prompt_prev or sd_options != sd_options_prev or seed != seed_prev:
        run_baseline = True
        prompt_prev = prompt
        sd_options_prev = sd_options
        seed_prev = seed

    if run_baseline:
        register_free_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
       
        torch.manual_seed(seed)
        print("Generating SD:")
        sd_image = pip(prompt, num_inference_steps=25).images[0]  
        sd_image_prev = sd_image
    else:
        sd_image = sd_image_prev

    
    register_free_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s1)
    register_free_crossattn_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s1)

    torch.manual_seed(seed)
    print("Generating FreeU:")
    freeu_image = pip(prompt, num_inference_steps=25).images[0]  

    # First SD, then freeu
    images = [sd_image, freeu_image]

    return images


examples = [
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ],
    [
        "half human half cat, a human cat hybrid",
    ],
    [
        "a hedgehog using a calculator",
    ],
    [
        "kanye west | diffuse lighting | fantasy | intricate elegant highly detailed lifelike photorealistic digital painting | artstation",
    ],
    [
        "astronaut pig",
    ],
    [
        "two people shouting at each other",
    ],
    [
        "A linked in profile picture of Elon Musk",
    ],
    [
        "A man looking out of a rainy window",
    ],
    [
        "close up, iron man, eating breakfast in a cabin, symmetrical balance, hyper-realistic --ar 16:9 --style raw"
    ],
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
    ],
    [
        'A mecha robot in a favela in expressionist style',
    ],
    [
        'an insect robot preparing a delicious meal',
    ],
]
    
    
css = """
h1 {
  text-align: center;
}

#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

block = gr.Blocks(css='style.css')

options = ['SD1.4', 'SD1.5', 'SD2.1']

with block:
    gr.Markdown("SD vs. FreeU.")
    with gr.Group():
        with gr.Row(): 
            sd_options = gr.Dropdown(["SD1.4", "SD1.5", "SD2.1"], label="SD options")

            # if sd_options == 'SD1.5':
            #     sd = 1.5
            # elif sd_options == 'SD2.1':
            #     sd = 2.1
            # else:
            #     sd = 1.4
            
            # pip = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            # pip = pip.to("cuda")
            
            with gr.Row():
                with gr.Row():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                    btn = gr.Button("Generate image", scale=0)
                
                seed = gr.Slider(label='seed',
                                        minimum=0,
                                        maximum=1000,
                                        step=1,
                                        value=42)

    
    with gr.Group():
        with gr.Row():
            with gr.Accordion('FreeU Parameters: b', open=True):
                b1 = gr.Slider(label='b1: backbone factor of the first stage block of decoder',
                                        minimum=1,
                                        maximum=1.6,
                                        step=0.01,
                                        value=1)
                b2 = gr.Slider(label='b2: backbone factor of the second stage block of decoder',
                                        minimum=1,
                                        maximum=1.6,
                                        step=0.01,
                                        value=1)
            with gr.Accordion('FreeU Parameters: s', open=True):
                s1 = gr.Slider(label='s1: skip factor of the first stage block of decoder',
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=1)
                s2 = gr.Slider(label='s2: skip factor of the second stage block of decoder',
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=1)    
                    
    with gr.Row():
        with gr.Group():
            # btn = gr.Button("Generate image", scale=0)
            with gr.Row():
                with gr.Column() as c1:
                    image_1 = gr.Image(interactive=False)
                    image_1_label = gr.Markdown("SD")
            
        with gr.Group():
            # btn = gr.Button("Generate image", scale=0)
            with gr.Row():
                with gr.Column() as c2:
                    image_2 = gr.Image(interactive=False)
                    image_2_label = gr.Markdown("FreeU")
        
        
    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2], cache_examples=False)
    ex.dataset.headers = [""]

    text.submit(infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2])
    btn.click(infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2])

block.launch()
# block.queue(default_enabled=False).launch(share=False)
