import gradio as gr
from PIL import Image  
import torch

from diffusers import DiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d



model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "./stable-diffusion-2-1"
pip_2_1 = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_2_1 = pip_2_1.to("cuda")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pip_XL = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_XL = pip_XL.to("cuda")

prompt_prev = None
sd_options_prev = None
seed_prev = None 
sd_image_prev = None

def infer(prompt, sd_options, seed, b1, b2, s1, s2):
    global prompt_prev
    global sd_options_prev
    global seed_prev
    global sd_image_prev

    if sd_options == 'SD2.1':
        pip = pip_2_1
    elif sd_options == 'SDXL':
        pip = pip_XL
    else:
        pip = pip_2_1

    # pip = pip_2_1

    run_baseline = False
    if prompt != prompt_prev or sd_options != sd_options_prev or seed != seed_prev:
        run_baseline = True
        prompt_prev = prompt
        sd_options_prev = sd_options
        seed_prev = seed

    if run_baseline:
        # register_free_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
       
        torch.manual_seed(seed)
        print("Generating SD:")
        sd_image = pip(prompt).images[0]  
        sd_image_prev = sd_image
    else:
        sd_image = sd_image_prev

    
    # register_free_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s1)
    register_free_crossattn_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s1)

    torch.manual_seed(seed)
    print("Generating FreeU:")
    freeu_image = pip(prompt).images[0]  

    # First SD, then freeu
    images = [sd_image, freeu_image]

    return images


examples = [
    [
        "RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, clearly face, a futuristic visage with cybernetic enhancements seamlessly integrated into human features",
    ],
    [
        "Sculpt a life-sized animal using discarded plastic bottles and metal scraps, highlighting it's beauty, highly detailed, 8k",
    ],
    [
        "A robot standing in the rain reading newspaper, rusty and worn down,  in a dystopian cyberpunk street, photo-realistic , urbanpunk",
    ],
    [
        "an outdoor full size sculpture using discarded car parts, highlighting it's beauty, highly detailed, 8k",
    ],
    [
        "1955, moon landing, sci-fi, 8k, photorealistic, no atmosphere, earth in the sky, terraforming, style by Dean ellis",
    ],
    [
        "a futuristic home , spaceship design,beautiful interior , high end design",
    ],
    [
        "Hypnotic Maze, Fantasy Castle, Challenging Maze, Impossible Geometry, Mc Escher, Surreal Photography Within A Glass Sphere, Diorama, Beautiful Abundance, Medieval detailing , Digital Painting, Digital Illustration, Extreme Detail, Digital Art, 8k, Ultra Hd, Fantasy Art, Hyper Detailed, Hyperrealism, Elaborate, Vray, Unrea",
    ],
    [
        "photo of half life combine standing outside city 17, glossy robot, rainy, rtx, octane, unreal",
    ],
    [
        "new art : landscape into a Underground oasis in egypt. satara by johnny taylor, in the style of brushstroke-inmersive landscape, cinematic elegance, golden light, dark proportions, flowing brushwork, multilayered realism,  --ar 61:128 --s 750 --v 5.2",
    ],
    [
        "A horse galloping on the ocean",
    ],
    [
        "a teddy bear walking in the snowstorm"
    ],
    [
        "Campfire at night in a snowy forest with starry sky in the background."
    ],
    [
        "a fantasy landscape, trending on artstation"
    ],
    [
        "An astronaut flying in space, 4k, high resolution."
    ],
    [
        "An astronaut is riding a horse in the space in a photorealistic style."
    ],
    [
        "Turtle swimming in ocean."
    ],
    [
        "A storm trooper vacuuming the beach."
    ],
    [
        "Fireworks."
    ],
    [
        "A fat rabbit wearing a purple robe walking through a fantasy landscape."
    ],
    [
        "A koala bear playing piano in the forest."
    ],
    [
        "An astronaut flying in space, 4k, high resolution."
    ],
    [
        "Flying through fantasy landscapes, 4k, high resolution."
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ],
    [
        "half human half cat, a human cat hybrid",
    ],
    [
        "a drone flying over a snowy forest."
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

options = ['SD2.1']

with block:
    gr.Markdown("# SD vs. FreeU")
    with gr.Group():
        with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
            with gr.Column():
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                    )
            btn = gr.Button("Generate image", scale=0)
        
    with gr.Group():
        with gr.Row():
            with gr.Accordion('FreeU Parameters (feel free to adjust these parameters based on your prompt): ', open=False):
                with gr.Row():             
                    sd_options = gr.Dropdown(["SD2.1", "SDXL"], label="SD options", value="SDXL", visible=True)
                with gr.Row():
                    b1 = gr.Slider(label='b1: backbone factor of the first stage block of decoder',
                                            minimum=1,
                                            maximum=2.0,
                                            step=0.01,
                                            value=1.3)
                    b2 = gr.Slider(label='b2: backbone factor of the second stage block of decoder',
                                            minimum=1,
                                            maximum=2.0,
                                            step=0.01,
                                            value=1.4)
                with gr.Row():
                    s1 = gr.Slider(label='s1: skip factor of the first stage block of decoder',
                                            minimum=0,
                                            maximum=1,
                                            step=0.1,
                                            value=0.9)
                    s2 = gr.Slider(label='s2: skip factor of the second stage block of decoder',
                                            minimum=0,
                                            maximum=1,
                                            step=0.1,
                                            value=0.2)    
                
                seed = gr.Slider(label='seed',
                             minimum=0,
                             maximum=1000,
                             step=1,
                             value=42)
                    
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
