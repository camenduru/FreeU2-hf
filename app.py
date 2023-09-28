import gradio as gr
from PIL import Image  
import torch

from diffusers import StableDiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


torch.manual_seed(42)
model_id = "CompVis/stable-diffusion-v1-4"
        
# pip_sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pip_sd = pip_sd.to("cuda")

# pip_freeu = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pip_freeu = pip_freeu.to("cuda")
# # -------- freeu block registration
# register_free_upblock2d(pip_freeu, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
# register_free_crossattn_upblock2d(pip_freeu, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
# # -------- freeu block registration

def infer(prompt):
    
    print("Generating SD:")
    sd_image = pip_sd(prompt).images[0]  

    print("Generating FreeU:")
    freeu_image = pip_freeu(prompt).images[0]  

    # First SD, then freeu
    images = [sd_image, freeu_image]

    return images


examples = [
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ],
    [
        "a monkey doing yoga on the beach",
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

block = gr.Blocks(css=css)

options = ['SD1.4', 'SD1.5', 'SD2.1']

with block:
    gr.Markdown("SD vs. FreeU.")
    with gr.Row():
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
    
            with gr.Row():
                with gr.Column(min_width=256) as c1:
                    image_1 = gr.Image(interactive=False)
                    image_1_label = gr.Markdown("SD")
            
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
    
            with gr.Row():
                with gr.Column(min_width=256) as c2:
                    image_2 = gr.Image(interactive=False)
                    image_2_label = gr.Markdown("FreeU")
        
        
        # with gr.Column():
        #     sd_options = gr.Dropdown(options, value='SD1.4', label="SD options")
        #     model_id = "CompVis/stable-diffusion-v1-4"
        #     with gr.Accordion('FreeU Parameters', open=False):
        #         b1 = gr.Slider(label='b1: backbone factor of the first stage block of decoder',
        #                                 minimum=1,
        #                                 maximum=1.6,
        #                                 step=0.1,
        #                                 value=1)
        #         b2 = gr.Slider(label='b2: backbone factor of the second stage block of decoder',
        #                                 minimum=1,
        #                                 maximum=1.6,
        #                                 step=0.1,
        #                                 value=1)
        #         s1 = gr.Slider(label='s1: skip factor of the first stage block of decoder',
        #                                 minimum=0,
        #                                 maximum=1,
        #                                 step=0.1,
        #                                 value=1)
        #         s2 = gr.Slider(label='s2: skip factor of the second stage block of decoder',
        #                                 minimum=0,
        #                                 maximum=1,
        #                                 step=0.1,
        #                                 value=1)
    # with gr.Group():
    #     with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
    #         with gr.Column():
    #             text = gr.Textbox(
    #                 label="Enter your prompt",
    #                 show_label=False,
    #                 max_lines=1,
    #                 placeholder="Enter your prompt",
    #                 container=False,
    #             )
    #         btn = gr.Button("Generate image", scale=0)
        
    #     with gr.Row():
    #         with gr.Column(min_width=256) as c1:
    #             image_1 = gr.Image(interactive=False)
    #             image_1_label = gr.Markdown("SD")
    #         with gr.Column(min_width=256) as c2:
    #             image_2 = gr.Image(interactive=False)
    #             image_2_label = gr.Markdown("FreeU")
            

    ex = gr.Examples(examples=examples, fn=infer, inputs=[text], outputs=[image_1, image_2], cache_examples=False)
    ex.dataset.headers = [""]

    text.submit(infer, inputs=[text], outputs=[image_1, image_2])
    btn.click(infer, inputs=[text], outputs=[image_1, image_2])

block.launch()
# block.queue(default_enabled=False).launch(share=False)
