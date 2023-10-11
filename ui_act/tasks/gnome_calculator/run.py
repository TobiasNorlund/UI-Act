import json
import numpy as np
import torch
import time
import pyautogui
import matplotlib.pyplot as plt
import subprocess
from time import sleep
from PIL import ImageGrab
from pathlib import Path
from ui_act.model import UIActModelConfig, UIActLightningModel
from .data import encode
from Xlib import X, display


def get_active_window_id():
    process = subprocess.run(['xdotool', 'getactivewindow'], 
        stdout=subprocess.PIPE, 
        universal_newlines=True
    )
    window_id = process.stdout
    return window_id


def minimize_window(window_id):
    subprocess.run(['xdotool', 'windowminimize', str(window_id)])


def activate_window(window_id):
    subprocess.run(['xdotool', 'windowactivate', str(window_id)])


class ActionSuggestionIndicator:
    def __init__(self):
        self.circle_size = 15
        self.d = display.Display()
        root = self.d.screen().root
        default_screen = self.d.screen()
        attrs = {
            'override_redirect': True,
            'colormap': self.d.screen().default_colormap
        }
        vinfo = X.CopyFromParent
        self.overlay = root.create_window(
            1000, 1000, self.circle_size, self.circle_size, 0, 
            default_screen.root_depth, X.InputOutput, 
            vinfo,
            **attrs
        )
        red_color = default_screen.default_colormap.alloc_color(65535, 0, 0)
        self.gc_red = self.overlay.create_gc(foreground=red_color.pixel)

    def show_at(self, x, y):
        self.overlay.configure(x=x-self.circle_size//2, y=y-self.circle_size//2)
        start_time = time.time()
        while time.time() - start_time < 1:
            self.overlay.map()

            # Draw the circle
            self.overlay.fill_arc(self.gc_red, 0, 0, self.circle_size, self.circle_size, 0, 360 * 64)
            self.d.sync()
            time.sleep(0.2)

            self.overlay.unmap()
            self.d.sync()
            time.sleep(0.2)


# Create indicator for indicating next action prediction
indicator = ActionSuggestionIndicator()


def execute_rpa(text: str, config: UIActModelConfig, model: UIActLightningModel):
    text = encode(text)
    text = torch.tensor(text)[None, ...]
    text_attention_mask = torch.ones_like(text, dtype=torch.long)

    action = 0
    frames = torch.zeros((1, 0, config.frame_resolution.height, config.frame_resolution.width, config.frame_input_channels), dtype=torch.uint8)
    frames_attention_mask = torch.ones((1, 0), dtype=torch.long)

    while action != 1:

        frame = ImageGrab.grab()
        org_width = frame.size[0]
        frame = frame.resize(config.frame_resolution)
        scale_factor = org_width / frame.size[0] * 4 # Mult by 4 since cursor preds are downsampled by 4x
        frames = torch.cat((
            frames,
            torch.tensor(np.array(frame))[None, None, ...], # Convert to tensor and add batch + frames dim
        ), dim=1)
        frames_attention_mask = torch.cat((
            frames_attention_mask,
            torch.ones((1, 1), dtype=torch.long)
        ), dim=1)

        res = model(
            frames, 
            frames_attention_mask, 
            text, 
            text_attention_mask
        )

        pred_pos_flat = res.cursor_position_logits[0, -1].argmax()
        pred_pos_y = int(pred_pos_flat / (config.frame_resolution.width / 4) * scale_factor)
        pred_pos_x = int(pred_pos_flat % (config.frame_resolution.width / 4) * scale_factor)

        action = res.event_logits[0, -1].argmax()

        if action == 0:
            indicator.show_at(x=pred_pos_x, y=pred_pos_y)
            pyautogui.click(
                x=pred_pos_x,
                y=pred_pos_y
            )


def main(args):
    config = UIActModelConfig(**json.load(args.config.open()))
    model = UIActLightningModel.load_from_checkpoint(args.checkpoint, config=config, map_location=torch.device("cpu")).eval()

    while True:
        try:
            text = input("Input arithmetic expression:\n")
            if "=" not in text:
                # Add ending "=" if user didn't
                text = text + "="

            # minimize terminal window
            terminal_window_id = get_active_window_id()
            minimize_window(terminal_window_id)
            pyautogui.moveTo(100, 100)

            sleep(2)  # Add some sleep to have time to minimize terminal window
            execute_rpa(text, config, model)

            # activate terminal window (make it visible again)
            activate_window(terminal_window_id)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/workspace/data/gnome_calculator_model/model_config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/data/gnome_calculator_model/epoch=14-step=1875.ckpt"))
    args = parser.parse_args()
    main(args)