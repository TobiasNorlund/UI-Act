# UI-Act

UI-Act is a Transformer model for interacting with a computer using the graphical user interface.
The model shares the same interface as a human operator, i.e. screen input and low-level mouse/keyboard output.
The motivation for this is to allow for seamless integration into human-computer workflows, where the model can naturally be trained using expert human demonstrations.

https://github.com/TobiasNorlund/UI-Act/assets/2678217/5683b08f-b9b0-401d-9984-e7fdb67e064e

## How it works

UI-Act is an encoder-decoder Transformer model, similar to a language model like T5.
The core difference is that the _decoder_ takes a screenshot as input, and predicts a low-level action such as a mouse click on a specific pixel location.
The action is also conditioned on text via cross-attention to the _encoder_, enabling instructive prompting.
The input screenshot is encoded into a (single) embedding using a convolutional neural network before being fed into the decoder.

![UI-Act model](/data/UI-Act-model.png)

Once the predicted action has been executed, a new screenshot is taken, encoded, and fed into the next position of the decoder.
This allows the action to also be conditioned on previous states, via causal self-attention in the decoder.

The actions are predicted using two linear heads based on the output hidden states from the decoder.
The first head classifies the _action_type_, including EOS. In the demo, only left clicks are valid.
In case of a click prediction, the second head classifies the (flattened) pixel location. 
To limit the number of locations, we classify onto a 4x4 pixel grid.  

For more details on the model architecture, see [ui_act/model.py](ui_act/model.py).


## Demo: Add and subtract using GNOME Calculator in Ubuntu

In the above demo, the model has been trained to compute simple (up to 2-digit) arithmetic expressions using an open and visible GNOME Calculator window.

The model is trained end-to-end from scratch on demonstrations, i.e. behavioral cloning, and learns implicitly to perform OCR to detect the calculator buttons.
When trained on demonstrations adding or subtracting numbers between 0-50, it also generalizes to numbers between 50-99.
Due to the translational invariant CNN representations, the window can be put anywhere on the screen, and in any size.
However, the model is very overfit to the visual environment it is trained on, and is easily confused by a new desktop background or resolution.
Having other windows open also puts it out of distribution.

This demo model is very small, with a total of only 1.8M parameters. This makes it lightweight to run, even on a laptop CPU.

**Try on your own:**

1. [Download](https://storage.googleapis.com/ui-act/UI-Act-VM.ova) this VirtualBox VM (user: ui-act, pw: ui-act)
2. [Import](https://www.alphr.com/ova-virtualbox/) it to VirtualBox
3. Start it, you'll be automatically signed in
4. If not set already, set the desktop resolution to 1920x1080 which is what the model was trained using
5. Open a Calculator window from the Ubuntu taskbar
6. Open a terminal and run:
```bash
# Navigate to the already cloned repo
cd ~/UI-Act

# Git pull latest version of repo
git pull

# Run start script to build/run provided docker container
./start.sh

# In docker, run demo script
python -m ui_act.tasks.gnome_calculator.run

# When prompted, enter an arithmetic expression (it's been trained to compute [0-50][+-][0-50])
```

## Vision

So far, the model is only trained on expert demonstrations for this simple toy task, and is very brittle for any visual deviations from its training data.
There is great potential in pre-training this model, and chances are it would both generalize much better (i.e. be less sensitive to invariant features such as OS theme setting or desktop background) and become more data efficient (i.e. learn a new task from fewer expert demonstrations).
For example, one can think of the following pre-training strategies:
 - **Pre-train model end-to-end using Video Pre-Training (VPT), similar to [OpenAI VPT](https://openai.com/research/vpt)**<br/>
   Using VPT, the full model can be pre-trained using e.g. instructive video tutorials. This requires Inverse Dynamics Models (IDM) to extract actions from video data.

 - **Pre-train visual encoder**<br/>
   The model is likely to benefit from only pre-training the visual encoder (CNN). For example, pre-training the visual encoder on detecting buttons/clickable regions has potential to yield useful representations for learning actions downstream.


## Feedback and contributions

This project is only in its infancy. If you have ideas, feedback or want to contribute, you are outmost welcome to create an Issue or reach out to tobias@norlund.se!
