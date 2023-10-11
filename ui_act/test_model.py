import torch
from .model import *


def _get_test_model_config():
    return UIActModelConfig(
        text_conditioned=True,
        text_num_embeddings=10,
        text_max_length=10,
        text_hidden_dim=16,
        text_num_heads=4,
        text_ffn_dim=64,
        text_num_layers=2,
        text_dropout_p=0.1,

        frame_input_channels=4,
        frame_resolution=Resolution(width=128, height=72),
        frame_base_feature_maps=16,
        frame_residual_layers=0,
        frame_feature_maps=32,

        action_max_length=10,
        action_hidden_dim=32,
        action_num_heads=4,
        action_ffn_dim=256,
        action_num_layers=2,
        action_dropout_p=0.1,

        event_classes=[ActionType.LEFT_CLICK, ActionType.END]
    )


def _get_test_input(
    config: UIActModelConfig,
    batch_size=1,
    num_frames=5,
    text_len=8,
):
    frames = torch.randint(0, 256, (batch_size, num_frames, config.frame_resolution.height, config.frame_resolution.width, 4), dtype=torch.uint8)
    frames_attention_mask = torch.ones((batch_size, num_frames), dtype=torch.long)
    text = torch.randint(0, config.text_num_embeddings, (batch_size, text_len), dtype=torch.long)
    text_attention_mask = torch.ones_like(text)
    target_events = torch.randint(0, 1, (batch_size, num_frames))
    target_cursor_positions = torch.rand((batch_size, num_frames, 2))
    target_cursor_positions[:, :, 0] *= config.frame_resolution.width // 4
    target_cursor_positions[:, :, 1] *= config.frame_resolution.height // 4

    return frames, frames_attention_mask, text, text_attention_mask, target_events, target_cursor_positions


def set_gradient(obj, name):
    def _set_fn(grad):
        obj[name] = grad
    return _set_fn


def test_batch_independence():
    """
    Test that any output hidden state in one batch example has a
    zero gradient wrt input hidden states of another batch example
    """

    config = _get_test_model_config()
    model = UIActModel(config)

    batch  = _get_test_input(config, batch_size=2)
    output = model(*batch, return_all_hidden_states=True)
    
    input_hidden_states = output.hidden_states[0]
    output_hidden_states = output.hidden_states[-1]

    gradients = {}
    input_hidden_states.register_hook(set_gradient(gradients, "input_hidden_states"))
    
    l = output_hidden_states[0].sum()  # Sum all output hidden states for first batch example
    l.backward()

    # sum of absolute values over all input hidden state gradients
    gradient_magnitudes_per_batch_ex = gradients["input_hidden_states"].abs().sum(dim=[1,2])  
    assert gradient_magnitudes_per_batch_ex[0] > 0., "Gradient should be greater than zero for first example"
    assert gradient_magnitudes_per_batch_ex[1] == 0., "Gradient should be equal to zero for second example"


def test_causality():
    """
    Test that any output hidden state has zero gradient wrt input hidden states of following positions
    """

    config = _get_test_model_config()
    model = UIActModel(config)

    batch = _get_test_input(config)
    output = model(*batch, return_all_hidden_states=True)

    input_hidden_states = output.hidden_states[0]
    output_hidden_states = output.hidden_states[-1]

    gradients = {}
    input_hidden_states.register_hook(set_gradient(gradients, "input_hidden_states"))

    l = output_hidden_states[0, 1, :].sum() # Take the second position's output hidden states
    l.backward()

    gradient_magnitudes_per_pos = gradients["input_hidden_states"][0, :, :].abs().sum(-1)

    assert all(gradient_magnitudes_per_pos[:2] > 0.), "Gradient magnitudes should be greater than zero for previous and current positions"
    assert all(gradient_magnitudes_per_pos[2:] == 0.), "Gradient magnitudes should be equal to zero for future positions"
    

def test_text_masking():
    """
    Test that:
    1. output text hidden states have zero gradient wrt masked input text hidden states
    2. output hidden states have zero gradient wrt masked input text hidden states
    """

    config = _get_test_model_config()
    model = UIActModel(config)

    frames, frames_attention_mask, text, text_attention_mask, target_events, target_cursor_positions = \
        _get_test_input(config)
    text_attention_mask[0, -1] = 0  # Mask last text token
    
    # 1.
    output = model(
        frames, 
        frames_attention_mask, 
        text, 
        text_attention_mask, 
        target_events, 
        target_cursor_positions, 
        return_all_hidden_states=False,
        return_all_text_hidden_states=True
    )

    input_text_hidden_states = output.text_hidden_states[0]
    output_text_hidden_states = output.text_hidden_states[-1]

    gradients = {}
    input_text_hidden_states.register_hook(set_gradient(gradients, "input_text_hidden_states"))

    l = output_text_hidden_states[0, 0, :].sum()  # Take gradient of first token with is unmasked
    l.backward()

    gradient_magnitudes_per_pos = gradients["input_text_hidden_states"][0, :, :].abs().sum(-1)
    assert gradient_magnitudes_per_pos[-1] == 0., "Gradient should be zero for masked text token"
    assert all(gradient_magnitudes_per_pos[:-1] > 0.), "Gradient should be non-zero for unmasked text tokens"

    # 2.
    model.zero_grad()
    output = model(
        frames, 
        frames_attention_mask, 
        text, 
        text_attention_mask, 
        target_events, 
        target_cursor_positions, 
        return_all_hidden_states=False,
        return_all_text_hidden_states=False
    )

    output_hidden_states = output.hidden_states
    output_text_hidden_states = output.text_hidden_states

    gradients = {}
    output_text_hidden_states.register_hook(set_gradient(gradients, "output_text_hidden_states"))

    l = output_hidden_states.sum()
    l.backward()

    gradient_magnitudes_per_pos = gradients["output_text_hidden_states"][0, :, :].abs().sum(-1)
    assert gradient_magnitudes_per_pos[-1] == 0., "Gradient should be zero for masked text token"
    assert all(gradient_magnitudes_per_pos[:-1] > 0.), "Gradient should be non-zero for unmasked text tokens"


if __name__ == "__main__":
    pass
