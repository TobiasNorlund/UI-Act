import decord
import av
import json
import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import List
from torch.utils.data import Dataset, DataLoader
from ui_act.model import ActionType, Resolution, CursorPosition, UIActModelInput
from decord import VideoReader, cpu


VIDEO_FILE_NAME = "screen.mp4"
METADATA_FILE_NAME = "events.jsonl"
VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "-": 11,
    "=": 12
}
INV_VOCAB = {v:k for k, v in VOCAB.items()}


def encode(text: str):
    return [VOCAB[c] for c in text]


def decode(encoded_text: List[int]):
    return "".join([INV_VOCAB[token] for token in encoded_text])


def get_target_resolution(org_width: int, org_height: int, max_width: int, max_height: int):
    if org_width / org_height >= max_width / max_height:
        return max_width, int(max_width / org_width * org_height)
    else:
        return int(max_height / org_height * org_width), max_height


class RPADataset(Dataset):

    event_class_map = {
        ActionType.LEFT_CLICK: 0,
        ActionType.END: 1
    }

    def __init__(
        self, 
        screencast_paths: List[Path], 
        resolution: Resolution, 
        frame_lag: int, 
        add_cursor_channel: bool,
        start_frame_idx: int=0
    ):
        super().__init__()
        self.screencast_paths = screencast_paths
        self.resolution = resolution
        self.frame_lag = frame_lag
        self.add_cursor_channel = add_cursor_channel
        self.start_frame_idx = start_frame_idx

        self.cache = [None] * len(self)

        decord.bridge.set_bridge("torch")

    def __len__(self):
        return len(self.screencast_paths)

    def __getitem__(self, screencast_index: int):
        if self.cache[screencast_index] is not None:
            return self.cache[screencast_index]

        # Tokenize text
        text = self.screencast_paths[screencast_index].name
        tokenized_text = encode(text)

        # Load screencast metadata
        with (self.screencast_paths[screencast_index] / METADATA_FILE_NAME).open() as metadata_file:
            metadata = [json.loads(line) for line in metadata_file]
        
        # Extract click frames + cursor pos
        frame_indices = [self.start_frame_idx]
        frame_cursor_positions = [CursorPosition(x=metadata[self.start_frame_idx]["cur_x"], y=metadata[self.start_frame_idx]["cur_y"])]
        target_cursor_positions = []
        target_events = []
        for frame_idx, frame_meta in enumerate(metadata):
            if "events" in frame_meta:
                for event in frame_meta["events"]:
                    if event["type"] == "ButtonPress" and event["code"] == 1:
                        # Add this event + cursor_pos as targets for last frame in frame_indices
                        target_events.append(self.event_class_map[ActionType.LEFT_CLICK])
                        target_cursor_positions.append(
                            CursorPosition(x=frame_meta["cur_x"], y=frame_meta["cur_y"])
                        )
                    elif event["type"] == "ButtonRelease" and event["code"] == 1:
                        next_input_frame = frame_idx + self.frame_lag
                        frame_indices.append(next_input_frame)  # TODO: What if this is out of range?
                        frame_cursor_positions.append(
                            CursorPosition(
                                x=metadata[next_input_frame]["cur_x"], 
                                y=metadata[next_input_frame]["cur_y"]
                            )
                        )
        # Add END event
        target_events.append(self.event_class_map[ActionType.END])
        target_cursor_positions.append(CursorPosition(x=0, y=0))

        assert len(frame_indices) == len(target_cursor_positions) == len(target_events)

        # Read video metadata
        with av.open(str(self.screencast_paths[screencast_index] / VIDEO_FILE_NAME), metadata_errors="ignore") as container:
            target_width, target_height = get_target_resolution(
                org_width=container.streams.video[0].width,
                org_height=container.streams.video[0].height,
                max_width=self.resolution.width,
                max_height=self.resolution.height
            )
            scaling_factor = target_width / container.streams.video[0].width

        # Load video frames
        reader = VideoReader(
            str(self.screencast_paths[screencast_index] / VIDEO_FILE_NAME), 
            ctx=cpu(0), 
            width=target_width, 
            height=target_height
        )
        frames = reader.get_batch(frame_indices)  # [frames, target_height, target_width, 3]
        if self.add_cursor_channel:
            cursor_channel = torch.zeros((*frames.shape[:-1], 1), dtype=torch.uint8) # [frames, target_height, target_width, 1]
            frame_cursor_positions = torch.tensor(frame_cursor_positions) * scaling_factor # [frames, 2]
            frame_cursor_positions = torch.round(frame_cursor_positions).type(torch.long)
            cursor_channel[
                torch.arange(frames.shape[0]), 
                frame_cursor_positions[:, 1], 
                frame_cursor_positions[:, 0],
                0
            ] = 255
            frames = torch.cat((frames, cursor_channel), dim=-1)

        frames_attention_mask = torch.ones((frames.shape[0],), dtype=torch.long)
        target_cursor_positions = torch.tensor(target_cursor_positions) * scaling_factor  # [frames, 2]
        target_events = torch.tensor(target_events)  # [frames]
        tokenized_text = torch.tensor(tokenized_text)  # [text len]
        text_attention_mask = torch.ones_like(tokenized_text)
        
        ex = UIActModelInput(frames, frames_attention_mask, tokenized_text, text_attention_mask, target_events, target_cursor_positions)
        self.cache[screencast_index] = ex

        return ex

def collate_fn(batch: List[UIActModelInput]):
    num_frames = max(ex.frames.shape[0] for ex in batch)
    text_len = max(ex.text.shape[0] for ex in batch) if batch[0].text is not None else None

    frames = torch.stack([
        torch.nn.functional.pad(ex.frames, (0,0,0,0,0,0,0, num_frames - ex.frames.shape[0]))
        for ex in batch
    ])
    frames_attention_mask = torch.stack([
        torch.nn.functional.pad(ex.frames_attention_mask, (0, num_frames - ex.frames_attention_mask.shape[0]))
        for ex in batch
    ])
    events = torch.stack([
        torch.nn.functional.pad(ex.events, (0, num_frames - ex.events.shape[0]))
        for ex in batch
    ])
    cursor_positions = torch.stack([
        torch.nn.functional.pad(ex.cursor_positions, (0,0,0, num_frames - ex.cursor_positions.shape[0]))
        for ex in batch
    ])
    text = torch.stack([
        torch.nn.functional.pad(ex.text, (0, text_len - ex.text.shape[0]))
        for ex in batch
    ]) if batch[0].text is not None else None
    text_attention_mask = torch.stack([
        torch.nn.functional.pad(ex.text_attention_mask, (0, text_len - ex.text_attention_mask.shape[0]))
        for ex in batch
    ]) if batch[0].text_attention_mask is not None else None
    target_cursor_positions = torch.stack([
        torch.nn.functional.pad(ex.target_cursor_positions, (0,0,0, num_frames - ex.target_cursor_positions.shape[0]))
        for ex in batch
    ])
    target_events = torch.stack([
        torch.nn.functional.pad(ex.target_events, (0, num_frames - ex.target_events.shape[0]), value=-100)
        for ex in batch
    ])

    return UIActModelInput(frames, frames_attention_mask, events, cursor_positions, text, text_attention_mask, target_events, target_cursor_positions)


class GnomeCalculatorDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        resolution: Resolution, 
        frame_lag: int,
        add_cursor_channel: bool, 
        batch_size: int,
        num_workers: int
    ):
        super().__init__()
        self.base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "screen_recordings" / "gnome_calculator"
        self.resolution = resolution
        self.frame_lag = frame_lag
        self.add_cursor_channel = add_cursor_channel
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_ds = RPADataset(
            list((self.base_dir / "add-sub-1/").glob("*/")),
            resolution=self.resolution,
            frame_lag=self.frame_lag,
            add_cursor_channel=self.add_cursor_channel,
            start_frame_idx=10  # Sometimes the calculator window isn't visible until a few frames
        )
        self.val_ds = RPADataset(
            list((self.base_dir / "add-sub-2/").glob("*/")),
            resolution=self.resolution,
            frame_lag=self.frame_lag,
            add_cursor_channel=self.add_cursor_channel,
            start_frame_idx=10
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True
        )


if __name__ == "__main__":
    from tqdm import tqdm
    ds = RPADataset(
        list(Path("../data/screen_recordings/gnome_calculator/add-sub-1/").glob("*/")),
        resolution=Resolution(width=512, height=288),
        frame_lag=0,
        add_cursor_channel=True
    )
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    for ex in tqdm(dl):
        pass