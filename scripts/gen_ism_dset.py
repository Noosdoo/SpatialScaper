import numpy as np
import spatialscaper as ss
import os
import sys

# Constants
NSCAPES = 20  # Number of soundscapes to generate
FOREGROUND_DIR = "/scratch/ci411/sounds/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    "/scratch/ci411/rirs/ism_RIRs"  # Directory containing Room Impulse Response (RIR) files
)
FORMAT = "foa"  # Output format specifier: could be 'mic' or 'foa'
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "/scratch/ci411/SELD/seld_datasets/ism_revert"  # Directory to store the generated soundscapes
REF_DB = (
    -65
)  # Reference decibel level for the background ambient noise. Try making this random too!

def generate_soundscape(room, room_idx, fold_idx, mix_idx):
    track_name = f"fold{str(fold_idx)}_room{str(room_idx)}_mix{mix_idx+1:03d}"
    # Initialize Scaper. 'max_event_overlap' controls the maximum number of overlapping sound events.
    ssc = ss.Scaper(
        DURATION,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        room,
        max_event_overlap=3,
        speed_limit=2.0,  # in meters per second
    )
    ssc.ref_db = REF_DB

    # static ambient noise
    #ssc.add_background()

    # Add a random number of foreground events, based on the specified mean and standard deviation.
    n_events = int(np.random.normal(N_EVENTS_MEAN, N_EVENTS_STD))
    n_events = n_events if n_events > 0 else 1  # n_events should be greater than zero

    for _ in range(n_events):
        ssc.add_event()  # randomly choosing and spatializing an FSD50K sound event

    audiofile = os.path.join(OUTPUT_DIR, "foa_dev", "audio", track_name)
    labelfile = os.path.join(OUTPUT_DIR, "metadata_dev", "labels", track_name)

    ssc.generate(audiofile, labelfile)

conf_list = [{'room': 'bomb_shelter', 'room_idx': 0, 'split': 1, 'n_mix': 150},
             {'room': 'gym', 'room_idx': 1, 'split': 1, 'n_mix': 150},
             {'room': 'pc226', 'room_idx': 2, 'split': 1, 'n_mix': 150},
             {'room': 'sa203', 'room_idx': 3, 'split': 1, 'n_mix': 150},
             {'room': 'sc203', 'room_idx': 4, 'split': 1, 'n_mix': 150},
             {'room': 'tb103', 'room_idx': 5, 'split': 1, 'n_mix': 150},
             {'room': 'pb132', 'room_idx': 0, 'split': 2, 'n_mix': 100},
             {'room': 'se203', 'room_idx': 1, 'split': 2, 'n_mix': 100},
             {'room': 'tc352', 'room_idx': 2, 'split': 2, 'n_mix': 100}]

conf = conf_list[int(sys.argv[1])]
print(f"Conf: {conf}")

for mix_idx in range(conf['n_mix']):
    print(f"Generating mix {mix_idx}")
    generate_soundscape(conf['room'], conf['room_idx'], conf['split'], mix_idx)








    