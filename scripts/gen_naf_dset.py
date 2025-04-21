import numpy as np
import spatialscaper as ss
import os
import sys

# Constants
NSCAPES = 20  # Number of soundscapes to generate
FOREGROUND_DIR = "/scratch/ci411/sounds/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    "/scratch/ci411/rirs/naf_RIRs"  # Directory containing Room Impulse Response (RIR) files
)
FORMAT = "foa"  # Output format specifier: could be 'mic' or 'foa'
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "/scratch/ci411/SELD/seld_datasets/NAFBaseline"  # Directory to store the generated soundscapes
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

room_idx = int(sys.argv[1])

room = f'naf_room{room_idx}'
if room_idx>=6:
    n_mix = 100
    split = 2
    room_idx = room_idx-6
else:
    n_mix = 150
    split = 1
    
conf = {'room': room, 'room_idx': room_idx, 'split': split, 'n_mix': n_mix}
print(f"Conf: {conf}")

for mix_idx in range(conf['n_mix']):
    generate_soundscape(conf['room'], conf['room_idx'], conf['split'], mix_idx)








    