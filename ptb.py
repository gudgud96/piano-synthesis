import os
import pretty_midi
import magenta
from magenta.music import events_lib
from magenta.music import melodies_lib
from magenta.music import Melody
from magenta.music import PolyphonicMelodyError
from magenta.music import sequences_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
import numpy as np
from magenta.models.score2perf.music_encoders import MidiPerformanceEncoder, TextMelodyEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import random

PR_TIME_STEPS = 64
NUM_VELOCITY_BINS = 64
STEPS_PER_SECOND = 100
STEPS_PER_QUARTER = 12
MIN_PITCH = 21
MAX_PITCH = 108
MIN_NOTE_DENSITY = 0
MAX_NOTE_DENSITY = 13
MIN_TEMPO = 57
MAX_TEMPO = 258
MIN_VELOCITY = 0
MAX_VELOCITY = 126

# ============== ENCODING, DECODING AND EXTRACTING FUNCTIONS =========== #

def magenta_encode_midi(midi_filename, is_eos=False):
    mpe = MidiPerformanceEncoder(
            steps_per_second=STEPS_PER_SECOND,
            num_velocity_bins=NUM_VELOCITY_BINS,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
            add_eos=is_eos,
            is_ctrl_changes=True)

    ns = magenta.music.midi_file_to_sequence_proto(midi_filename)
    return mpe.encode_note_sequence(ns)


def magenta_decode_midi(notes, is_eos=False):
    mpe = MidiPerformanceEncoder(
            steps_per_second=STEPS_PER_SECOND,
            num_velocity_bins=NUM_VELOCITY_BINS,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
            add_eos=is_eos,
            is_ctrl_changes=True)

    pm = mpe.decode(notes, return_pm=True)
    return pm


def magenta_encode_melody(ns):
    '''
    Note sequence need not be quantized.
    '''
    mpe = TextMelodyEncoder(
            steps_per_quarter=12,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH)

    return mpe.encode_note_sequence(ns)


def extract_melodies(quantized_sequence,
                     search_start_step=0,
                     min_bars=7,
                     max_steps_truncate=None,
                     max_steps_discard=None,
                     gap_bars=1.0,
                     min_unique_pitches=5,
                     ignore_polyphonic_notes=True,
                     pad_end=False,
                     filter_drums=True):
  """Extracts a list of melodies from the given quantized NoteSequence.
  This function will search through `quantized_sequence` for monophonic
  melodies in every track at every time step.
  Once a note-on event in a track is encountered, a melody begins.
  Gaps of silence in each track will be splitting points that divide the
  track into separate melodies. The minimum size of these gaps are given
  in `gap_bars`. The size of a bar (measure) of music in time steps is
  computed from the time signature stored in `quantized_sequence`.
  The melody is then checked for validity. The melody is only used if it is
  at least `min_bars` bars long, and has at least `min_unique_pitches` unique
  notes (preventing melodies that only repeat a few notes, such as those found
  in some accompaniment tracks, from being used).
  After scanning each instrument track in the quantized sequence, a list of all
  extracted Melody objects is returned.
  Args:
    quantized_sequence: A quantized NoteSequence.
    search_start_step: Start searching for a melody at this time step. Assumed
        to be the first step of a bar.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    max_steps_truncate: Maximum number of steps in extracted melodies. If
        defined, longer melodies are truncated to this threshold. If pad_end is
        also True, melodies will be truncated to the end of the last bar below
        this threshold.
    max_steps_discard: Maximum number of steps in extracted melodies. If
        defined, longer melodies are discarded.
    gap_bars: A melody comes to an end when this number of bars (measures) of
        silence is encountered.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.
    ignore_polyphonic_notes: If True, melodies will be extracted from
        `quantized_sequence` tracks that contain polyphony (notes start at
        the same time). If False, tracks with polyphony will be ignored.
    pad_end: If True, the end of the melody will be padded with NO_EVENTs so
        that it will end at a bar boundary.
    filter_drums: If True, notes for which `is_drum` is True will be ignored.
  Returns:
    melodies: A python list of Melody instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  Raises:
    NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  # TODO(danabo): Convert `ignore_polyphonic_notes` into a float which controls
  # the degree of polyphony that is acceptable.
  melodies = []
  # pylint: disable=g-complex-comprehension
  stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
               ['polyphonic_tracks_discarded',
                'melodies_discarded_too_short',
                'melodies_discarded_too_few_pitches',
                'melodies_discarded_too_long',
                'melodies_truncated'])
  # pylint: enable=g-complex-comprehension
  # Create a histogram measuring melody lengths (in bars not steps).
  # Capture melodies that are very small, in the range of the filter lower
  # bound `min_bars`, and large. The bucket intervals grow approximately
  # exponentially.
  stats['melody_lengths_in_bars'] = statistics.Histogram(
      'melody_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, min_bars // 2, min_bars,
       min_bars + 1, min_bars - 1])
  instruments = set(n.instrument for n in quantized_sequence.notes)
  steps_per_bar = int(
      sequences_lib.steps_per_bar_in_quantized_sequence(quantized_sequence))
  for instrument in instruments:
    instrument_search_start_step = search_start_step
    # Quantize the track into a Melody object.
    # If any notes start at the same time, only one is kept.
    while 1:
      melody = Melody()
      try:
        melody.from_quantized_sequence(
            quantized_sequence,
            instrument=instrument,
            search_start_step=instrument_search_start_step,
            gap_bars=gap_bars,
            ignore_polyphonic_notes=ignore_polyphonic_notes,
            pad_end=pad_end,
            filter_drums=filter_drums)
      except PolyphonicMelodyError:
        stats['polyphonic_tracks_discarded'].increment()
        break  # Look for monophonic melodies in other tracks.
      # Start search for next melody on next bar boundary (inclusive).
      instrument_search_start_step = (
          melody.end_step +
          (search_start_step - melody.end_step) % steps_per_bar)
      if not melody:
        break

      # Require a certain melody length.
      if len(melody) < melody.steps_per_bar * min_bars:
        stats['melodies_discarded_too_short'].increment()
        continue

      # Discard melodies that are too long.
      if max_steps_discard is not None and len(melody) > max_steps_discard:
        stats['melodies_discarded_too_long'].increment()
        continue

      # Truncate melodies that are too long.
      if max_steps_truncate is not None and len(melody) > max_steps_truncate:
        truncated_length = max_steps_truncate
        if pad_end:
          truncated_length -= max_steps_truncate % melody.steps_per_bar
        melody.set_length(truncated_length)
        stats['melodies_truncated'].increment()

      # Require a certain number of unique pitches.
      note_histogram = melody.get_note_histogram()
      unique_pitches = np.count_nonzero(note_histogram)
      if unique_pitches < min_unique_pitches:
        stats['melodies_discarded_too_few_pitches'].increment()
        continue

      # TODO(danabo)
      # Add filter for rhythmic diversity.

      stats['melody_lengths_in_bars'].increment(
          len(melody) // melody.steps_per_bar)

      melodies.append(melody)

  return melodies, list(stats.values())


def encode_performance_and_melody(filename):
    # encode performance tokens
    performance_tokens = magenta_encode_midi(filename)

    # encode melody tokens
    ns = magenta.music.midi_file_to_sequence_proto(filename)
    melody_tokens = magenta_encode_melody(ns)

    return performance_tokens, melody_tokens


# ============== DATALOADERS =========== #
class MaestroDataset(Dataset):
    def __init__(self, filenames, mode="train"):
        super().__init__()

        if mode == "train":
            self.data = np.load("data/performance_tokens_augmented_train_v2.npy", allow_pickle=True)
        elif mode == "val":
            self.data = np.load("data/performance_tokens_augmented_val_v2.npy", allow_pickle=True)[:200]
        elif mode == "test":
            self.data = np.load("data/performance_tokens_augmented_test_v2.npy", allow_pickle=True)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return torch.Tensor(data)


def get_paths():
    years = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
    directory = ["/data/maestro-v2.0.0/{}/".format(k) for k in years]
    
    files = []
    for path in directory:
        files += [path + k for k in os.listdir(path)]
    return files