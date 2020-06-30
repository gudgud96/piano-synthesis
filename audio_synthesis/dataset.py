# Adapting Raven Cheuk's MAESTRO dataloader for transcription tasks.
# Modified with emotion labels reading.

import json
import os
from abc import abstractmethod
from glob import glob
import sys


import numpy as np
import soundfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import pretty_midi
import random
from utils import parse_midi
import time

DEFAULT_DEVICE = 0
SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        self.data = []

        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group): #self.files is defined in MAPS class
                # torch.save(input_files, 'input_files')
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first
    
    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()

        # prepare velocity label (related to dynamics)
        velocity_label = np.sum(result['velocity'].cpu().numpy(), axis=-1)
        velocity_counter = result['velocity'].cpu().numpy().copy()
        velocity_counter[velocity_counter > 0] = 1
        velocity_counter = np.sum(velocity_counter, axis=-1) + 1e-12

        velocity_label_average = (velocity_label / velocity_counter) / 127
        velocity_label_average[velocity_label_average > 0.55] = 1 
        velocity_label_average[velocity_label_average <= 0.55] = 0

        # smoothing for velocity label
        velocity_label_average_new = velocity_label_average.copy()
        for i in range(len(velocity_label_average_new)):
            if i > 0 and i < len(velocity_label_average_new) - 1:
                if velocity_label_average[i - 1] == 0 and velocity_label_average[i + 1] == 0:
                    velocity_label_average_new[i] = 0
                if velocity_label_average[i - 1] == 1 and velocity_label_average[i + 1] == 1:
                    velocity_label_average_new[i] = 1
        
        # for onset piano roll, change all onsets to 1
        result['onset'][result['onset'] > 0] = 1

        # prepare articulation label
        frame_label = np.sum(result['frame'].cpu().numpy(), axis=-1)
        onset_label = np.sum(result['onset'].cpu().numpy(), axis=-1)
        frame_label_2 = frame_label - onset_label   # do not include onset label
        frame_label[frame_label > 0] = 1
        frame_label_2[frame_label_2 > 0] = 1

        # smoothing for articulation label
        frame_label_new = frame_label_2.copy()
        for i in range(len(frame_label_new)):
            if i > 0 and i < len(frame_label_new) - 1:
                if frame_label_2[i - 1] == 0 and frame_label_2[i + 1] == 0:
                    frame_label_new[i] = 0
                if frame_label_2[i - 1] == 1 and frame_label_2[i + 1] == 1:
                    frame_label_new[i] = 1
        
        duration, velocity = torch.Tensor(frame_label_new), torch.Tensor(velocity_label_average_new)
        return result['audio'], result['onset'], result['frame'], (duration, velocity)

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """

        new_maestro_dataset = self._create_maestro_dataset()

        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        
        if os.path.exists(saved_data_path) and self.refresh==False: 
            # Check if .pt files exist, if so just load the files
            saved_results = torch.load(saved_data_path)
            maestro_entry = new_maestro_dataset[audio_path.replace("/data/MAESTRO/", "")]

            return saved_results
        
        # Otherwise, create the .pt files
        else:
            audio, sr = soundfile.read(audio_path.replace(".wav", ".flac"), dtype='int16')
            assert sr == SAMPLE_RATE

            audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
            audio_length = len(audio)

            n_keys = MAX_MIDI - MIN_MIDI + 1
            n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps

            label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
            velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

            tsv_path = tsv_path
            midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

            for onset, offset, note, vel in midi:
                left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
                onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
                frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
                offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                f = int(note) - MIN_MIDI
                label[left:onset_right, f] = 3
                label[onset_right:frame_right, f] = 2
                label[frame_right:offset_right, f] = 1
                velocity[left:frame_right, f] = vel

            data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
            print("Writing to {}".format(saved_data_path))
            torch.save(data, saved_data_path)
            return data
    
    def _create_maestro_dataset(self):
        # create maestro dataset with audio filename as key
        with open("/data/MAESTRO/maestro-v2.0.0.json", "r+") as f:
            maestro_dataset = json.load(f)
        new_maestro_dataset = {}
        for m in maestro_dataset:
            new_maestro_dataset[m["audio_filename"]] = m
        
        return new_maestro_dataset


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='../../MAESTRO/', groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):

        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    def available_groups(self):
        return ["train", "validation", "test", "train_emotion", "validation_emotion", "test_emotion",
                "train_all", "validation_all", "test_all"]

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, 'maestro-v2.0.0.json')))

        def get_split(group):
            if "train" in group:
                return "train"
            elif "validation" in group:
                return "validation"
            elif "test" in group:
                return "test"

        files = sorted([(os.path.join(self.path, row['audio_filename']),
                            os.path.join(self.path, row['midi_filename'])) for row in metadata 
                            if row['split'] == get_split(group)])
        
        random.Random(777).shuffle(files)
        
        result = []

        for audio_path, midi_path in files:

            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                print("Writing to {}".format(tsv_filename))
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))

        return result
