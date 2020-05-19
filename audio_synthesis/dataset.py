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
            # step_begin = 0
            # print(f'step_begin = {step_begin}')
            
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
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        # application to this work
        # get melspectrogram as audio representation


        # for onset piano roll, change all onsets to 1
        result['onset'][result['onset'] > 0] = 1

        # get emotion dict
        if "emotion_dict" in data:
            result['emotion_dict'] = data['emotion_dict']
            return result['audio'], result['onset'], result['emotion_dict']['res']['emotion_class']
        else:
            return result['audio'], result['onset']

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

        new_maestro_dataset, new_emotion_dataset = self._create_maestro_and_emotion_dataset()

        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        
        if os.path.exists(saved_data_path) and self.refresh==False: 
            # Check if .pt files exist, if so just load the files
            saved_results = torch.load(saved_data_path)
            maestro_entry = new_maestro_dataset[audio_path.replace("/data/MAESTRO/", "")]
            q_composer, q_title = maestro_entry["canonical_composer"], maestro_entry["canonical_title"]
            if q_composer + " " + q_title in new_emotion_dataset:
                saved_results["emotion_dict"] = new_emotion_dataset[q_composer + " " + q_title]
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
    
    def _create_maestro_and_emotion_dataset(self):
        # create maestro dataset with audio filename as key
        with open("/data/MAESTRO/maestro-v2.0.0.json", "r+") as f:
            maestro_dataset = json.load(f)
        new_maestro_dataset = {}
        for m in maestro_dataset:
            new_maestro_dataset[m["audio_filename"]] = m

        # create emotion annotation dataset with q_composer + q_title as key
        if os.path.exists("final_yamaha_emotion.json"):
            with open("final_yamaha_emotion.json", "r+") as f:
                new_emotion_dataset = json.load(f)
        
        else:
            with open("filtered_yamaha_emotion.json", "r+") as f:
                emotion_dataset = json.load(f)
            new_emotion_dataset = {}

            emotions = []

            for e in emotion_dataset:
                # set classes: split energy and valence into 4 clusters
                if e["res"] > 0.25 and e["res"]["valence"] > 0.25:
                    e["res"]["emotion_class"] = 0
                elif e["res"]["energy"] > 0.25 and e["res"]["valence"] <= 0.25:
                    e["res"]["emotion_class"] = 1
                elif e["res"]["energy"] <= 0.25 and e["res"]["valence"] <= 0.25:
                    e["res"]["emotion_class"] = 2
                elif e["res"]["energy"] <= 0.25 and e["res"]["valence"] > 0.25:
                    e["res"]["emotion_class"] = 3

                new_emotion_dataset[e["q_composer"] + " " + e["q_title"]] = e
            
            with open("final_yamaha_emotion.json", "w+") as f:
                json.dump(new_emotion_dataset, f)
        
        # prepare the list of songs that have emotion annotations with it
        with open("emotion_data_v2.txt", "w+") as f:
            for audio_filename in new_maestro_dataset.keys():
                if new_maestro_dataset[audio_filename]["canonical_composer"] + " " \
                    + new_maestro_dataset[audio_filename]["canonical_title"] in new_emotion_dataset:
                    f.write("/data/MAESTRO/" + audio_filename + "\n")
        
        return new_maestro_dataset, new_emotion_dataset


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='../../MAESTRO/', groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    def available_groups(self):
        return ["train", "validation", "test", "train_emotion", "validation_emotion", "test_emotion"]

    def files(self, group):
        with open("emotion_data_v2.txt", "r+") as f:
            # can control percentage of semi-supervised here
            lines = [k.replace("\n", "") for k in f.readlines()]
        
        if "emotion" in group:
            result = [(line.replace("pt", "wav"), None) for line in lines]
            random.Random(777).shuffle(result)

            if group == "train_emotion":
                result = result[:int(0.8*len(result))]
            elif group == "validation_emotion":
                result = result[int(0.8*len(result)):int(0.9*len(result))]
            elif group == "test_emotion":
                result = result[int(0.9*len(result)):]

            return result

        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v2.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename']),
                                os.path.join(self.path, row['midi_filename'])) for row in metadata 
                                if row['split'] == group])
            
            # for labelled data, put them in the emotion group
            new_files = []
            for audio, midi in files:
                if audio in lines:
                    pass
                else:
                    new_files.append((audio, midi))

            files = new_files
            random.Random(777).shuffle(files)

            result = []
            for audio_path, midi_path in files:
                # if audio_path.replace(".wav", ".pt") in lines:
                #     continue
                tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
                if not os.path.exists(tsv_filename):
                    midi = parse_midi(midi_path)
                    print("Writing to {}".format(tsv_filename))
                    np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
                result.append((audio_path, tsv_filename))
        
        return result
