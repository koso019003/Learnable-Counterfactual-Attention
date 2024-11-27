import torch
import random
import lightning
import numpy as np
import torchaudio.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor


class GTZANDataModule(lightning.LightningDataModule):
    def __init__(self, pre_trained_name, data_config, batch_size: int = 16):
        super().__init__()
        self.pre_trained_name = pre_trained_name
        self.batch_size = batch_size
        self.id2class = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
        self.nb_classes = data_config['nb_classes']
        self.slice_length = data_config['slice_length']
        self.seconds_overlap = data_config['seconds_overlap']
        self.save_hyperparameters()
        
        self.train_set = None
        self.valid_set = None
        self.test_set = None
    
    def ids2class(self, ids):
        return [self.id2class[int(id)] for id in ids]
    
    def setup(self, stage: str):
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.pre_trained_name, trust_remote_code=True
        )
        # GTZAN audio default sampling rate
        sampling_rate = 22050
        resample_rate = processor.sampling_rate

        # Make sure the sample_rate aligned
        print(f"Setting sample rate from {sampling_rate} to {resample_rate}")
        
        resampler = T.Resample(sampling_rate, resample_rate)

        # Load data
        if stage == "train":
            x_train, y_train, s_train, x_test, y_test, s_test, x_val, y_val, s_val = load_data(resampler, datasplit='songs')
            x_train, y_train, s_train = slice_songs(
                x_train,
                y_train,
                s_train,
                sample_rate=resample_rate,
                length=self.slice_length,
                overlap=self.seconds_overlap,
            )
            x_val, y_val, s_val = slice_songs(
                x_val,
                y_val,
                s_val,
                sample_rate=resample_rate,
                length=self.slice_length,
                overlap=self.seconds_overlap,
            )
            x_train = processor(
                x_train, padding=True, sampling_rate=resample_rate, return_tensors="pt"
            )
            x_val = processor(
                x_val, padding=True, sampling_rate=resample_rate, return_tensors="pt"
            )
            print("Training set label counts:", np.unique(y_train, return_counts=True))

            self.train_set = MusicDataset(x_train, y_train, s_train)
            self.valid_set = MusicDataset(x_val, y_val, s_val)
        else:
            _, _, _, x_test, y_test, s_test, _, _, _ = load_data(resampler, datasplit='songs', stage="test")

        # Create slices out of the songs
        x_test, y_test, s_test = slice_songs(
            x_test,
            y_test,
            s_test,
            sample_rate=resample_rate,
            length=self.slice_length,
            overlap=self.seconds_overlap,
        )

        x_test = processor(
            x_test, padding=True, sampling_rate=resample_rate, return_tensors="pt"
        )
        
        self.y_test_original = y_test
        self.test_set = MusicDataset(x_test, y_test, s_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=2)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=2)


class MusicDataset(Dataset):
    def __init__(self, x, y, s):
        self.x = x
        self.y = torch.Tensor(y).to(torch.long)
        self.s = np.asarray(s)
        self.size = len(self.y)

    def __getitem__(self, index):
        return (
            {
                "input_values": self.x.input_values[index],
                "attention_mask": self.x.attention_mask[index],
            },
            self.y[index],
        )

    def __len__(self):
        return self.size


def load_data(resampler, datasplit='songs', stage='train'):
    dataset = load_dataset("marsyas/gtzan", trust_remote_code=True)["train"]
    x_train, x_test, x_val = [], [], []
    y_train, y_test, y_val = [], [], []
    s_train, s_test, s_val = [], [], []
    
    if datasplit == 'random':
        x, y, s = [], [], []
        tmp_x, tmp_y, tmp_s = [], [], []
        last_genre = 0
        for data in dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(torch.float32)
            if data["genre"] != last_genre:
                x.append(tmp_x)
                y.append(tmp_y)
                s.append(tmp_s)
                tmp_x, tmp_y, tmp_s = [], [], []
                last_genre = data["genre"]
            tmp_x.append(resampler(audio).numpy())
            tmp_y.append(data["genre"])
            tmp_s.append(data["file"].split("/")[-1])
        x.append(tmp_x)
        y.append(tmp_y)
        s.append(tmp_s)

        for genre in range(len(x)):
            random_ids = list(range(len(x[genre])))
            random.shuffle(random_ids)

            tmp_x = [x[genre][ids] for ids in random_ids]
            tmp_y = [y[genre][ids] for ids in random_ids]
            tmp_s = [s[genre][ids] for ids in random_ids]

            # train, test, val = 50, 25, 25
            x_train.extend(tmp_x[:50])
            x_test.extend(tmp_x[50:75])
            x_val.extend(tmp_x[75:])

            y_train.extend(tmp_y[:50])
            y_test.extend(tmp_y[50:75])
            y_val.extend(tmp_y[75:])

            s_train.extend(tmp_s[:50])
            s_test.extend(tmp_s[50:75])
            s_val.extend(tmp_s[75:])
    elif datasplit =='songs' or datasplit == 'filtered':
        with open(f"../dataset/test_{datasplit}.txt", "r") as f:
            test_songs = f.readlines()
        test_songs = [line.strip() for line in test_songs]

        train_songs = []
        val_songs = []
        if stage == "train":
            with open(f"../dataset/train_{datasplit}.txt", "r") as f:
                train_songs = f.readlines()
            with open(f"../dataset/val_{datasplit}.txt", "r") as f:
                val_songs = f.readlines()
            train_songs = [line.strip() for line in train_songs]
            val_songs = [line.strip() for line in val_songs]
            
        for data in dataset:
            genre = data["genre"]
            song = data["file"].split("/")[-1]
            if song in train_songs:
                audio = torch.from_numpy(data["audio"]["array"]).to(torch.float32)
                x_train.append(resampler(audio).numpy())
                y_train.append(genre)
                s_train.append(song)
            elif song in val_songs:
                audio = torch.from_numpy(data["audio"]["array"]).to(torch.float32)
                x_val.append(resampler(audio).numpy())
                y_val.append(genre)
                s_val.append(song)
            elif song in test_songs:
                audio = torch.from_numpy(data["audio"]["array"]).to(torch.float32)
                x_test.append(resampler(audio).numpy())
                y_test.append(genre)
                s_test.append(song)
            # else:
            #     print(song)
            #     raise ValueError
    else:
        raise ValueError(f'dtasplit must be "random" or "songs" or "filtered" but got {datasplit}')
    return x_train, y_train, s_train, x_test, y_test, s_test, x_val, y_val, s_val


def slice_songs(x, y, s, sample_rate=22050, length=10, overlap=0):
    """Slices the wav into slice according to length (seconds)"""

    length = sample_rate * length
    overlap = sample_rate * overlap

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    shift = length - overlap

    for i in range(len(x)):
        slices = (x[i].shape[0] - overlap) // shift
        for j in range(slices):
            spectrogram.append(x[i][j * shift : j * shift + length])
            artist.append(y[i])
            song_name.append(s[i])

    return spectrogram, artist, song_name
