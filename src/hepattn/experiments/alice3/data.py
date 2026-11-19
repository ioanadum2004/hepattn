from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset


class Alice3Dataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        hit_coord_scale: float = 0.001,  # CHECK against your x/y/z ranges
        particle_min_pt: float = 0.3,
        particle_max_pt: float | None = None,  # set for a pT band
        particle_max_abs_eta: float = 4.0,
        particle_min_num_hits: int = 3,
        event_max_num_particles: int = 2500,
        strict_max_objects: bool = False,
    ):
        super().__init__()

        event_names = sorted(Path(file).stem.replace("-parts", "") for file in Path(dirpath).glob("event*-parts.parquet"))
        num_available = len(event_names)

        if num_events > num_available:
            raise ValueError(f"Requested {num_events} events, only {num_available} in {dirpath}")
        if num_events < 0:
            num_events = num_available
        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]
        self.sample_ids = [int(name.removeprefix("event")) for name in self.event_names]

        self.hit_volume_ids = hit_volume_ids
        self.hit_coord_scale = hit_coord_scale
        self.particle_min_pt = particle_min_pt
        self.particle_max_pt = particle_max_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_hits = particle_min_num_hits
        self.event_max_num_particles = event_max_num_particles
        self.strict_max_objects = strict_max_objects

    def __len__(self):
        return int(self.num_events)

    def __getitem__(self, idx):
        inputs = {}
        targets = {}

        hits, particles = self.load_event(idx)
        num_particles = len(particles)

        # Hit inputs
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((len(hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field].values).unsqueeze(0).half()

        # Particle slot padding
        if num_particles > self.event_max_num_particles:
            if self.strict_max_objects:
                raise ValueError(f"Event {idx} has {num_particles} particles, limit is {self.event_max_num_particles}")
            particles = particles.iloc[: self.event_max_num_particles]
            num_particles = self.event_max_num_particles

        num_padding = self.event_max_num_particles - num_particles
        targets["particle_valid"] = torch.cat([torch.full((num_particles,), True), torch.full((num_padding,), False)]).unsqueeze(0)

        # Particle-hit association
        selected_particle_ids = torch.from_numpy(particles["particle_id"].values)
        particle_ids = torch.cat([selected_particle_ids, torch.full((num_padding,), -999)])
        hit_particle_ids = torch.from_numpy(hits["particle_id"].values)
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)
        targets["hit_particle_id"] = hit_particle_ids.unsqueeze(0)
        targets["particle_id"] = particle_ids.unsqueeze(0)

        # Hit filter target (ignores event_max_num_particles truncation)
        for target_feature, fields in self.targets.items():
            if "on_valid_particle" in fields:
                targets[f"{target_feature}_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"].to_numpy()).unsqueeze(0)

        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int32)

        # Particle regression targets (null slots are NaN)
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                x = torch.full((self.event_max_num_particles,), torch.nan)
                x[:num_particles] = torch.from_numpy(particles[field].to_numpy()[: self.event_max_num_particles])
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

    def load_event(self, idx):
        name = self.event_names[idx]
        particles = pd.read_parquet(self.dirpath / f"{name}-parts.parquet")
        hits = pd.read_parquet(self.dirpath / f"{name}-hits.parquet")

        if self.hit_volume_ids:
            hits = hits[hits["volume_id"].isin(self.hit_volume_ids)].copy()

        # Scale coordinates so values are ~1
        for coord in ["x", "y", "z"]:
            hits[coord] = hits[coord] * self.hit_coord_scale

        # Derived hit features (guard against hits at the origin)
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2).clip(lower=1e-9)
        hits["theta"] = np.arccos(hits["z"] / hits["s"])
        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits["eta"] = -np.log(np.tan(hits["theta"] / 2))

        # Derived particle fields
        particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])

        # Particle cuts: this defines what the filter KEEPS (label True)
        particles = particles[particles["pt"] > self.particle_min_pt]
        if self.particle_max_pt is not None:
            particles = particles[particles["pt"] < self.particle_max_pt]
        particles = particles[particles["eta"].abs() < self.particle_max_abs_eta]

        # Cut on number of hits per particle
        counts = hits["particle_id"].value_counts()
        keep_ids = counts[counts >= self.particle_min_num_hits].index.to_numpy()
        particles = particles[particles["particle_id"].isin(keep_ids)]

        # Hit label: noise (particle_id = -1) is automatically False
        hits["on_valid_particle"] = hits["particle_id"].isin(particles["particle_id"])

        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(hits) != 0, "No hits remaining - loosen selection!"
        assert particles["particle_id"].nunique() == len(particles), "Non-unique particle ids"

        return hits, particles


class Alice3DataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dataset = Alice3Dataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs)
        if stage == "fit":
            self.val_dataset = Alice3Dataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs)
            rank_zero_info(f"Training events: {len(self.train_dataset):,}, validation events: {len(self.val_dataset):,}")
        if stage == "test":
            assert self.test_dir is not None, "No test dir specified, see --data.test_dir"
            self.test_dataset = Alice3Dataset(dirpath=self.test_dir, num_events=self.num_test, **self.kwargs)

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset=dataset, batch_size=None, collate_fn=None, sampler=None,
                          num_workers=self.num_workers, shuffle=shuffle, pin_memory=self.pin_memory)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
