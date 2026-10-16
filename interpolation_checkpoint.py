"""Durable, per-catalog progress for the interpolation driver."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


class InterpolationCheckpoint:
    """Record updates only after the corresponding HDF5 file has closed.

    A lock prevents two jobs from using this checkpoint simultaneously. Jobs
    using different checkpoints must still use disjoint galaxy catalogs.
    """

    def __init__(self, path, metadata, start_plane=1):
        self.path = Path(path).resolve()
        self.metadata = metadata
        self.requested_start = start_plane
        self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(str(self.path) + '.lock', 'a')
        try:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Checkpoint is already in use: {self.path}') from exc
            if self.path.exists():
                with self.path.open() as source:
                    self.data = json.load(source)
                if self.data.get('version') != 1 or self.data.get('metadata') != self.metadata:
                    raise ValueError(
                        f'Checkpoint configuration/code mismatch: {self.path}. '
                        'Use a new checkpoint path for an intentional new run.'
                    )
                if not isinstance(self.data.get('completed'), dict):
                    raise ValueError(f'Invalid checkpoint: {self.path}')
                print(f"[restart] Loaded {len(self.data['completed'])} completed catalog intervals from {self.path}", flush=True)
            else:
                self.data = dict(version=1, metadata=self.metadata,
                                 start_plane=self.requested_start, completed={})
                self._save()
                print(f'[restart] Created {self.path}', flush=True)
            if self.requested_start > self.data['start_plane']:
                self.data['start_plane'] = self.requested_start
                self._save()
            self.start_plane = self.data['start_plane']
            if self.start_plane > 1:
                print(f'[restart] Starting no earlier than logged plane {self.start_plane}; earlier planes are trusted from the previous run.', flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def _save(self):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                             prefix=self.path.name + '.', delete=False) as target:
                temp_path = target.name
                json.dump(self.data, target, sort_keys=True)
                target.write('\n')
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _key(filename, step_idx, first_plane):
        return json.dumps([int(step_idx), bool(first_plane), os.path.realpath(filename)])

    def is_complete(self, filename, step_idx, first_plane=False):
        return self._key(filename, step_idx, first_plane) in self.data['completed']

    def pending_files(self, filenames, step_idx, first_plane=False):
        return [f for f in filenames if not self.is_complete(f, step_idx, first_plane)]

    def mark_complete(self, filename, step_idx, first_plane, selected):
        self.data['completed'][self._key(filename, step_idx, first_plane)] = int(selected)
        self._save()


def checkpoint_metadata(sim, path_gals, path_end, patches, mag_cols, flux_cols, steps):
    """Bind progress to inputs, output convention, configuration, and code."""
    root = Path(__file__).resolve().parent
    names = ('run_interpolation_roman.py', 'diffsky_routines.py',
             'interpolation_routines.py', 'interpolation_checkpoint.py',
             'hacc_sims.py', 'simulation.py', 'utils.py', 'write_funcs.py',
             'angle_updates.py', 'transport_xyz.py')
    configuration = {
        name: sim[name] for name in
        ('nside', 'nplanes', 'nsteps', 'n_steps_cmb', 'zinit', 'zfin',
         'h', 'Omega_m', 'step_list_min', 'step_list_max')
    }
    # Normalize NumPy scalar/list values without importing NumPy here.
    configuration = json.loads(json.dumps(configuration, default=lambda value: value.tolist()))
    return dict(
        galaxy_directory=os.path.realpath(path_gals),
        ray_directory=os.path.realpath(sim['output_path_rt']),
        file_suffix=path_end, patches=sorted(set(patches)),
        magnitude_columns=list(mag_cols), flux_columns=list(flux_cols),
        galaxy_steps=[str(step) for step in steps], simulation=configuration,
        code_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    )


def default_checkpoint_path(path_gals, path_end, patches):
    identity = json.dumps([os.path.realpath(path_gals), path_end, sorted(set(patches))])
    token = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return str(Path.cwd() / f'interpolation_restart_{token}.json')
