import pathlib
import time
import h5py


class HitlRecorder:
    def __init__(self,
        output_dir: str,
        cam_names: list,
        max_timesteps: int,
    ):
        """
        For each timestep:
        observations
        - images
            - cam_high          (120, 160, 3) 'uint8'
            - cam_low           (120, 160, 3) 'uint8'
            - cam_left_wrist    (120, 160, 3) 'uint8'
            - cam_right_wrist   (120, 160, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'
        action                  (14,)         'float32'
        mode                    ()            'int32'
        """
        self.output_dir = output_dir
        self.episode_idx = self.get_auto_index()
        self.data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
            '/mode': [],
        }
        for cam_name in cam_names:
            self.data_dict[f'/observations/images/{cam_name}'] = []

        self.cam_names = cam_names
        self.max_timesteps = max_timesteps

    def get_auto_index(self, name_suffix='hdf5'):
        max_idx = 1000
        for i in range(max_idx):
            filename = f'episode_{i}.{name_suffix}'
            if not pathlib.Path(self.output_dir, filename).exists():
                return i
        raise ValueError(f'No available index in {self.output_dir}.')

    def store_one_step(self, ts, action, mode):
        data_dict = self.data_dict
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        for cam_name in self.cam_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        data_dict['/action'].append(action)
        data_dict['/mode'].append(mode)

    def store_episode(self):
        filename = f'episode_{self.episode_idx}.hdf5'
        dataset_path = pathlib.Path(self.output_dir, filename)

        # scan mode and turn
        # auto: 1/pause: 2/human: 3
        # into 
        # auto: 1/pre: 2/human: 3/demo: 0
        mode = self.data_dict['/mode']
        assert mode[0] != 2 and mode[-1] != 2, f"Pause at the beginning or end of the episode is not allowed."
        for i in range(len(mode)):
            if mode[i] == 2:
                if mode[i-1] == 1 and mode[i+1] == 3:
                    mode[i] = 3
                elif mode[i-1] == 3 and mode[i+1] == 1:
                    mode[i] = 1
                elif mode[i-1] == 1 and mode[i+1] == 1:
                    mode[i] = 1
                else:
                    mode[i] = 3
        for m in mode:
            assert m in [1, 3], f"Invalid mode: {m}."

        for i in range(1, len(mode)):
            # mark 1 sec (50 steps) before human mode as pre mode
            if mode[i] == 3 and mode[i-1] == 1:
                for j in range(50):
                    idx = max(0, i-j-1)
                    mode[idx] = 2 if mode[idx] == 1 else mode[idx]

        t0 = time.time()
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.cam_names:
                _ = image.create_dataset(
                    cam_name,
                    (self.max_timesteps, 120, 160, 3),
                    dtype='uint8',
                    chunks=(1, 120, 160, 3)
                )
            _ = obs.create_dataset('qpos', (self.max_timesteps, 14))
            _ = obs.create_dataset('qvel', (self.max_timesteps, 14))
            _ = obs.create_dataset('effort', (self.max_timesteps, 14))
            _ = root.create_dataset('action', (self.max_timesteps, 14), dtype='float32')
            _ = root.create_dataset('mode', (self.max_timesteps,), dtype='int32')

            for name, array in self.data_dict.items():
                root[name][...] = array

        print(f'Saving: {time.time() - t0:.1f} secs @ {dataset_path}')