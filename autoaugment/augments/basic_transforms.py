import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
from scipy.spatial.transform import Rotation as R


class AugmentTransform(nn.Module, ABC):
    def __init__(self, cond_dim=8, n_channels=1, seq_len=100, device=None):
        super().__init__()
        self.n_features = cond_dim
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.device = device
        self.__initialize__()

    def __initialize__(self):
        # reserved for child classes
        pass

    def forward(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, n_channels, 1) or (batch, n_channels)
        :return: augmented batch_x, batch_y, batch_mask
        """
        try:
            batch, n_channels, seq_len = batch_x.shape
            if len(strength.shape)<3:
                strength = strength.reshape((batch, n_channels, 1))
            aug_x, aug_y, aug_mask = self.augment(batch_x, batch_y, batch_f, batch_mask, strength)
            return aug_x, aug_y, aug_mask
        except Exception as e:
            print(f"[{self.__class__.__name__}] transform failed")
            raise e

    @abstractmethod
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, n_channels, 1)
        :return: augmented batch_x, batch_y, batch_mask
        """
        raise NotImplementedError("Not implemented augmentation method.")

class Raw(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        return batch_x, batch_y, batch_mask

class Jitter(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, n_channels, 1)
        :return: augmented batch_x, batch_y, batch_mask
        """
        eps = torch.randn_like(batch_x, device=batch_x.device)
        # print(f"Jitter_strength: {strength}")
        noise = eps * strength
        return batch_x + noise, batch_y, batch_mask

class Scale(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, n_channels, 1) or (batch, n_channels)
        :return: augmented batch_x, batch_y, batch_mask
        """
        # print(f"pre_tensor:{batch_x.shape}")
        scale_factor = (1 + torch.rand_like(batch_x, device=batch_x.device) * (strength * 2-1))
        # print(f"ope_tensor:{(batch_x * scale_factor).shape}")
        return batch_x * scale_factor, batch_y, batch_mask

class MagnitudeWarp(AugmentTransform):

    def __initialize__(self):
        self.warp_t = torch.linspace(0, 2 * torch.pi, self.seq_len, device=self.device)

    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        batch, n_channels, seq_len = batch_x.shape  # batch_x: (B, C, T)
        batch_x = batch_x.reshape(-1, seq_len)     # (B*C, T)
        if strength.shape[1] == 1 and n_channels > 1:
            strength = strength.repeat(1, n_channels, 1)
        strength = strength.reshape(-1, 1)         # (B*C, 1)

        warp_frq = torch.randn(batch * n_channels, 1, device=batch_x.device)
        warp_phs = torch.randn(batch * n_channels, 1, device=batch_x.device)

        warp_tgt = torch.sin(warp_frq * self.warp_t.unsqueeze(0) + warp_phs)  # (B*C, T)
        warp_result = batch_x + strength * (warp_tgt - batch_x)               # (B*C, T)
        warp_result = warp_result.reshape((batch, n_channels, seq_len))       # (B, C, T)

        return warp_result, batch_y, batch_mask


class FreqWarp(AugmentTransform):
    def __initialize__(self):
        self.window_size = self.seq_len
        self.sample_rate = 16000 # sampling rate
        self.n_mels = 10 # number of mel filters
        self.fmin = 0
        self.fmax = 8000
        self.Fhi = 4800  # boundary frequency
        
        # Compute the frequency axis. d is the sampling interval:
        # freq[k] = k / (N * d) = k / window_size * sample_rate.
        self.freqs = torch.fft.rfftfreq(self.window_size, d=1/self.sample_rate, device=self.device)
        self.freq_num = len(self.freqs)
        # Time-step array.
        self.t_array = torch.arange(self.window_size, device=self.device).float()

    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        batch, n_channels, seq_len = batch_x.shape

        # Reshape input
        x = batch_x.reshape(-1, seq_len) # (B*C, T)

        if strength.shape[1] == 1 and n_channels > 1:
            strength = strength.repeat(1, n_channels, 1)
        strength = strength.reshape(-1) # (B*C,)

        # Generate alpha based on strength
        # alpha = 1.0 + (torch.randn_like(strength) * 0.1 * strength)
        alpha = 1.0 + (torch.randn(x.shape[0], device=x.device) * 0.1 * strength)
        alpha = torch.clamp(alpha, min=0.9, max=1.1)

        # Warp frequency
        freqs_warped = self._warp_frequency(alpha) # (B*C, T/2+1)
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        
        # Reconstruct
        y = self._reconstruct_signal(magnitude, phase, freqs_warped) # (B*C, T)
        
        # Reshape back
        y = y.reshape(batch, n_channels, seq_len)
        
        return y, batch_y, batch_mask
    
    def _warp_frequency(self, alpha):
        freqs = self.freqs.unsqueeze(0)  # [1, T/2+1]
        alpha = alpha.unsqueeze(1) # [N*C, 1]
        alpha2 = torch.clamp(alpha, max=1.0)
        
        # Case f <= Fhi: f' = f * alpha.
        f_low = freqs * alpha
        
        # Case f > Fhi: f' = S - (S - Fhi * alpha2) / (S - Fhi * alpha2 / alpha) * (S - f).
        S = self.sample_rate / 2
        Fhi_alpha2 = self.Fhi * alpha2
        denominator = S - Fhi_alpha2 / alpha
        denominator = torch.where(denominator < 1e-8, torch.tensor(1e-8, device=alpha.device), denominator)
        f_high = S - (S - Fhi_alpha2) / denominator * (S - freqs)
        
        # Select the result by condition.
        freqs_bound = self.Fhi * alpha2 / alpha
        freqs_warped = torch.where(freqs <= freqs_bound, f_low, f_high)
        
        return freqs_warped

    def _reconstruct_signal(self, magnitude, phase, freqs):
        omega = 2 * torch.pi * freqs.unsqueeze(2) * self.t_array[None, None, :] / self.sample_rate  # [N*C, T/2+1, T]
        components = (magnitude.unsqueeze(2) * torch.cos(omega + phase.unsqueeze(2))) # [N*C, T/2+1, T]
        signal = 2*components.sum(dim=1) - components[:, 0, :] - components[:, self.freq_num-1, :]
        return signal / self.window_size


class WindowSliceWarp(AugmentTransform):

    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        batch, n_channels, seq_len = batch_x.shape
        batch_x = batch_x.reshape(-1, seq_len)
        strength = strength.reshape(-1, 1)
        batch_x = self._aug_optimized(batch_x, strength)
        batch_x = batch_x.reshape((batch, n_channels, seq_len))
        return batch_x, batch_y, batch_mask

    def _aug_optimized(self, x, strength):
        """
        :param x:  (batch, seq_len)
        :param strength:  (batch, 1)
        :return:   (batch, seq_len)
        """
        batch_size, seq_len = x.shape
        # starts_max: (batch, 1)
        starts_max = torch.floor(strength * self.seq_len / 2).long().to(x.device)
        # target_len: (batch, 1)
        target_len = self.seq_len - starts_max
        # starts:(batch, 1)
        starts = torch.floor(starts_max * torch.rand_like(strength)).long()
        ends = starts + target_len + 1

        # Create the result tensor.
        result = torch.zeros_like(x).to(x.device)
        for i in range(batch_size):
            # Slice and interpolate each sample separately.
            slice_data = x[i, starts[i]:ends[i]]
            interpolated = F.interpolate(slice_data.unsqueeze(0).unsqueeze(0),
                                         size=seq_len,
                                         mode='linear',
                                         align_corners=False).to(x.device)
            result[i] = interpolated.squeeze()
        return result

class IAAFT(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        batch, n_channels, seq_len = batch_x.shape
        batch_x = batch_x.reshape(-1, seq_len)
        strength = strength.reshape(-1, 1)
        batch_x = self._iaaft2(batch_x, strength)
        batch_x = batch_x.reshape((batch, n_channels, seq_len))
        return batch_x, batch_y, batch_mask

    def _iaaft2(self, x, s, max_iterations=20):
        """
        Iterated Amplitude Adjusted Fourier Transform with strength parameter
        :param x: input timeseries (batch, seq_len)
        :param s: strength (batch, 1)
        :param max_iterations: max iteration
        :return: transformed timeseries (batch, seq_len)
        """
        batch_size = x.shape[0]
        result = torch.zeros_like(x, device=x.device)
        for i in range(batch_size):
            x_i = x[i]
            amplitudes = torch.abs(fft(x_i))
            sort_x, _ = torch.sort(x_i)
            y = x_i[torch.randperm(x_i.size(0))]
            for _ in range(max_iterations):
                fft_y = fft(y)
                adjusted_fft = amplitudes * torch.exp(1j * torch.angle(fft_y))
                new_y = ifft(adjusted_fft).real
                y = sort_x[torch.argsort(torch.argsort(new_y))]
            result[i] = s[i] * y + (1 - s[i]) * x_i
        return result

class DRC(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, 1) or scalar
        :return: augmented batch_x, batch_y, batch_mask
        """
        threshold = strength.squeeze()
        ratio = 2.0 # The implementation may be adjusted later.
        batch_x_np = batch_x.cpu().numpy()
        compressed = np.clip(batch_x_np, -threshold, threshold)
        compressed = compressed - (1 - 1/ratio) * compressed
        compressed[batch_x_np < -threshold] = batch_x_np[batch_x_np < -threshold]
        compressed[batch_x_np > threshold] = batch_x_np[batch_x_np > threshold]

        compressed_tensor = torch.tensor(compressed, device = batch_x.device)
        return compressed_tensor, batch_y, batch_mask

# class AAFT(AugmentTransform):
#     def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
#         """
#         :param batch_x: (batch, n_channels, seq_len)
#         :param batch_y: (batch, label_len)
#         :param batch_f: (batch, n_channels, feature_len)
#         :param batch_mask: (batch, seq_len)
#         :param strength: (batch, n_channels, 1)
#         :return augmented batch_x, batch_y, batch_mask
#         """
#         batch, n_channels, seq_len = batch_x.shape
#         batch_x = batch_x.reshape(-1, seq_len)

class Perm(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, n_channels, 1) or (batch, n_channels)
        :return augmented batch_x, batch_y, batch_mask
        """
        batch, n_channels, seq_len = batch_x.shape
        batch_x = batch_x.reshape(-1, seq_len)
        strength = strength.reshape(-1, 1)
        batch_x = self._permute_segments(batch_x, strength, seq_len)
        batch_x = batch_x.reshape((batch, n_channels, seq_len))
        return batch_x, batch_y, batch_mask

    def _permute_segments(self, x, strength, seq_len):
        """
        Apply permutation to segments in the time series
        :param x: (batch, seq_len)
        :param strength: (batch, 1)
        # :param nPerm: Number of segments
        # :param minSegLength: Minimum segment length
        :param seq_len: length of the sequence
        :return Permutated time series
        """

        # get nPerm and minSeglength
        nPerm = max(2, seq_len // 20)
        minSegLength = max(1, seq_len // (2 * nPerm))
        
        batch_size = x.shape[0]
        result = torch.zeros_like(x, device = x.device)

        for i in range(batch_size):
            bWhile = True
            while bWhile:
                # generate segment breakpoints
                segs = torch.zeros(nPerm + 1, dtype = torch.long, device = x.device) # Segment breakpoints; count = segments + 1.
                segs[1:-1] = torch.sort(torch.randint(minSegLength, seq_len - minSegLength, (nPerm - 1)).to(x.device))[0]
                segs[-1] = seq_len

                # check minimum segment length
                if torch.min(seg[1:] - seg[:-1]) > minSegLength:
                    bWhile = False

        # Random permutation of segments
        perm_idx = torch.random(nPerm, device = x.device)
        start = 0
        for j in range(nPerm):
            seg_start = segs[perm_idx[j]]
            seg_end = segs[perm_idx[j] + 1]
            segment = x[i, seg_start:seg_end]
            result[i, start:start + segment.size(0)] = segment
            start += segment.size(0)

        return result

class Rotation(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, feature_len)
        :param batch_mask: (batch, seq_len)
        :param strength: (batch, n_channels, 1)
        :return: rotated batch_x, batch_y, batch_mask
        """
        batch, n_channels, seq_len = batch_x.shape
        batch_x = batch_x.clone()

        if n_channels < 3:
            raise ValueError("Rotation augmentation requires at least 3 channels for 3D rotation.")

        # select 3 channels to rotate randomly
        selected_channels = np.random.choice(n_channels, size = 3, replace = False)
        spatial_channels = batch_x[:, selected_channels, :]

        # rotation
        rotated_spatial_channels = self._apply_rotation(spatial_channels)

        batch_x[:, selected_channels, :] = rotated_spatial_channels
        
        # batch_x = batch_x.reshape(-1, n_channels, seq_len)

        # if n_channels != 3:
        #     raise ValueError("Rotation augmentation requires exactly 3 channels for 3D rotation.")

        # # Apply rotation for each instance in the batch
        # rotated_batch_x = self._apply_rotation(batch_x)
        # rotated_batch_x = rotated_batch_x.reshape(batch, n_channels, seq_len)


        
        
        return batch_x, batch_y, batch_mask

    def _apply_rotation(self, x):
        """
        Apply a random 3D rotation to each sequence in x.
        :param x: input data(batch, n_channels, seq_len)
        :return: rotated_data
        """
        batch_size, n_channels, seq_len = x.shape
        rotated_data = torch.zeros_like(x, device = x.device)

        for i in range(batch_size):
            # Generate a random rotation
            axis = np.random.uniform(low = -1, high = 1, size = (n_channels, ))
            angle = np.random.uniform(low = -np.pi, high = np.pi)
            rotation_matrix = torch.tensor(R.from_rotvec(axis * angle).as_matrix(), dtype = x.dtype, device = x.device)

        return rotated_data
        

class Downsampling(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: 
        """

        batch_size, n_channels, seq_len = batch_x.shape
        augmented_batch = []

        min_factor = 2
        max_factor = 5

        # for i in range(batch_size):
        #     # Shared sampling factor.
        #     shared_strength = strength[i, :, 0].mean().item()
        #     downsample_factor = int(min_factor + (max_factor - min_factor) * shared_strength)
        #     downsample_factor = max(1, downsample_factor) # At least 1.
        #     # print(downsample_factor)
        #     print(shared_strength)

        #     # Downsample each channel.
        #     indices = torch.arange(0, seq_len, step=downsample_factor)
        #     channel_augmented = [batch_x[i, j, indices].unsqueeze(0) for j in range(n_channels)]

        #     # Concatenate each channel's augmented result.
        #     augmented_sample = torch.cat(channel_augmented, dim=0)
        #     augmented_batch.append(augmented_sample.unsqueeze(0))
        
        for i in range(batch_size):
            # shared downsampling factor
            shared_strength = strength[i, :, 0].mean().item()
            downsample_factor = int(min_factor + (max_factor - min_factor) * shared_strength)
            downsample_factor = max(1, downsample_factor)

            # downsample each channel
            indices = torch.arange(0, seq_len, step=downsample_factor)
            channel_augmented = []

            for j in range(n_channels):
                # downsample
                downsample_series = batch_x[i, j, indices]
                # Interpolate.
                restored_series = F.interpolate(
                    downsample_series.unsqueeze(0).unsqueeze(0), # (1, 1, downsampled_len)
                    size=seq_len, # Restore to the original length.
                    mode='linear',
                    align_corners=False
                ).squeeze(0).squeeze(0) # Restore dimensions.
                channel_augmented.append(restored_series.unsqueeze(0))
            
            # Concatenate the augmented result of each channel.
            augmented_sample = torch.cat(channel_augmented, dim=0)
            augmented_batch.append(augmented_sample.unsqueeze(0))

        # Concatenate augmented results for all samples.
        augmented_batch_x = torch.cat(augmented_batch, dim=0)
        # print(f"batch_x_shape: {batch_x.shape}")
        # torch.save(batch_x, 'batch_x.pt')
        # torch.save(augmented_batch_x, 'augmented_batch_x.pt')
        # print(f"augment_batch_x_shape: {augmented_batch_x.shape}")
        
        # exit(0)
        return augmented_batch_x, batch_y, batch_mask
    
class Resampling(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len) - input time series
        :param batch_y: (batch, label_len) - label
        :param batch_f: (batch, n_channels, feature_len) - optional features
        :param batch_mask: (batch, seq_len) - sequence mask
        :param strength: (batch, n_channels, 1) - augmentation strength, unused here
        :return: augmented batch_x, batch_y, batch_mask
        """
        batch, n_channels, seq_len = batch_x.shape
        augmented_batch = []

        for i in range(batch):
            # Simulate characteristic point detection.
            characteristic_points = self._detect_characteristic_points(seq_len)
#             print(characteristic_points)
            subsequence = batch_x[i, :, characteristic_points[0]:characteristic_points[-1]]  # Extract subsequence.

            # Repeat and concatenate the subsequence.
            concatenated_sequence = torch.cat([subsequence, subsequence], dim=-1)

            # Slice with a sliding window and restore by interpolation.
            window_size = seq_len  # Sliding-window length equals the original length.
            sliced_sequences = self._slice_and_interpolate(concatenated_sequence, window_size)

            augmented_batch.append(sliced_sequences.unsqueeze(0))

        # Concatenate augmented samples.
        augmented_batch_x = torch.cat(augmented_batch, dim=0)
        
        return augmented_batch_x, batch_y, batch_mask

    def _detect_characteristic_points(self, seq_len):
        """
        :param seq_len: sequence length
        :return: characteristic point indices
        """
        num_points = int(0.2 * seq_len) 
        return np.linspace(0, seq_len - 1, num_points, dtype=int)
        
    def _slice_and_interpolate(self, sequence, window_size):
        """
        Slice with a sliding window and restore by interpolation.
        :param sequence: (n_channels, seq_len) concatenated sequence
        :param window_size: target length for restoration
        :return: restored sequence
        """
        sliced_sequence = sequence[:, :window_size]  # Take a fixed-length slice.
        # print(len(sliced_sequence))
        interpolated_sequence = F.interpolate(
            sliced_sequence.unsqueeze(0),  # Add batch dimension.
            size=window_size,
            mode='linear',
            align_corners=False
        ).squeeze(0)  # Restore dimensions.
        # print(len(interpolated_sequence))
        return interpolated_sequence

class Slice(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        """
        :param batch_x: (batch, n_channels, seq_len) - Input time series
        :param batch_y: (batch, label_len) - Label
        :param batch_f: (batch, n_channels, feature_len) - Features
        :param batch_mask: (batch, seq_len) - Time series mask
        :param strength: (batch, n_channels, 1) - Strength
        :return: augmented_batch_x, batch_y, batch_mask
        """
        batch, n_channels, seq_len = batch_x.shape
        augmented_batch = []

        for i in range(batch):
            window_size = int(seq_len * (0.5 + 0.5 * strength[i, 0, 0].item())) # min 50% max 100%
            step_size = int(window_size * (0.2 + 0.8 * strength[i, 0, 0].item()))

            sliced_sequences = self._slice_and_interpolate(batch_x[i], window_size, step_size, seq_len)
            augmented_batch.append(sliced_sequences.unsqueeze(0))
        
        augmented_batch_x = torch.cat(augmented_batch, dim=0)
        return augmented_batch_x, batch_y, batch_mask


    def _slice_and_interpolate(self, sequence, window_size, step_size, target_size):
        """
        :param sequence: (n_channels, seq_len) - Input
        :param window_size: - Window Size
        :param step_size: - Step Size
        :param target_size: - Target time series length
        :return: Augmented time series
        """
        n_channels, seq_len = sequence.shape
        slices = []

        for start in range(0, seq_len - window_size + 1, step_size):
            sliced_sequence = sequence[:, start:start + window_size]
            interpolated_sequence = F.interpolate(
                sliced_sequence.unsqueeze(0), # (1, n_channels, window_size)
                size=target_size, # Restore to the original length.
                mode='linear',
                align_corners=False
            ).squeeze(0) # (n_channels, target_size)
            slices.append(interpolated_sequence)
        
        return torch.stack(slices).mean(dim=0)

class TimeWarp(AugmentTransform):
    def augment(self, batch_x, batch_y, batch_f, batch_mask, strength):
        batch, n_channels, seq_len = batch_x.shape
        device = batch_x.device
        
        if strength.shape[1] == 1 and n_channels > 1:
            strength = strength.repeat(1, n_channels, 1)
        
        knot = 4
        num_knots = knot + 2
        
        # Generate random warps
        strength_expanded = strength.expand(-1, -1, num_knots)
        random_warps = 1.0 + torch.randn(batch, n_channels, num_knots, device=device) * strength_expanded
        
        # Create warp_steps (values at knots)
        warp_steps = torch.linspace(0, seq_len-1, num_knots, device=device)
        
        # y_knots = warp_steps * random_warps
        y_knots = random_warps * warp_steps.view(1, 1, num_knots)
        
        # Interpolate to generate smooth curve
        # Reshape to (B*C, 1, 1, K) for bicubic interpolation
        y_knots_reshaped = y_knots.reshape(batch * n_channels, 1, 1, num_knots)
        
        # Output size: (1, seq_len)
        time_warp_vals = F.interpolate(y_knots_reshaped, size=(1, seq_len), mode='bicubic', align_corners=True)
        time_warp_vals = time_warp_vals.reshape(batch, n_channels, seq_len)
        
        # Scale to ensure last point maps to seq_len-1
        scale = (seq_len - 1) / time_warp_vals[..., -1:]
        new_time_coords = time_warp_vals * scale
        new_time_coords = torch.clamp(new_time_coords, 0, seq_len - 1)
        
        # Grid sample
        # batch_x: (B, C, T) -> (B*C, 1, 1, T)
        x_reshaped = batch_x.reshape(batch * n_channels, 1, 1, seq_len)
        
        # Normalize coordinates to [-1, 1]
        grid_x = 2 * new_time_coords.reshape(batch * n_channels, seq_len) / (seq_len - 1) - 1
        grid_y = torch.zeros_like(grid_x)
        
        # Grid: (B*C, 1, T, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(1)
        
        # Sample
        warped_x = F.grid_sample(x_reshaped, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Reshape back
        warped_x = warped_x.reshape(batch, n_channels, seq_len)
        
        return warped_x, batch_y, batch_mask



AVAILABLE_TRANSFORMS = [
    Raw,
    Scale,
    Jitter,
    # MagnitudeWarp,
    # WindowSliceWarp,
    # IAAFT,
    # DRC,
    # Perm,
    # Rotation,
    Downsampling,
    Resampling,
    # Slice,
    FreqWarp
    # TimeWarp
]
