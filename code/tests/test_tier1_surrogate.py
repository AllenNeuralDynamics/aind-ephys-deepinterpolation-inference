import unittest

import numpy as np

from tier1_surrogate import validate_frame_alignment


class DummyRecording:
    def __init__(self, sampling_frequency=30_000.0, segments=1, samples=1_000):
        self.sampling_frequency = sampling_frequency
        self.segments = segments
        self.samples = samples

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_num_segments(self):
        return self.segments

    def get_num_samples(self, segment_index=0):
        return self.samples


class DummySorting:
    def __init__(self, spike_trains, sampling_frequency=30_000.0, segments=1):
        self.spike_trains = spike_trains
        self.sampling_frequency = sampling_frequency
        self.segments = segments
        self.unit_ids = list(spike_trains)

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_num_segments(self):
        return self.segments

    def get_unit_spike_train(self, unit_id, segment_index=0):
        return self.spike_trains[unit_id]


class FrameAlignmentTests(unittest.TestCase):
    def test_accepts_shared_integer_frame_coordinates(self):
        recording = DummyRecording()
        sorting = DummySorting({1: np.array([0, 25, 999], dtype=np.int64)})
        self.assertIsNone(validate_frame_alignment(recording, sorting))

    def test_rejects_sampling_frequency_mismatch(self):
        with self.assertRaisesRegex(ValueError, "sampling-frequency mismatch"):
            validate_frame_alignment(
                DummyRecording(),
                DummySorting({1: np.array([10])}, sampling_frequency=29_999.0),
            )

    def test_rejects_segment_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "segment-count mismatch"):
            validate_frame_alignment(
                DummyRecording(segments=1),
                DummySorting({1: np.array([10])}, segments=2),
            )

    def test_rejects_noninteger_spike_times(self):
        with self.assertRaisesRegex(ValueError, "integer frame indices"):
            validate_frame_alignment(
                DummyRecording(),
                DummySorting({1: np.array([10.5])}),
            )

    def test_rejects_out_of_range_spikes(self):
        with self.assertRaisesRegex(ValueError, "outside recording"):
            validate_frame_alignment(
                DummyRecording(samples=1_000),
                DummySorting({1: np.array([1_000], dtype=np.int64)}),
            )


if __name__ == "__main__":
    unittest.main()