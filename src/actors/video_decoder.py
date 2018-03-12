import ray
import skvideo.io


@ray.remote
class VideoDecoder:
    """Actor which decodes video frame by frame.
    """
    def __init__(self, filename, sample_rate=1.):
        """Initializes the actor.

        Args:
            filename (str): Path to video to decode.
            sample_rate (float): Rate at which to sample the video relative
                to the video's frame rate. For example, if the video's
                frame rate is 30 FPS and `sample_rate` is 0.1, the decoder
                will downsample the video to 10 FPS.
        """
        self.decoder = skvideo.io.vreader(filename)
        self.time = 0
        self.dt = 1 / sample_rate
        self.current_frame = next(self.decoder)

    def next(self):
        """Returns the next decoded frame.

        Returns:
            np.array: The next decoded frame.
            None: if all frames have been decoded.
        """
        if self.time == 0 or (self.time + self.dt) // 1 == self.time // 1:
            self.time += self.dt
            return self.current_frame
        try:
            start_time = self.time
            for _ in range(max(1, int(self.dt))):
                self.current_frame = next(self.decoder)
            self.time += max(1, self.dt // 1)
            return self.current_frame
        except StopIteration:
            return None
