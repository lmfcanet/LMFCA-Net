"""
    This script demonstrates the real-time audio processing potential of LMFCA-Net 
    using a buffering technique. Please note that the code is not performance-optimized, 
    and the current implementation introduces an algorithmic latency of 635 samples 
    (approximately 40 ms).
"""
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
from collections import deque
from threading import Thread
import queue
from lmfca_net import lmfcaNet

class RealTimeProcessor:
    def __init__(self, model, n_fft, hop_length, win_length, samplerate=16000, device='cpu', channels=6):
        self.model = model.to(device).eval()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.samplerate = samplerate
        self.window = torch.hann_window(win_length).to(device)
        self.device = device
        self.q = queue.Queue()
        self.channels = channels
        self.buffer_size = hop_length * (4 - 1) + win_length
        self.output_buffer = deque()  # Buffer to store enhanced audio
        self.buffer = deque()  # Buffer to store input audio
        self.output_q = queue.Queue()
        self.new_stfts = deque(maxlen=4)
        self.model_buffer = torch.zeros((1, 12, 128, 16)).to(device)
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def process_stream(self):
        while True:
            indata = self.q.get()
            if indata is None:
                break
            self.buffer.extend(indata)
            if len(self.buffer) >= self.buffer_size:
                frame = np.array(list(self.buffer)[:self.buffer_size])
                remove_length = self.buffer_size - self.hop_length
                for _ in range(remove_length):
                    self.buffer.popleft()
                frame_tensor = torch.from_numpy(frame).T.float().to(self.device)
                # Compute STFT
                stft = torch.stft(frame_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                 win_length=self.win_length, window=self.window, return_complex=True, center=False)
                stft = torch.view_as_real(stft).contiguous()
                
                assert stft.numel() == 6 * 128 * 4 * 2, f"Unexpected STFT size: {stft.numel()}"
                
                stft = stft.view(1, self.channels*2, 128, 4).to(self.device)
                
                if stft.shape[-1] == 4:
                    self.model_buffer = torch.cat((self.model_buffer[:, :, :, 4:], stft), dim=-1)
                    # Feed the updated buffers into the model
                    with torch.no_grad():
                        mask = self.model(self.model_buffer).permute(0, 2, 3, 1).contiguous()
                        mask = torch.view_as_complex(mask)
                        #print(stft.permute(0, 2, 3, 1).contiguous()[:, :, :, 8:10].shape)
                        ori_stft = torch.view_as_complex(self.model_buffer.permute(0, 2, 3, 1).contiguous()[:, :, :, 8:10]) #(1, 128, model.buffer)
                        #print(stft.shape)
                        enhanced_stft = ori_stft * mask   #(1, 128, model.buffer)
                        
                        # Reconstruct time-domain signal via ISTFT
                        enhanced_signal = torch.istft(enhanced_stft, n_fft=self.n_fft, hop_length=self.hop_length,
                                                     win_length=self.win_length, window=self.window).cpu().numpy()

                        # Take the last self.buffer_size - self.hop_length samples
                        enhanced_signal = enhanced_signal[:, -(self.buffer_size - self.hop_length):]
                
                        # Normalize to prevent clipping
                        enhanced_signal = enhanced_signal / np.max(np.abs(enhanced_signal) + 1e-8)
                
                        # Add to output buffer
                        self.output_q.put(enhanced_signal.astype(np.float32))
    
    def play_stream(self):
        """
        Thread target for playing back enhanced audio in real-time.
        """
        def callback(outdata, frames, time_info, status):
            """
            Callback function for output audio stream.

            Args:
                outdata (numpy.ndarray): Buffer to be filled with audio data.
                frames (int): Number of frames to output.
                time_info (CData): Timing information.
                status (CallbackFlags): Status flags.
            """
            if status:
                print(f"Output Stream Status: {status}")

            try:
                # Attempt to retrieve a block from the output queue
                data = self.output_q.get_nowait()
                if data.shape != (1, self.buffer_size - self.hop_length):
                    print(f"Unexpected data shape: {data.shape} vs expected {(1, self.buffer_size - self.hop_length)}")
                    data = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
            except queue.Empty:
                # If no data is available, output silence
                data = np.zeros((self.buffer_size, self.channels), dtype=np.float32)

            # Assign data to outdata without reshaping
            outdata[:] = data
            
        with sd.OutputStream(
            callback=callback,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.buffer_size,  # Must match the enqueued block size
            dtype='float32'
        ):
            while not self.stop_flag:
                sd.sleep(100)  # Keep the thread alive
    
    def start(self):
        self.stop_flag = False
        self.process_thread = Thread(target=self.process_stream, daemon=True)
        self.process_thread.start()
        self.play_thread = Thread(target=self.play_stream, daemon=True)
        self.play_thread.start()
    
    def stop(self):
        self.q.put(None)
        self.stop_flag = True
        self.process_thread.join()
        self.play_thread.join()
        

if __name__ == "__main__":
    model = lmfcaNet(in_ch=12)
    processor = RealTimeProcessor(
        model=model,
        n_fft=254,         
        hop_length=127,    
        win_length=254,    
        samplerate=16000, 
        device='cpu'       
    )

    # Start processing and playback threads
    processor.start()
    # Open an input stream to capture real-time audio
    try:
        with sd.InputStream(callback=processor.audio_callback, channels=6, samplerate=16000, blocksize=254):
            print("Processing audio... Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        processor.stop()
