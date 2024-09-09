import os
import sys
import torch
import pickle
import numpy as np
import torchaudio
import pretty_midi
from pydub import AudioSegment
from helper import Model_SPEC2MIDI, Encoder_SPEC2MIDI, Decoder_SPEC2MIDI, EncoderLayer, DecoderLayer_Zero, DecoderLayer, MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

SR = 16000
FFT_BINS = 2048
WINDOW_LENGTH = 2048
HOP_SAMPLE = 256
PAD_MODE = 'constant'
MEL_BINS = 256
N_BINS = 256
LOG_OFFSET = 1e-8
WINDOWS = 'hann'

MARGIN_B = 32
MARGIN_F = 32
NUM_FRAME = 128

NOTE_MIN = 21
NOTE_MAX = 108
NUM_NOTE = 88
NUM_VELOCITY = 128

if LOG_OFFSET > 0.0:
    MIN_VALUE = np.log(LOG_OFFSET).astype(np.float32)
else:
    MIN_VALUE = LOG_OFFSET


class AMT():
    def __init__(self, model_path, batch_size=1, verbose_flag=False):
        if verbose_flag is True:
            print('torch version: '+torch.__version__)
            print('torch cuda   : '+str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if model_path is None:
            self.model = None
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model = self.model.to(self.device)
            self.model.eval()
            if verbose_flag is True:
                print(self.model)

        self.batch_size = batch_size



    def wav2feature(self, f_wav):
        ### torchaudio
        # torchaudio.transforms.MelSpectrogram()
        # default
        #  sapmle_rate(16000)
        #  win_length(n_fft)
        #  hop_length(win_length//2)
        #  n_fft(400)
        #  f_min(0)
        #  f_max(None)
        #  pad(0)
        #  n_mels(128)
        #  window_fn(hann_window)
        #  center(True)
        #  power(2.0)
        #  pad_mode(reflect)
        #  onesided(True)
        #  norm(None)
        ## melfilter: htk
        ## normalize: none -> slaney

        wave, sr = torchaudio.load(f_wav)
        wave_mono = torch.mean(wave, dim=0)
        tr_fsconv = torchaudio.transforms.Resample(sr, SR)
        wave_mono_16k = tr_fsconv(wave_mono)
        tr_mel = torchaudio.transforms.MelSpectrogram(sample_rate = SR, n_fft = FFT_BINS, win_length = WINDOW_LENGTH, hop_length = HOP_SAMPLE, pad_mode = PAD_MODE, n_mels = MEL_BINS, norm='slaney')
        mel_spec = tr_mel(wave_mono_16k)
        a_feature = (torch.log(mel_spec + LOG_OFFSET)).T

        return a_feature


    def transcript(self, a_feature, mode='combination', ablation_flag=False):
        # a_feature: [num_frame, n_mels]
        a_feature = np.array(a_feature, dtype=np.float32)

        a_tmp_b = np.full([MARGIN_B, N_BINS], MIN_VALUE, dtype=np.float32)
        len_s = int(np.ceil(a_feature.shape[0] / NUM_FRAME) * NUM_FRAME) - a_feature.shape[0]
        a_tmp_f = np.full([len_s + MARGIN_F, N_BINS], MIN_VALUE, dtype=np.float32)
        a_input = torch.from_numpy(np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0))
        # a_input: [margin_b+a_feature.shape[0]+len_s+margin_f, n_bins]

        a_output_onset_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
        a_output_offset_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
        a_output_mpe_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
        a_output_velocity_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.int8)

        if mode == 'combination':
            a_output_onset_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
            a_output_offset_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
            a_output_mpe_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
            a_output_velocity_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.int8)

        self.model.eval()
        for i in range(0, a_feature.shape[0], NUM_FRAME):
            input_spec = (a_input[i:i + MARGIN_B + NUM_FRAME + MARGIN_F]).T.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if mode == 'combination':
                    if ablation_flag is True:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    else:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    # output_onset: [batch_size, n_frame, n_note]
                    # output_offset: [batch_size, n_frame, n_note]
                    # output_mpe: [batch_size, n_frame, n_note]
                    # output_velocity: [batch_size, n_frame, n_note, n_velocity]
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = self.model(input_spec)

            a_output_onset_A[i:i + NUM_FRAME] = (output_onset_A.squeeze(0)).to('cpu').detach().numpy()
            a_output_offset_A[i:i + NUM_FRAME] = (output_offset_A.squeeze(0)).to('cpu').detach().numpy()
            a_output_mpe_A[i:i + NUM_FRAME] = (output_mpe_A.squeeze(0)).to('cpu').detach().numpy()
            a_output_velocity_A[i:i + NUM_FRAME] = (output_velocity_A.squeeze(0).argmax(2)).to('cpu').detach().numpy()

            if mode == 'combination':
                a_output_onset_B[i:i + NUM_FRAME] = (output_onset_B.squeeze(0)).to('cpu').detach().numpy()
                a_output_offset_B[i:i + NUM_FRAME] = (output_offset_B.squeeze(0)).to('cpu').detach().numpy()
                a_output_mpe_B[i:i + NUM_FRAME] = (output_mpe_B.squeeze(0)).to('cpu').detach().numpy()
                a_output_velocity_B[i:i + NUM_FRAME] = (output_velocity_B.squeeze(0).argmax(2)).to('cpu').detach().numpy()

        if mode == 'combination':
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A, a_output_onset_B, a_output_offset_B, a_output_mpe_B, a_output_velocity_B
        else:
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


    def transcript_stride(self, a_feature, n_offset, mode='combination', ablation_flag=False):
        # a_feature: [num_frame, n_mels]
        a_feature = np.array(a_feature, dtype=np.float32)

        half_frame = int(NUM_FRAME / 2)
        a_tmp_b = np.full([MARGIN_B + n_offset, N_BINS], MIN_VALUE, dtype=np.float32)
        tmp_len = a_feature.shape[0] + MARGIN_B + MARGIN_F + half_frame
        len_s = int(np.ceil(tmp_len / half_frame) * half_frame) - tmp_len
        a_tmp_f = np.full([len_s + MARGIN_F + (half_frame-n_offset), N_BINS], MIN_VALUE, dtype=np.float32)

        a_input = torch.from_numpy(np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0))
        # a_input: [n_offset+margin_b+a_feature.shape[0]+len_s+(half_frame-n_offset)+margin_f, n_bins]

        a_output_onset_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
        a_output_offset_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
        a_output_mpe_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
        a_output_velocity_A = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.int8)

        if mode == 'combination':
            a_output_onset_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
            a_output_offset_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
            a_output_mpe_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.float32)
            a_output_velocity_B = np.zeros((a_feature.shape[0]+len_s, NUM_NOTE), dtype=np.int8)

        self.model.eval()
        for i in range(0, a_feature.shape[0], half_frame):
            input_spec = (a_input[i:i + MARGIN_B + NUM_FRAME + MARGIN_F]).T.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if mode == 'combination':
                    if ablation_flag is True:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    else:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    # output_onset: [batch_size, n_frame, n_note]
                    # output_offset: [batch_size, n_frame, n_note]
                    # output_mpe: [batch_size, n_frame, n_note]
                    # output_velocity: [batch_size, n_frame, n_note, n_velocity]
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = self.model(input_spec)

            a_output_onset_A[i:i+half_frame] = (output_onset_A.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
            a_output_offset_A[i:i+half_frame] = (output_offset_A.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
            a_output_mpe_A[i:i+half_frame] = (output_mpe_A.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
            a_output_velocity_A[i:i+half_frame] = (output_velocity_A.squeeze(0)[n_offset:n_offset+half_frame].argmax(2)).to('cpu').detach().numpy()

            if mode == 'combination':
                a_output_onset_B[i:i+half_frame] = (output_onset_B.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
                a_output_offset_B[i:i+half_frame] = (output_offset_B.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
                a_output_mpe_B[i:i+half_frame] = (output_mpe_B.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
                a_output_velocity_B[i:i+half_frame] = (output_velocity_B.squeeze(0)[n_offset:n_offset+half_frame].argmax(2)).to('cpu').detach().numpy()

        if mode == 'combination':
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A, a_output_onset_B, a_output_offset_B, a_output_mpe_B, a_output_velocity_B
        else:
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


    def mpe2note(self, a_onset=None, a_offset=None, a_mpe=None, a_velocity=None, thred_onset=0.5, thred_offset=0.5, thred_mpe=0.5, mode_velocity='ignore_zero', mode_offset='shorter'):
        ## mode_velocity
        ##  org: 0-127
        ##  ignore_zero: 0-127 (output note does not include 0) (default)

        ## mode_offset
        ##  shorter: use shorter one of mpe and offset (default)
        ##  longer : use longer one of mpe and offset
        ##  offset : use offset (ignore mpe)

        a_note = []
        hop_sec = float(HOP_SAMPLE / SR)

        for j in range(NUM_NOTE):
            # find local maximum
            a_onset_detect = []
            for i in range(len(a_onset)):
                if (a_onset[i][j] >= thred_onset).any():
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_onset[i][j] > a_onset[ii][j]:
                            left_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_onset)):
                        if a_onset[i][j] > a_onset[ii][j]:
                            right_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_onset) - 1):
                            onset_time = i * hop_sec
                        else:
                            if a_onset[i-1][j] == a_onset[i+1][j]:
                                onset_time = i * hop_sec
                            elif a_onset[i-1][j] > a_onset[i+1][j]:
                                onset_time = (i * hop_sec - (hop_sec * 0.5 * (a_onset[i-1][j] - a_onset[i+1][j]) / (a_onset[i][j] - a_onset[i+1][j])))
                            else:
                                onset_time = (i * hop_sec + (hop_sec * 0.5 * (a_onset[i+1][j] - a_onset[i-1][j]) / (a_onset[i][j] - a_onset[i-1][j])))
                        a_onset_detect.append({'loc': i, 'onset_time': onset_time})
            a_offset_detect = []
            for i in range(len(a_offset)):
                if (a_offset[i][j] >= thred_offset).any():
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_offset[i][j] > a_offset[ii][j]:
                            left_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_offset)):
                        if a_offset[i][j] > a_offset[ii][j]:
                            right_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_offset) - 1):
                            offset_time = i * hop_sec
                        else:
                            if a_offset[i-1][j] == a_offset[i+1][j]:
                                offset_time = i * hop_sec
                            elif a_offset[i-1][j] > a_offset[i+1][j]:
                                offset_time = (i * hop_sec - (hop_sec * 0.5 * (a_offset[i-1][j] - a_offset[i+1][j]) / (a_offset[i][j] - a_offset[i+1][j])))
                            else:
                                offset_time = (i * hop_sec + (hop_sec * 0.5 * (a_offset[i+1][j] - a_offset[i-1][j]) / (a_offset[i][j] - a_offset[i-1][j])))
                        a_offset_detect.append({'loc': i, 'offset_time': offset_time})

            time_next = 0.0
            time_offset = 0.0
            time_mpe = 0.0
            for idx_on in range(len(a_onset_detect)):
                # onset
                loc_onset = a_onset_detect[idx_on]['loc']
                time_onset = a_onset_detect[idx_on]['onset_time']

                if idx_on + 1 < len(a_onset_detect):
                    loc_next = a_onset_detect[idx_on+1]['loc']
                    #time_next = loc_next * hop_sec
                    time_next = a_onset_detect[idx_on+1]['onset_time']
                else:
                    loc_next = len(a_mpe)
                    time_next = (loc_next-1) * hop_sec

                # offset
                loc_offset = loc_onset+1
                flag_offset = False
                #time_offset = 0###
                for idx_off in range(len(a_offset_detect)):
                    if loc_onset < a_offset_detect[idx_off]['loc']:
                        loc_offset = a_offset_detect[idx_off]['loc']
                        time_offset = a_offset_detect[idx_off]['offset_time']
                        flag_offset = True
                        break
                if loc_offset > loc_next:
                    loc_offset = loc_next
                    time_offset = time_next

                # offset by MPE
                # (1frame longer)
                loc_mpe = loc_onset+1
                flag_mpe = False
                #time_mpe = 0###
                for ii_mpe in range(loc_onset+1, loc_next):
                    if (a_mpe[ii_mpe][j] < thred_mpe).any():
                        loc_mpe = ii_mpe
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                # (right algorighm)
                loc_mpe = loc_onset
                flag_mpe = False
                for ii_mpe in range(loc_onset+1, loc_next+1):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe-1
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                pitch_value = int(j + NOTE_MIN)
                velocity_value = int(a_velocity[loc_onset][j])

                if (flag_offset is False) and (flag_mpe is False):
                    offset_value = float(time_next)
                elif (flag_offset is True) and (flag_mpe is False):
                    offset_value = float(time_offset)
                elif (flag_offset is False) and (flag_mpe is True):
                    offset_value = float(time_mpe)
                else:
                    if mode_offset == 'offset':
                        ## (a) offset
                        offset_value = float(time_offset)
                    elif mode_offset == 'longer':
                        ## (b) longer
                        if loc_offset >= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                    else:
                        ## (c) shorter
                        if loc_offset <= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                if mode_velocity != 'ignore_zero':
                    a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})
                else:
                    if velocity_value > 0:
                        a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})

                if (len(a_note) > 1) and \
                   (a_note[len(a_note)-1]['pitch'] == a_note[len(a_note)-2]['pitch']) and \
                   (a_note[len(a_note)-1]['onset'] < a_note[len(a_note)-2]['offset']):
                    a_note[len(a_note)-2]['offset'] = a_note[len(a_note)-1]['onset']

        a_note = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])
        return a_note


    def note2midi(self, a_note, f_midi):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for note in a_note:
            instrument.notes.append(pretty_midi.Note(velocity=note['velocity'], pitch=note['pitch'], start=note['onset'], end=note['offset']))
        midi.instruments.append(instrument)
        midi.write(f_midi)

        return
    
def transcribe_audio_to_midi(model_path, audio_path, output_midi_path):
    # Resolve the path to the model file relative to the current script
    base_dir = os.path.dirname(__file__)  # Get the directory of the current script
    model_path = os.path.join(base_dir, model_path)  # Join with the model file name
    audio_path = os.path.join(base_dir, audio_path)  # Join with the model file name
    output_midi_path = os.path.join(base_dir, output_midi_path)  # Join with the model file name

    amt = AMT(model_path)
    features = amt.wav2feature(audio_path)
    outputs = amt.transcript(features)
    notes = amt.mpe2note(a_onset = outputs[0], a_offset = outputs[1], a_mpe = outputs[2], a_velocity = outputs[3])
    amt.note2midi(notes, output_midi_path)
    print(f"Transcription complete. MIDI file saved to {output_midi_path}")

def convert_audio(input_file, output_format="wav"):
    try:
        output_file = os.path.splitext(input_file)[0] + "." + output_format
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format=output_format)
        return output_file
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    wav_file = convert_audio(input_file, "wav")
    
    transcribe_audio_to_midi(
        model_path='./model/best_model.pkl',
        audio_path=wav_file,
        output_midi_path=f"uploads\\{os.path.splitext(os.path.basename(wav_file))[0]}.midi"
    )
