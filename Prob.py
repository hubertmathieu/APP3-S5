import numpy as np
import scipy
import matplotlib.pyplot as plt

nb_harmonics = 32


def normalize(data):
    return np.divide(data, np.amax(data))


def extract_audio(file: str):
    sampling_rate, data = scipy.io.wavfile.read(file)

    return sampling_rate, normalize(data)


def fast_fourier_transform(data):
    fft = np.fft.fft(data)
    return fft


def fast_fourier_transform_with_window(data):
    data_hann = data * np.hanning(len(data))
    fft = np.fft.fft(data_hann)
    return fft


def create_figure(x_arr, y_arr, title, x_label, y_label, x_lim1, x_lim2):
    plt.plot(x_arr, y_arr)
    plt.xlim(x_lim1, x_lim2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()


def find_fundamental_frequency(amplitude_arr, fe, freq_min=261.6, freq_max=493.9):
    min_index = int(freq_min * len(amplitude_arr) / fe)
    max_index = int(freq_max * len(amplitude_arr) / fe)

    index_ff = max(range(min_index, max_index), key=lambda i: amplitude_arr[i])
    ff = (index_ff / len(amplitude_arr)) * fe
    return ff


def return_harmonics(amplitude_arr, phase_arr, nb_harmonics, ff, fe):
    harmonics_arr = []
    for i in range(1, nb_harmonics + 1):
        fh = ff * i  # Fréquence de l'harmonique (en Hz)
        index_fh = int(fh * len(amplitude_arr) / fe)  # m
        amplitude_harmonic = amplitude_arr[index_fh]  # Amplitude de l'harmonique
        phase_harmonic = phase_arr[index_fh]  # Phase des harmonique
        harmonics_arr.append((i, fh, amplitude_harmonic, phase_harmonic))

    return harmonics_arr


def print_harmonics(harmonics):
    # Afficher les 32 premières harmoniques
    for i, (index, frequency, amplitude, phase) in enumerate(harmonics, 1):
        print(f"Harmonique {i}: Indice = {index}, Fréquence = {frequency} Hz, Amplitude = {amplitude}, Phase = {phase}")


def find_order_of_signal():
    w = np.pi / 1000
    for K in range(1, 1000):
        h = (1 / K) * (np.sin(K * w / 2) / np.sin(w / 2))
        if 0.7070 <= h <= 0.7072:
            print("h : ", h, "K : ", K)
            return K  # ordre du filtre


def create_rif_filter(fe):
    N_order = find_order_of_signal()
    w = np.pi / 1000
    fc = (w * fe) / (2 * np.pi)
    K = ((2 * fc * N_order) / fe) + 1

    n = np.arange(-int(N_order / 2), int(N_order / 2))
    m = (2 * np.pi * n) / N_order

    h = (1 / N_order) * np.sin(np.pi * K * n / N_order) / (np.sin(np.pi * n / N_order) + 1e-20)
    h[int(N_order / 2)] = K / N_order

    H = np.fft.fft(h)
    H_shift = np.fft.fftshift(H)
    create_figure(m, (20 * np.log10(np.abs(H_shift))),
                  'Reponse impulsionnelle', 'Frequence', 'Amplitude (DB)', -4, 4)
    return normalize(h)


def create_temporal_envelope(x, h):
    y = np.convolve(np.abs(x), h)
    y_normalized = normalize(y)

    create_figure(np.arange(0, len(y)), y_normalized,
                  'Enveloppe Temporelle', 'Echantillon', 'Amplitude', 0, int(len(x)))
    return y_normalized


def signal_to_audio(harmonics_list, fundamental_freq, sampleRate, envelope, duration):
    t = np.linspace(0, duration, int(sampleRate * duration), endpoint=False)
    audio = np.zeros(int(sampleRate * duration))

    for i in range(len(harmonics_list)):
        audio += harmonics_list[i][2] * np.sin(2 * np.pi * fundamental_freq * t * i + harmonics_list[i][1])

    audio = np.multiply(audio, envelope[:len(audio)])
    audio = normalize(audio)

    return audio


def write_wav_file(filename, audio, sampleRate):
    scipy.io.wavfile.write(filename + '.wav', sampleRate, np.array(audio))


def extract_signal_la():
    Fe, audio_data = extract_audio('note_guitare_LAd.wav')

    amplitude, phase, db_fft, freqz = extracts_parameters(audio_data, Fe)

    create_figure(freqz, phase, 'Phase LA#', 'Freqs (Hz)', 'Phase (rad/éch)', 0, np.amax(freqz))
    create_figure(freqz, db_fft, 'Amplitude LA#', 'Freqs (Hz)', 'Amplitude (DB)', 0, np.amax(freqz))

    fundamental_frequency = find_fundamental_frequency(amplitude, Fe)
    harmonics = return_harmonics(amplitude, phase, nb_harmonics, fundamental_frequency, Fe)
    print_harmonics(harmonics)

    low_filter = create_rif_filter(Fe)
    envelope = create_temporal_envelope(audio_data, low_filter)

    audio = signal_to_audio(harmonics, harmonics[0][1], Fe, envelope, 2)
    write_wav_file("LA#test", audio, Fe)

    generate_betho(harmonics, Fe, envelope, harmonics[0][1])


def create_bandstop_filter(fe):
    N_order = 6000
    n = np.arange(-int(N_order / 2), int(N_order / 2))
    fc1 = 40
    k1 = ((2 * fc1 * N_order) / fe) + 1
    fc0 = 1000
    w0 = (2 * np.pi * fc0) / fe

    h_low_pass = (1 / N_order) * np.sin(np.pi * k1 * n / N_order) / (np.sin(np.pi * n / N_order) + 1e-20)
    h_low_pass[int(N_order / 2)] = k1 / N_order

    diract = np.zeros(N_order)
    diract[int(N_order / 2)] = 1

    h_basson = diract - np.multiply(2 * h_low_pass, np.cos(n * w0))
    create_figure(n, np.abs(h_basson),
                  'Reponse impulsionnelle du filtre coupe-bande', 'Frequence', 'Amplitude (DB)',
                  -int(N_order / 2) - 500, int(N_order / 2) + 500)

    return normalize(h_basson)


def extracts_parameters(audio_data, fe):
    N = len(audio_data)

    fft_result = fast_fourier_transform(audio_data)
    amplitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    db_fft = 20 * np.log10(amplitude)
    freqz = np.fft.fftfreq(N, d=1 / fe)

    return amplitude, phase, db_fft, freqz


def extract_signal_basson():
    Fe, audio_data = extract_audio('note_basson_plus_sinus_1000_Hz.wav')

    amplitude, phase, db_fft, freqz = extracts_parameters(audio_data, Fe)
    create_figure(freqz, phase, 'Phase Bruité', 'Freqs (Hz)', 'Phase (rad/éch)', 0, 1200)
    create_figure(freqz, db_fft, 'Amplitude Bruité', 'Freqs (Hz)', 'Amplitude (DB)', 0, 1200)

    h_basson = create_bandstop_filter(Fe)
    y = np.convolve(audio_data, h_basson)
    y = y * np.hamming(len(y))
    y = np.divide(y, np.amax(y))

    amplitude, phase, db_fft, freqz = extracts_parameters(y, Fe)
    create_figure(freqz, phase, 'Phase Filtré', 'Freqs (Hz)', 'Phase (rad/éch)', 0, 1200)
    create_figure(freqz, db_fft, 'Amplitude Filtré', 'Freqs (Hz)', 'Amplitude (DB)', 0, 1200)

    fundamental_frequency = find_fundamental_frequency(amplitude, Fe)
    harmonics = return_harmonics(amplitude, phase, nb_harmonics, fundamental_frequency, Fe)
    print_harmonics(harmonics)

    low_filter = create_rif_filter(Fe)
    envelope = create_temporal_envelope(y, low_filter)

    audio = signal_to_audio(harmonics, harmonics[0][1], Fe, envelope, 2)
    write_wav_file("Basson", audio, Fe)


def convert_frequencies(note, lad_freq):
    la_freq = lad_freq / 1.06

    match note:
        case "do":
            return la_freq * 0.595
        case "do#":
            return la_freq * 0.630
        case "re":
            return la_freq * 0.667
        case "re#":
            return la_freq * 0.707
        case "mi":
            return la_freq * 0.749
        case "fa":
            return la_freq * 0.794
        case "fa#":
            return la_freq * 0.841
        case "sol":
            return la_freq * 0.891
        case "sol#":
            return la_freq * 0.944
        case "la":
            return la_freq
        case "la#":
            return lad_freq
        case "si":
            return la_freq * 1.123

def silence(fe, duration = 2.0):
    return np.zeros(int(duration * fe))


def generate_betho(harmonics, fe, envelope, fundamental):
    f_sol = convert_frequencies('sol', fundamental)
    f_mib = convert_frequencies('re#', fundamental)
    f_fa = convert_frequencies('fa', fundamental)
    f_re = convert_frequencies('re', fundamental)

    sol = signal_to_audio(harmonics, f_sol, fe, envelope, 0.4)
    mib = signal_to_audio(harmonics, f_mib, fe, envelope, 1.5)
    fa = signal_to_audio(harmonics, f_fa, fe, envelope, 0.4)
    re = signal_to_audio(harmonics, f_re, fe, envelope, 1.5)

    audio = np.concatenate((sol, sol, sol, mib, silence(fe, 1.5), fa, fa, fa, re))
    write_wav_file('Beethoven', audio, fe)


if __name__ == '__main__':
    extract_signal_la()
    extract_signal_basson()
