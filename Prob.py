import numpy as np
import scipy
import matplotlib.pyplot as plt

nb_harmonics = 32


def normalize(data):
    return np.divide(data, np.amax(data))


def to_db(data):
    return 20 * np.log10(np.abs(data))


def extract_audio(file: str):
    sampling_rate, data = scipy.io.wavfile.read(file)

    return sampling_rate, normalize(data)


def fast_fourier_transform(data):
    fft = np.fft.fft(data)
    return fft


def fast_fourier_transform_with_window(data):
    data_hann = data * np.hanning(len(data))
    fft = fast_fourier_transform(data_hann)
    return fft


def create_figure(x_arr, y_arr, title, x_label, y_label, x_lim1, x_lim2):
    plt.plot(x_arr, y_arr)
    plt.xlim(x_lim1, x_lim2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def create_figure_harm(x_arr, y_arr, title, x_label, y_label, y_lim1, y_lim2):
    plt.stem(x_arr, y_arr)
    plt.xlim(0, 16000)
    plt.ylim(y_lim1, y_lim2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def find_fundamental_frequency(amplitude_arr, fe, freq_min=261.6, freq_max=493.9):
    min_index = int(freq_min * len(amplitude_arr) / fe)
    max_index = int(freq_max * len(amplitude_arr) / fe)

    index_ff = max(range(min_index, max_index), key=lambda i: amplitude_arr[i])
    ff = (index_ff / len(amplitude_arr)) * fe
    return ff


def plot_synth_singal(audio, Fe, name, end=0):
    amplitude, phase, db_fft, freqz = extracts_parameters(audio, Fe)
    if end == 0:
        end = np.amax(freqz)
    create_figure(freqz, db_fft, 'Amplitude ' + name + ' synth', 'Freqs (Hz)', 'Amplitude (DB)', 0, end)

    create_temporal_envelope(audio, create_rif_filter(Fe), name + ' synth')


def return_harmonics(amplitude_arr, phase_arr, nb_harmonics, ff, fe):
    amplitude_harmonic, phase_harmonic, freq_harmonic, = [], [], []
    for i in range(0, nb_harmonics + 1):
        freq_harmonic.append(ff * i) #Fréquence de l'harmonique (en Hz)
        index_fh = int(freq_harmonic[i] * len(amplitude_arr) / fe)  # m
        amplitude_harmonic.append(amplitude_arr[index_fh])  # Amplitude de l'harmonique
        phase_harmonic.append(phase_arr[index_fh])  # Phase des harmonique

    return amplitude_harmonic, phase_harmonic, freq_harmonic


def print_harmonics(amplitude, phase, frequency):
    # Afficher les 32 premières harmoniques
    for i in range(0, len(amplitude)):
        print(f"Harmonique {i}: Indice = {i}, Fréquence = {frequency[i]} Hz, Amplitude = {amplitude[i]}, Phase = {phase[i]}")


def find_order_of_signal():
    w = np.pi / 1000
    for K in range(1, 1000):
        h = (1 / K) * (np.sin(K * w / 2) / np.sin(w / 2))
        if 0.7070 <= h <= 0.7072:
            print("h : ", h, "K : ", K)
            return K  # ordre du filtre


def create_rif_filter(fe, show=False):
    N_order = find_order_of_signal()

    w = np.pi / 1000
    fc = (w * fe) / (2 * np.pi)
    K = ((2 * fc * N_order) / fe) + 1

    n = np.arange(-int(N_order / 2), int(N_order / 2))
    m = (2 * np.pi * n) / N_order

    h = (1 / N_order) * np.sin(np.pi * K * n / N_order) / (np.sin(np.pi * n / N_order) + 1e-20)
    h[int(N_order / 2)] = K / N_order

    if show:
        H = fast_fourier_transform(h)
        H_shift = np.fft.fftshift(H)
        create_figure(m, to_db(H_shift),
                      'Reponse impulsionnelle filtre l-p', 'Frequence Normalisée', 'Amplitude (DB)', -4, 4)
    return normalize(h)


def create_temporal_envelope(x, h, name):
    y = np.convolve(np.abs(x), h)
    y_normalized = normalize(y)

    create_figure(np.arange(0, len(y)), y_normalized,
                  'Enveloppe Temporelle ' + name, 'Echantillon', 'Amplitude', 0, int(len(x)))
    return y_normalized


def signal_to_audio(amplitude, phase, fundamental, sample_rate, envelope, duration):
    # Init signal
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros(int(sample_rate * duration))
    shorten_enveloppe = envelope[:len(audio)]

    # Créer signal audio avec phase, amplitude et freq des harmoniques
    for i in range(len(amplitude)):
        audio += amplitude[i] * np.sin(2 * np.pi * fundamental * i * t + phase[i])

    audio = np.multiply(audio, shorten_enveloppe)
    audio = normalize(audio)

    return audio


def write_wav_file(filename, audio, sampleRate):
    scipy.io.wavfile.write(filename + '.wav', sampleRate, np.array(audio))


def extract_signal_la():
    Fe, audio_data = extract_audio('note_guitare_LAd.wav')

    # Paramètres du signal
    amplitude, phase, db_fft, freqz = extracts_parameters(audio_data, Fe)
    create_figure(freqz, db_fft, 'Amplitude LA#', 'Freqs (Hz)', 'Amplitude (DB)', 0, np.amax(freqz))

    # Harmoniques
    fundamental_frequency = find_fundamental_frequency(amplitude, Fe)
    amplitude_ham, phase_ham, freq_ham = return_harmonics(amplitude, phase, nb_harmonics, fundamental_frequency, Fe)
    print_harmonics(amplitude_ham, phase_ham, freq_ham)

    # Création de l'enveloppe
    low_filter = create_rif_filter(Fe, True)
    envelope = create_temporal_envelope(audio_data, low_filter, ' LA#')

    # Synth de la note LA#
    audio = signal_to_audio(amplitude_ham, phase_ham, fundamental_frequency, Fe, envelope, 3.5)
    write_wav_file("LA#test", audio, Fe)
    plot_synth_singal(audio, Fe, "LA#")

    generate_betho(amplitude_ham, phase_ham, Fe, envelope, fundamental_frequency)


def create_bandstop_filter(fe):
    # Variable nécessaires
    N_order = 6000
    n = np.arange(-int(N_order / 2), int(N_order / 2))
    fc1 = 40
    k1 = ((2 * fc1 * N_order) / fe) + 1
    fc0 = 1000
    w0 = (2 * np.pi * fc0) / fe

    # Création du filtre l-p
    h_low_pass = (1 / N_order) * np.sin(np.pi * k1 * n / N_order) / (np.sin(np.pi * n / N_order) + 1e-20)
    h_low_pass[int(N_order / 2)] = k1 / N_order

    # initialiser Diract
    diract = np.zeros(N_order)
    diract[int(N_order / 2)] = 1

    # Création b-s
    h_basson = diract - np.multiply(2 * h_low_pass, np.cos(n * w0))

    amplitude, phase, db_fft, freqz = extracts_parameters(h_basson, fe)

    create_figure(freqz, db_fft,
                  'Reponse impulsionnelle du filtre coupe-bande', 'Frequence', 'Amplitude (DB)',
                  0, 5000)

    return normalize(h_basson)


def extracts_parameters(audio_data, fe):
    N = len(audio_data)

    # Domaine frequentiel
    fft_result = fast_fourier_transform(audio_data)
    amplitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    db_fft = to_db(amplitude)
    freqz = np.fft.fftfreq(N, d=1 / fe)

    return amplitude, phase, db_fft, freqz


def extract_signal_basson():
    Fe, audio_data = extract_audio('note_basson_plus_sinus_1000_Hz.wav')

    # Création du filtre l-p pour l'enveloppe
    low_filter = create_rif_filter(Fe)

    # Extraction of parameters for the original sound of the basson
    amplitude, phase, db_fft, freqz = extracts_parameters(audio_data, Fe)
    create_figure(freqz, db_fft, 'Amplitude du basson bruité', 'Freqs (Hz)', 'Amplitude (DB)', 0, 1200)

    # BandStop filter creation
    h_basson = create_bandstop_filter(Fe)

    # Convolution between the original signal and the filter while applying a window and normalizing on 1 the signal
    y = np.convolve(audio_data, h_basson)
    y = y * np.hamming(len(y))
    y = normalize(y)

    # Extraction of parameters for new sound of the basson
    amplitude, phase, db_fft, freqz = extracts_parameters(y, Fe)
    create_figure(freqz, db_fft, 'Amplitude Filtré Basson', 'Freqs (Hz)', 'Amplitude (DB)', 0, 1200)

    # Harmonics for the filtered sound of the basson
    fundamental_frequency = find_fundamental_frequency(amplitude, Fe)
    amplitude_ham, phase_ham, freq_ham = return_harmonics(amplitude, phase, nb_harmonics, fundamental_frequency, Fe)
    print_harmonics(amplitude_ham, phase_ham, freq_ham)

    # Create figure for the harmonics
    create_figure_harm(freq_ham, amplitude_ham, 'Amplitudes des harmoniques', 'Freqs (Hz)', 'Amplitude', np.amin(amplitude_ham), np.amax(amplitude_ham))
    create_figure_harm(freq_ham, phase_ham, 'Phases des harmoniques', 'Freqs (Hz)', 'Phase', np.amin(phase_ham), np.amax(phase_ham))

    # Create envelope for the synthesized sound
    envelope = create_temporal_envelope(y, low_filter, 'basson')

    # Singal Audio
    audio = signal_to_audio(amplitude_ham, phase_ham, fundamental_frequency, Fe, envelope, 3)

    audio = np.convolve(audio, h_basson)
    audio = audio * np.hamming(len(audio))
    audio = normalize(audio)

    write_wav_file("Basson", audio, Fe)
    plot_synth_singal(audio, Fe, "Basson", 1200)


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


def generate_betho(amplitude, phase, fe, envelope, fundamental):
    f_sol = convert_frequencies('sol', fundamental)
    f_mib = convert_frequencies('re#', fundamental)
    f_fa = convert_frequencies('fa', fundamental)
    f_re = convert_frequencies('re', fundamental)

    sol = signal_to_audio(amplitude, phase, f_sol, fe, envelope, 0.4)
    mib = signal_to_audio(amplitude, phase, f_mib, fe, envelope, 1.5)
    fa = signal_to_audio(amplitude, phase, f_fa, fe, envelope, 0.4)
    re = signal_to_audio(amplitude, phase, f_re, fe, envelope, 1.5)

    audio = np.concatenate((sol, sol, sol, mib, silence(fe, 1.5), fa, fa, fa, re))
    write_wav_file('Beethoven', audio, fe)


if __name__ == '__main__':
    extract_signal_la()
    extract_signal_basson()
