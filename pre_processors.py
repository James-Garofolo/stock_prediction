import numpy as np

def do_fft(in_data):
    dtft = np.fft.fft(in_data)
    r_dtft = np.real(dtft)
    i_dtft = np.imag(dtft)

    return r_dtft, i_dtft

def derivative(in_data):
    d = np.array(in_data[1]-in_data[0])
    for a in range(1, len(in_data)-1):
        d = np.append(d, in_data[a+1]-in_data[a])

    return d

def package(windows, sizes):
    features = []
    for a in range(len(sizes)):
        window = windows[a]
        features.extend(window[len(window)-sizes[a]:len(window)])

    features = np.array(features)
    return features

def pre_process(in_data, dsize, fftsize, ddtsize):
    r_fft, i_fft = do_fft(in_data)
    ddt = derivative(in_data)
    avg = [np.average(in_data)]
    st_dev = [np.std(in_data)]
    features = package([in_data, r_fft, i_fft, ddt, avg, st_dev],
                        [dsize, fftsize, fftsize, ddtsize, 1, 1])

    return features



if __name__ == "__main__":
    a = [[1, 2, 3, 4], [5, 6, 7, 8]]
    sizes = [2, 3]

    b = package(a, sizes)

    print(b)
