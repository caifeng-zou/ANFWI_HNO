import torch
import torch.nn.functional as F
import cupy as cp
import numpy as np

from datetime import datetime
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import bandpass

class InputNormalizer(object): 
    def __init__(self, x, eps=0, nmax=None):
        super(InputNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        if nmax is not None:
            self.mean = torch.mean(x[:nmax], dim=(0, 1, 2))
            self.std = torch.std(x[:nmax], dim=(0, 1, 2))
        else:
            self.mean = torch.mean(x, dim=(0, 1, 2))
            self.std = torch.std(x, dim=(0, 1, 2))
            
        self.std[self.std==0] = 1
        self.eps = eps

    def encode(self, x):
        x.sub_(self.mean).div_(self.std + self.eps)  # in-place operations to save memory

    def decode(self, x):
        x.mul_(self.std + self.eps).add_(self.mean)  # in-place operations to save memory

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class OutputNormalizer(object): 
    def __init__(self, x):
        super(OutputNormalizer, self).__init__()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)  # (N, nx, ny, nv, 2)
        nv = x.shape[-2]
        cplx = torch.view_as_complex(x)  # (N, nx, ny, nv)
        means = cplx.abs().mean(dim=(0, 1, 2))  # (nv,)
        stds = cplx.abs().std(dim=(0, 1, 2))
        self.mean = torch.zeros(2*nv)
        self.std = torch.zeros(2*nv)
        for i in range(nv):
            self.mean[2*i:2*i+2] = means[i]
            self.std[2*i:2*i+2] = stds[i]

    def encode(self, x):
        x.sub_(self.mean).div_(self.std)  # in-place operations to save memory

    def decode(self, x):
        x.mul_(self.std).add_(self.mean)  # in-place operations to save memory

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

def sort_datetime(date_strings):
    # Convert date strings to datetime objects
    date_objects = [datetime.strptime(date, '%Y%m%d%H') for date in date_strings]
    # Sort the datetime objects
    date_objects.sort()
    # Convert sorted datetime objects back to the original string format
    sorted_date_strings  = [date.strftime('%Y%m%d%H') for date in date_objects]
    return sorted_date_strings

def convert_to_freq_in_and_out(input_in_time, output_in_time, fs, freqmin, freqmax, pad=0, taper=None):

    #input_in_time: (N, S, S, 3), the last dimension includes Vp, Vs, and srcx
    #output_in_time: (N, S, S, T, 2), the last dimension includes real u and v 
    #-> 
    #output_in_freq: (N*NF, S, S, 4), the last dimension includes complex U and V 
    #input_in_freq: (N*NF, S, S, 4), the last dimension includes Vp, Vs, source location, and w 

    if output_in_time.shape[-2] % 2 == 0:
        output_in_time = F.pad(output_in_time.permute(0, 1, 2, 4, 3), (pad, pad), mode='constant', value=0).permute(0, 1, 2, 4, 3)
    else:
        output_in_time = F.pad(output_in_time.permute(0, 1, 2, 4, 3), (pad, pad+1), mode='constant', value=0).permute(0, 1, 2, 4, 3)
    
    nt = output_in_time.shape[-2]
    freqs = torch.arange(nt // 2 + 1) * fs / (nt - 1)
    ws = 2 * torch.pi * freqs
    freq_to_keep = torch.where((freqs>=freqmin)&(freqs<=freqmax))[0].tolist()
    NF = len(freq_to_keep)

    # output
    if taper is None:
        window = torch.hann_window(output_in_time.size(3)).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(output_in_time.size(0), 
                                                                                                                    output_in_time.size(1), 
                                                                                                                    output_in_time.size(2), 
                                                                                                                    1,
                                                                                                                    output_in_time.size(-1))
    else:
        window = torch.from_numpy(cosine_taper(output_in_time.size(3), p=0.1)).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(output_in_time.size(0), 
                                                                                                                    output_in_time.size(1), 
                                                                                                                    output_in_time.size(2), 
                                                                                                                    1,
                                                                                                                    output_in_time.size(-1))
    
    output_in_freq = torch.fft.rfft(output_in_time * window, 
                                    dim=3, norm='backward')  # Negative frequencies omitted 
    output_in_freq = output_in_freq[:, :, :, freq_to_keep, :]
    output_in_freq = torch.view_as_real(output_in_freq.permute(0, 1, 2, 4, 3))  # view_as_real operates on the last dimension (N, S, S, 2, NF, 2c)
    output_in_freq = output_in_freq.permute(0, 4, 1, 2, 3, 5)  # Move the frenquency domain forward (N, NF, S, S, 2, 2c)
    output_in_freq = output_in_freq.flatten(-2, -1)  # Make complex u, v in the channel dimension
    output_in_freq = output_in_freq.flatten(0, 1)  # Make the freq dimension at the batch location for parallelization
    
    if input_in_time is None:
        return output_in_freq
    
    # input 
    notw = input_in_time.unsqueeze(-2).repeat(1, 1, 1, NF, 1)  # (N, S, S, NF, 3)
    w = ws[freq_to_keep]
    w = w.view(1, 1, 1, -1, 1).repeat(notw.size(0), notw.size(1), notw.size(2), 1, 1)  # (N, S, S, NF, 1)
    input_in_freq = torch.cat((notw, w), dim=-1)  # (N, S, S, NF, 4)
    input_in_freq = input_in_freq.permute(0, 3, 1, 2, 4)  # (N, NF, S, S, 4)
    input_in_freq = input_in_freq.flatten(0, 1)  # (N*NF, S, S, 4), repeated w[0] to w[-1]

    return input_in_freq, output_in_freq

def return_to_time(data_in_freq, nt, fs, freqmin, freqmax, depad=0):

    #nt: signal length after padding
    #data_in_freq: (N*NF, S, S, 4), the last dimension includes complex V 
    #->
    #data_in_time: (N, S, S, T, 2), the last dimension includes real v 

    mark = True
    if nt % 2 != 0:
        mark = False
        nt += 1

    freqs = torch.arange(nt // 2 + 1) * fs / (nt - 1)
    freq_to_keep = torch.where((freqs>=freqmin)&(freqs<=freqmax))[0].tolist()
    NF = len(freq_to_keep)

    device = data_in_freq.device
    data_in_time = data_in_freq.view(-1, NF, data_in_freq.size(-3), data_in_freq.size(-2), data_in_freq.size(-1))  # (N, NF, S, S, Vel*Cplx)
    data_in_time = data_in_time.view(data_in_time.size(0), data_in_time.size(1), data_in_time.size(2), data_in_time.size(3), 2, 2)  # (N, NF, S, S, Vel, Cplx)
    data_in_time = data_in_time.permute(0, 2, 3, 4, 1, 5).contiguous()  # (N, S, S, Vel, NF, Cplx)
    data_in_time = torch.view_as_complex(data_in_time)  # (N, S, S, Vel, NF)
    kept_freq = torch.zeros(data_in_time.size(0), 
                            data_in_time.size(1), 
                            data_in_time.size(2), 
                            data_in_time.size(3),
                            len(freqs), dtype=torch.cfloat, device=device)
    kept_freq[:, :, :, :, freq_to_keep] = data_in_time
    data_in_time = torch.fft.irfft(kept_freq, dim=-1, norm='backward')  #(N, S, S, Vel, T)
    if mark:
        data_in_time = data_in_time[:, :, :, :, depad:nt-depad]
    else:
        data_in_time = data_in_time[:, :, :, :, depad:nt-depad-1]

    data_in_time = data_in_time.permute(0, 1, 2, 4, 3)
    return data_in_time

def add_noise(y, perc):
    std = y.std().item()
    scale = std * perc
    noise = torch.randn(y.shape) * scale
    y = y + noise.to(y.device)
    return y

def get_mask(srcx, xrec, offsetmin, offsetmax):
    '''
    srcx: torch.Size([N])
    ->mask: torch.Size([N, nrec])
    '''
    if srcx.ndim == 0:
        srcx = srcx.unsqueeze(dim=-1)

    assert srcx.ndim == 1
    mask = ((xrec.unsqueeze(0).repeat(len(srcx), 1) - srcx.unsqueeze(-1).repeat(1, len(xrec))).abs() <= offsetmax) &\
            ((xrec.unsqueeze(0).repeat(len(srcx), 1) - srcx.unsqueeze(-1).repeat(1, len(xrec))).abs() > offsetmin)
    return mask

def one_bit_whiten(x):
    '''
    x: (..., nreal), torch.tensor
    ->
    x: (..., nreal), torch.tensor
    '''
    x = x.view(*x.shape[:-1], -1, 2)  # (..., nv, 2)
    x = torch.view_as_complex(x)  # (..., ncomp)
    x = x / (x.abs() + 1e-30)  # one-bit whitening
    x = torch.view_as_real(x)  # (..., nv, 2)
    x = x.flatten(-2, -1)
    return x

def spectral_whitening(waveform, sampling_rate, freqmin=None, freqmax=None, demean=True, whiten=True, window=0, taper=False):
    # If window == 0, amplitude equals to 1 at all frequencies 
    if demean:
        waveform = waveform - waveform.mean()
    
    if freqmin == None and freqmax == None:
        return waveform

    fs = sampling_rate
    nt = waveform.shape[0]

    if taper:
        taper = cosine_taper(nt, p=0.1)
    else:
        taper = np.ones(nt)

    if whiten:
        waveform = cp.array(waveform)
        freqs = cp.arange(nt // 2 + 1) * fs / (nt - 1)
        df = freqs[1] - freqs[0]
        complex_spectrum = cp.fft.rfft(waveform)  # for tapering, * cp.hanning(nt)
        amp_spectrum = cp.abs(complex_spectrum)
        nf = int(cp.round(window/2/df).item())
        WSZ = 2 * nf + 1
        g = cp.array(smooth(cp.abs(amp_spectrum), WSZ))
        whitened_spectrum = complex_spectrum / (g + 1e-30)  # in case 0 in g resulting in nan
        whitened_waveform = cp.fft.irfft(whitened_spectrum, n=nt)
        whitened_waveform = cp.asnumpy(whitened_waveform)
        out = bandpass(whitened_waveform*taper, freqmin=freqmin, freqmax=freqmax, df=fs, zerophase=True)
    else:
        out = bandpass(waveform*taper, freqmin=freqmin, freqmax=freqmax, df=fs, zerophase=True)
        
    return out

def smooth(a, WSZ, usecupy=True):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    if type(a) == torch.Tensor:
        a = a.numpy()
    
    if usecupy:
        a = cp.array(a)
        out0 = cp.convolve(a, cp.ones(WSZ, dtype=int), 'valid') / WSZ    
        r = cp.arange(1, WSZ-1, 2)
        start = cp.cumsum(a[:WSZ-1])[::2] / r
        stop = (cp.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
        out = cp.concatenate((start, out0, stop))
    else:
        out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ    
        r = np.arange(1, WSZ-1, 2)
        start = np.cumsum(a[:WSZ-1])[::2] / r
        stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
        out = np.concatenate((start, out0, stop))
        
    return out

def apply_phase_shift(trace, shift):
    '''
    trace: a trace of seismogram, torch.tensor, (n,), real
    shift: a real number (not in degrees)
    -> A new trace with phase shift applied
    '''
    nt = len(trace) if len(trace) % 2 == 0 else len(trace) + 1
    Y = torch.fft.rfft(trace, n=nt)
    r = Y.abs()
    fi = Y.angle()
    new_fi = fi + shift
    new_real = r * torch.cos(new_fi)
    new_imag = r * torch.sin(new_fi)
    newY = torch.complex(new_real, new_imag)
    ry = torch.fft.irfft(newY)
    return ry[:len(trace)]

def vpvsbrocher(vp, vs, lam):
    vp_brocher = brocher_vp_from_vs(vs)
    L = ((vp - vp_brocher)**2).mean()
    return L * lam

def brocher_vp_from_vs(vs):
    '''
    Calculate vp from vs using Brocher, 2005
    valid for vs between 0 and 4.5 km/s
    vs: m/s
    vp: m/s
    '''
    vs = vs / 1000  # to km/s
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    vp = vp * 1000  # to m/s
    return vp

def lp_prior_single(v, mean, std, eps, coef, p, vmean, vstd):
    # denormalize
    v = v * (std + eps) + mean
    L = (((v - vmean) / vstd) ** p).mean()
    return L * coef