clear; close all; clc

load('dataset_Guy.mat'); % Make sure the dataset file (.mat) is in the current directory
pqd_signals = vertcat(SignalsDataBase.signals);
N = size(pqd_signals, 1);
length = size(pqd_signals, 2);
pqd_labels = string(reshape({SignalsDataBase(:).labels}, [N 1]));

% Define CWT filter bank and its parameters
fs = 32000;
freq_limits = [100 2000];
voices_per_octave = 48;

fb = cwtfilterbank("SignalLength", length, "Wavelet", "Morse", ...
    "VoicesPerOctave", voices_per_octave, "SamplingFrequency", fs, ...
    "FrequencyLimits", freq_limits);
%% Long run
MySignalsDataBase = struct('labels', {}, 'cwts', {});

for i=1:N
    i
%     Compute CWT of a PQ signal
    signal_cwt = abs(fb.wt(pqd_signals(i, :)));

%     Convert to image with 3 channels and resize it to some shape
    signal_cwt_img = ind2rgb(im2uint8(rescale(signal_cwt)), jet);
    signal_cwt_img = imresize(signal_cwt_img, [256, 256]);

    MySignalsDataBase(end+1).('labels') = pqd_labels(i);
    MySignalsDataBase(end).('cwts') = signal_cwt_img;
end

save('my_dataset.mat', 'MySignalsDataBase', '-v7.3');

%% Get example signals for possible PQDs
% Define seed for reproducability
rng('default');
rng(1);

indices_normal = find(strcmp(pqd_labels, 'Normal'));
index_normal = randsample(indices_normal, 1);

indices_sag = find(strcmp(pqd_labels, 'Sag'));
index_sag = randsample(indices_sag, 1);

indices_swell = find(strcmp(pqd_labels, 'Swell'));
index_swell = randsample(indices_swell, 1);

indices_interruption = find(strcmp(pqd_labels, 'Interruption'));
index_interruption = randsample(indices_interruption, 1);

indices_harmonics = find(strcmp(pqd_labels, 'Harmonics'));
index_harmonics = randsample(indices_harmonics, 1);

indices_oscillatory_transient = find(strcmp(pqd_labels, 'Oscillatory transient'));
index_oscillatory_transient = randsample(indices_oscillatory_transient, 1);

indices_sag_harmonics = find(strcmp(pqd_labels, 'Sag+Harmonics'));
index_sag_harmonics = randsample(indices_sag_harmonics, 1);

indices_swell_harmonics = find(strcmp(pqd_labels, 'Swell+Harmonics'));
index_swell_harmonics = randsample(indices_swell_harmonics, 1);

indices_flicker = find(strcmp(pqd_labels, 'Flicker'));
index_flicker = randsample(indices_flicker, 1);

indices_notch = find(strcmp(pqd_labels, 'Notch'));
index_notch = randsample(indices_notch, 1);

indices_spike = find(strcmp(pqd_labels, 'Spike'));
index_spike = randsample(indices_spike, 1);

indices_flicker_harmonics = find(strcmp(pqd_labels, 'Flicker+Harmonics'));
index_flicker_harmonics = randsample(indices_flicker_harmonics, 1);

indices_interruption_harmonics = find(strcmp(pqd_labels, 'Interruption+Harmonics'));
index_interruption_harmonics = randsample(indices_interruption_harmonics, 1);

indices_flicker_sag = find(strcmp(pqd_labels, 'Flicker+Sag'));
index_flicker_sag = randsample(indices_flicker_sag, 1);

indices_flicker_swell = find(strcmp(pqd_labels, 'Flicker+Swell'));
index_flicker_swell = randsample(indices_flicker_swell, 1);

indices_impulsive_transient = find(strcmp(pqd_labels, 'Impulsive Transient'));
index_impulsive_transient = randsample(indices_impulsive_transient, 1);

normal = pqd_signals(index_normal, :);
sag = pqd_signals(index_sag, :);
swell = pqd_signals(index_swell, :);
interruption = pqd_signals(index_interruption, :);
harmonics = pqd_signals(index_harmonics, :);
oscillatory_transient = pqd_signals(index_oscillatory_transient, :);
sag_harmonics = pqd_signals(index_sag_harmonics, :);
swell_harmonics = pqd_signals(index_swell_harmonics, :);
flicker = pqd_signals(index_flicker, :);
notch = pqd_signals(index_notch, :);
spike = pqd_signals(index_spike, :);
flicker_harmonics = pqd_signals(index_flicker_harmonics, :);
interruption_harmonics = pqd_signals(index_interruption_harmonics, :);
flicker_sag = pqd_signals(index_flicker_sag, :);
flicker_swell = pqd_signals(index_flicker_swell, :);
impulsive_transient = pqd_signals(index_impulsive_transient, :);

%% Plot signals
figure(1);

subplot(4, 4, 1);
plot(normal);
title('Normal')

subplot(4, 4, 2);
plot(sag);
title('Sag')

subplot(4, 4, 3);
plot(swell);
title('Swell')

subplot(4, 4, 4);
plot(interruption);
title('Interruption')

subplot(4, 4, 5);
plot(harmonics);
title('Harmonics')

subplot(4, 4, 6);
plot(oscillatory_transient);
title('Oscillatory transient')

subplot(4, 4, 7);
plot(sag_harmonics);
title('Sag + Harmonics')

subplot(4, 4, 8);
plot(swell_harmonics);
title('Swell + Harmonics')

subplot(4, 4, 9);
plot(flicker);
title('Flicker')

subplot(4, 4, 10);
plot(notch);
title('Notch')

subplot(4, 4, 11);
plot(spike);
title('Spike')

subplot(4, 4, 12);
plot(flicker_harmonics);
title('Flicker + Harmonics')

subplot(4, 4, 13);
plot(interruption_harmonics);
title('Interruption + Harmonics')

subplot(4, 4, 14);
plot(flicker_sag);
title('Flicker + Sag')

subplot(4, 4, 15);
plot(flicker_swell);
title('Flicker + Swell')

subplot(4, 4, 16);
plot(impulsive_transient);
title('Impulsive Transient')

%%
fs = 32000; % Sampling frequency is now 32[kHz] instead of 3.2[kHz]
freq_limits = [100 2000]; % The same as in the paper
voices_per_octave = 48; % The same as in the paper
signal_length = size(normal, 2);
fb = cwtfilterbank("SignalLength", signal_length, "Wavelet", "Morse", ...
    "VoicesPerOctave", voices_per_octave, "SamplingFrequency", fs, ...
    "FrequencyLimits", freq_limits);

% Compute CWT
normal_cwt = fb.wt(normal);
sag_cwt = fb.wt(sag);
swell_cwt = fb.wt(swell);
interruption_cwt = fb.wt(interruption);
harmonics_cwt = fb.wt(harmonics);
oscillatory_transient_cwt = fb.wt(oscillatory_transient);
sag_harmonics_cwt = fb.wt(sag_harmonics);
swell_harmonics_cwt = fb.wt(swell_harmonics);
flicker_cwt = fb.wt(flicker);
notch_cwt = fb.wt(notch);
spike_cwt = fb.wt(spike);
flicker_harmonics_cwt = fb.wt(flicker_harmonics);
interruption_harmonics_cwt = fb.wt(interruption_harmonics);
flicker_sag_cwt = fb.wt(flicker_sag);
flicker_swell_cwt = fb.wt(flicker_swell);
impulsive_transient_cwt = fb.wt(impulsive_transient);

%%
% Defince time and frequency axis
time = (0:signal_length-1) / fs;
frequencies = fb.BPfrequencies;

% Plot CWT results
figure(2)

subplot(4, 4, 1)
imagesc(time, frequencies, abs(normal_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Normal');


subplot(4, 4, 2)
imagesc(time, frequencies, abs(sag_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Sag')

subplot(4, 4, 3)
imagesc(time, frequencies, abs(swell_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Swell')

subplot(4, 4, 4)
imagesc(time, frequencies, abs(interruption_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Interruption')

subplot(4, 4, 5)
imagesc(time, frequencies, abs(harmonics_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Harmonics')

subplot(4, 4, 6)
imagesc(time, frequencies, abs(oscillatory_transient_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Oscillattory transient')

subplot(4, 4, 7)
imagesc(time, frequencies, abs(sag_harmonics_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Sag + Harmonics')

subplot(4, 4, 8)
imagesc(time, frequencies, abs(swell+harmonics_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Swell + Harmonics')

subplot(4, 4, 9)
imagesc(time, frequencies, abs(flicker_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Flicker')

subplot(4, 4, 10)
imagesc(time, frequencies, abs(notch_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Notch')

subplot(4, 4, 11)
imagesc(time, frequencies, abs(spike_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Spike')

subplot(4, 4, 12)
imagesc(time, frequencies, abs(flicker_harmonics_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Flicker + Harmonics')

subplot(4, 4, 13)
imagesc(time, frequencies, abs(interruption_harmonics_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Interruption + Harmonics')

subplot(4, 4, 14)
imagesc(time, frequencies, abs(flicker_sag_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Flicker + Sag')

subplot(4, 4, 15)
imagesc(time, frequencies, abs(flicker_swell_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Flicker + Swell')

subplot(4, 4, 16)
imagesc(time, frequencies, abs(impulsive_transient_cwt));
set(gca, 'YDir', 'normal');  % Flip the Y-axis direction for better visualization
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Impulsive transient')

impixelinfo


